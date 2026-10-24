"""
AirTeach AI — Interactive Earth Sciences & Geophysics: Seismic Waves,
P-Wave / S-Wave Travel-Time Curves & Earthquake Epicenter Triangulation Simulator.

Simulates:
  - Body waves propagation:
      P-wave (Primary, longitudinal/compressional): v_p = sqrt((K + 4/3*mu)/rho)
      S-wave (Secondary, transverse/shear): v_s = sqrt(mu/rho) (cannot traverse liquids where mu=0)
  - Travel-time curves & S-P interval time lag:
      Δt_{S-P} = t_s - t_p = d * (1/v_s - 1/v_p) = d * (v_p - v_s) / (v_p * v_s)
      d = Δt_{S-P} * (v_p * v_s) / (v_p - v_s)
  - 3-Station Epicenter Triangulation:
      (x - x_i)² + (y - y_i)² = d_i²  (for stations A, B, C)
  - Synthetic seismogram wave trace:
      Ambient noise -> P-wave arrival -> S-wave arrival -> Surface waves (Rayleigh/Love)
  - Richter Magnitude scale estimation:
      M_L = log10(A_max) + 3*log10(8 * Δt_{S-P}) - 2.92
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QPainterPath,
    QLinearGradient,
    QRadialGradient,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class SeismicStation:
    """Seismic recording station coordinate and metadata."""
    name: str
    label: str
    x_km: float
    y_km: float
    color: QColor


@dataclass(frozen=True)
class SeismicPreset:
    """Represents a curriculum scenario for Geophysics & Seismology."""
    code: str
    title_vi: str
    badge_label: str
    event_name: str
    epicenter_x_km: float
    epicenter_y_km: float
    magnitude_mw: float
    p_wave_speed_km_s: float     # v_p in km/s
    s_wave_speed_km_s: float     # v_s in km/s
    depth_km: float
    description: str


SEISMIC_PRESETS: dict[str, SeismicPreset] = {
    "TOHOKU_2011_MEGATHRUST": SeismicPreset(
        code="TOHOKU_2011_MEGATHRUST",
        title_vi="Siêu Động Đất Tohoku 2011 — Đới Hút Chìm Rãnh Nhật Bản (Megathrust)",
        badge_label="TOHOKU M9.1",
        event_name="Trận động đất Tohoku-Oki 2011",
        epicenter_x_km=140.0,
        epicenter_y_km=-60.0,
        magnitude_mw=9.1,
        p_wave_speed_km_s=7.6,
        s_wave_speed_km_s=4.3,
        depth_km=29.0,
        description="Động đất đứt gãy nghịch đới hút chìm Thái Bình Dương. Vận tốc sóng dọc P đạt 7.6 km/s đến các trạm quan sát trước sóng ngang S mang năng lượng phá hủy lớn.",
    ),
    "SAN_ANDREAS_TRANSVERSE_SLIP": SeismicPreset(
        code="SAN_ANDREAS_TRANSVERSE_SLIP",
        title_vi="Đứt Gãy Trượt Bằng San Andreas — Động Đất Nông Vỏ Lục Địa",
        badge_label="SAN ANDREAS M7.1",
        event_name="Động đất Ridgecrest / San Andreas",
        epicenter_x_km=45.0,
        epicenter_y_km=85.0,
        magnitude_mw=7.1,
        p_wave_speed_km_s=6.0,
        s_wave_speed_km_s=3.5,
        depth_km=8.0,
        description="Động đất trượt ngang nông đặc trưng dọc đứt gãy chuyển dạng San Andreas (California). Khoảng cách trạm gần cho phép quan sát rõ nét thời gian trễ S-P chỉ vài giây.",
    ),
    "HIMALAYAN_COLLISION_DEEP": SeismicPreset(
        code="HIMALAYAN_COLLISION_DEEP",
        title_vi="Đới Va Chạm Lục Địa Himalaya — Vỏ Trái Đất Dày 70km",
        badge_label="HIMALAYA M7.8",
        event_name="Động đất Gorkha Nepal",
        epicenter_x_km=-90.0,
        epicenter_y_km=110.0,
        magnitude_mw=7.8,
        p_wave_speed_km_s=6.8,
        s_wave_speed_km_s=3.9,
        depth_km=15.0,
        description="Va chạm giữa mảng Ấn Độ và mảng Á-Âu tạo nên cấu trúc vỏ lục địa dày gấp đôi bình thường, làm tăng tán xạ sóng địa chấn và kéo dài dư chấn địa phương.",
    ),
    "MOONQUAKE_APOLLO_SEISMIC": SeismicPreset(
        code="MOONQUAKE_APOLLO_SEISMIC",
        title_vi="Động Đất Mặt Trăng (Moonquake) — Mạng Địa Chấn Apollo",
        badge_label="APOLLO MOONQUAKE",
        event_name="Chấn động sâu Mặt Trăng Apollo 16",
        epicenter_x_km=-120.0,
        epicenter_y_km=-80.0,
        magnitude_mw=4.5,
        p_wave_speed_km_s=8.0,
        s_wave_speed_km_s=4.6,
        depth_km=800.0,
        description="Động đất Mặt Trăng sâu do lực thủy triều từ Trái Đất. Không có nước làm suy giảm nên sóng địa chấn dao động cộng hưởng kéo dài suốt hàng giờ đồng hồ.",
    ),
    "EARTH_OUTER_CORE_SHADOW_ZONE": SeismicPreset(
        code="EARTH_OUTER_CORE_SHADOW_ZONE",
        title_vi="Vùng Tối Sóng S (Shadow Zone) — Chứng Minh Lõi Ngoài Trái Đất Là Chất Lỏng",
        badge_label="OUTER CORE LIQUID",
        event_name="Quan sát Sóng Địa Chấn Xuyên Lõi",
        epicenter_x_km=0.0,
        epicenter_y_km=0.0,
        magnitude_mw=8.2,
        p_wave_speed_km_s=8.1,
        s_wave_speed_km_s=4.5,
        depth_km=35.0,
        description="Khám phá kinh điển của Inge Lehmann và Richard Oldham: sóng ngang S không thể truyền qua chất lỏng (μ = 0) tạo ra vùng tối 104° - 180° chứng minh lõi ngoài Trái Đất là sắt nóng chảy lỏng.",
    ),
}

# 3 Default Seismic Recording Stations for Triangulation
DEFAULT_STATIONS: list[SeismicStation] = [
    SeismicStation("Trạm Tokyo (A)", "STA_A", -150.0, -100.0, QColor("#38bdf8")),
    SeismicStation("Trạm Sendai (B)", "STA_B", 180.0, -120.0, QColor("#10b981")),
    SeismicStation("Trạm Sapporo (C)", "STA_C", 50.0, 160.0, QColor("#f59e0b")),
]


class InteractiveSeismicWavesItem(QGraphicsObject):
    """
    Interactive PySide6 Graphics Item simulating Seismic Waves,
    Travel-Time curves, S-P lag times, and 3-station Epicenter Triangulation.
    """

    seismic_computed = Signal(float, float, float)  # dist_A_km, sp_lag_s, mag
    stamp_requested = Signal(dict)

    CARD_W = 740.0
    CARD_H = 650.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "TOHOKU_2011_MEGATHRUST"
        self._preset: SeismicPreset = SEISMIC_PRESETS[self._active_preset_code]

        self._epicenter_x: float = self._preset.epicenter_x_km
        self._epicenter_y: float = self._preset.epicenter_y_km
        self._vp: float = self._preset.p_wave_speed_km_s
        self._vs: float = self._preset.s_wave_speed_km_s
        self._magnitude: float = self._preset.magnitude_mw
        self._depth_km: float = self._preset.depth_km

        # Wavefront animation expansion
        self._wave_phase_km: float = 0.0
        self._is_running: bool = True

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Timer for 60fps real-time wave animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Physical & Mathematical Computations (Geophysics & Seismology)
    # -------------------------------------------------------------------------
    def station_distance_km(self, st: SeismicStation) -> float:
        """Computes epicentral distance to station: d = sqrt((x - x_st)² + (y - y_st)² + depth²)."""
        dx = self._epicenter_x - st.x_km
        dy = self._epicenter_y - st.y_km
        return math.sqrt(dx * dx + dy * dy + self._depth_km * self._depth_km)

    def p_arrival_time_s(self, st: SeismicStation) -> float:
        """P-wave travel time t_p = d / v_p."""
        d = self.station_distance_km(st)
        return d / self._vp

    def s_arrival_time_s(self, st: SeismicStation) -> float:
        """S-wave travel time t_s = d / v_s."""
        d = self.station_distance_km(st)
        return d / self._vs

    def sp_time_lag_s(self, st: SeismicStation) -> float:
        """S-P arrival time interval: Δt_{S-P} = t_s - t_p."""
        return self.s_arrival_time_s(st) - self.p_arrival_time_s(st)

    @property
    def omori_factor_km_per_s(self) -> float:
        """Omori velocity conversion coefficient k = (v_p * v_s) / (v_p - v_s)."""
        diff = max(0.1, self._vp - self._vs)
        return (self._vp * self._vs) / diff

    # -------------------------------------------------------------------------
    # Controls & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in SEISMIC_PRESETS:
            return
        self._active_preset_code = code
        self._preset = SEISMIC_PRESETS[code]
        self._epicenter_x = self._preset.epicenter_x_km
        self._epicenter_y = self._preset.epicenter_y_km
        self._vp = self._preset.p_wave_speed_km_s
        self._vs = self._preset.s_wave_speed_km_s
        self._magnitude = self._preset.magnitude_mw
        self._depth_km = self._preset.depth_km
        self._wave_phase_km = 0.0

        st_a = DEFAULT_STATIONS[0]
        self.seismic_computed.emit(
            self.station_distance_km(st_a),
            self.sp_time_lag_s(st_a),
            self._magnitude,
        )
        self.update()

    def set_epicenter(self, x_km: float, y_km: float) -> None:
        """Sets the earthquake epicenter coordinates in km [-200, 200]."""
        self._epicenter_x = max(-200.0, min(200.0, x_km))
        self._epicenter_y = max(-200.0, min(200.0, y_km))
        st_a = DEFAULT_STATIONS[0]
        self.seismic_computed.emit(
            self.station_distance_km(st_a),
            self.sp_time_lag_s(st_a),
            self._magnitude,
        )
        self.update()

    def set_wave_speeds(self, vp: float, vs: float) -> None:
        """Sets P-wave and S-wave speeds in km/s."""
        self._vp = max(4.0, min(12.0, vp))
        self._vs = max(2.0, min(self._vp - 0.5, vs))
        st_a = DEFAULT_STATIONS[0]
        self.seismic_computed.emit(
            self.station_distance_km(st_a),
            self.sp_time_lag_s(st_a),
            self._magnitude,
        )
        self.update()

    def reset_defaults(self) -> None:
        """Resets parameters to active preset defaults."""
        self._epicenter_x = self._preset.epicenter_x_km
        self._epicenter_y = self._preset.epicenter_y_km
        self._vp = self._preset.p_wave_speed_km_s
        self._vs = self._preset.s_wave_speed_km_s
        self._magnitude = self._preset.magnitude_mw
        self._depth_km = self._preset.depth_km
        self._wave_phase_km = 0.0
        st_a = DEFAULT_STATIONS[0]
        self.seismic_computed.emit(
            self.station_distance_km(st_a),
            self.sp_time_lag_s(st_a),
            self._magnitude,
        )
        self.update()

    def _on_sim_tick(self) -> None:
        """Animates outward propagating circular seismic wavefronts."""
        if not self._is_running:
            return
        self._wave_phase_km += self._vp * 0.8
        if self._wave_phase_km > 380.0:
            self._wave_phase_km = 0.0
        self.update()

    # -------------------------------------------------------------------------
    # Layout Buttons
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Defines touch button bounding boxes at the bottom."""
        by = self.CARD_H - 46.0
        bw = 88.0
        bh = 32.0
        spacing = 10.0
        start_x = 18.0

        keys = [
            ("btn_preset", 130.0),
            ("btn_x_down", 68.0),
            ("btn_x_up", 68.0),
            ("btn_y_down", 68.0),
            ("btn_y_up", 68.0),
            ("btn_vp_down", 72.0),
            ("btn_vp_up", 72.0),
            ("btn_reset", 66.0),
            ("btn_stamp", 112.0),
        ]

        curr_x = start_x
        for key, width in keys:
            self._buttons[key] = QRectF(curr_x, by, width, bh)
            curr_x += width + spacing

    # -------------------------------------------------------------------------
    # Rendering Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Card Container
        card_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor("#080c14"))
        grad.setColorAt(1.0, QColor("#0f172a"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header
        self._paint_header(painter)

        # 3. Left Panel: 3-Station Triangulation Map
        self._paint_triangulation_map(painter)

        # 4. Right Panel: Synthetic Seismogram & Travel-Time Curves
        self._paint_seismogram_and_curves(painter)

        # 5. Bottom Metrics & Omori Formula
        self._paint_metrics_panel(painter)

        # 6. Bottom Touchless Buttons
        self._paint_buttons(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Badge
        badge_rect = QRectF(20.0, 16.0, 185.0, 24.0)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"🌍 {self._preset.badge_label}")

        # Title
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(218.0, 33.0, self._preset.title_vi)

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            22.0,
            58.0,
            f"Sự kiện: {self._preset.event_name} (M_w = {self._magnitude:.1f}, Độ sâu h = {self._depth_km:.1f} km) | Vận tốc: vp = {self._vp:.1f} km/s, vs = {self._vs:.1f} km/s",
        )

    def _paint_triangulation_map(self, painter: QPainter) -> None:
        """Renders 2D geographic coordinate map with 3 seismic stations, distance circles, and epicenter."""
        px = 22.0
        py = 74.0
        pw = 350.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#050811")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "1. Bản Đồ Định Vị Tâm Chấn (3 Trạm Địa Chấn)")

        # Coordinate transform: Map [-220, 220] km to pixels
        map_cx = px + pw / 2.0
        map_cy = py + ph / 2.0 + 8.0
        scale = (pw - 40.0) / 440.0  # px per km

        def km_to_px(x_km: float, y_km: float) -> tuple[float, float]:
            return map_cx + x_km * scale, map_cy - y_km * scale

        # Grid lines (every 100 km)
        grid_pen = QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for g_km in [-200.0, -100.0, 0.0, 100.0, 200.0]:
            gx, _ = km_to_px(g_km, 0.0)
            painter.drawLine(QPointF(gx, py + 28.0), QPointF(gx, py + ph - 8.0))
            _, gy = km_to_px(0.0, g_km)
            painter.drawLine(QPointF(px + 10.0, gy), QPointF(px + pw - 10.0, gy))

        # Origin axes crosshair
        ox_px, oy_px = km_to_px(0.0, 0.0)
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawLine(QPointF(ox_px, py + 28.0), QPointF(ox_px, py + ph - 8.0))
        painter.drawLine(QPointF(px + 10.0, oy_px), QPointF(px + pw - 10.0, oy_px))

        # Propagating P-wave and S-wave circular ripples from epicenter
        ep_px, ep_py = km_to_px(self._epicenter_x, self._epicenter_y)
        p_r_px = self._wave_phase_km * scale
        s_r_px = (self._wave_phase_km * (self._vs / self._vp)) * scale

        # P-wave wavefront (Cyan)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(56, 189, 248, 120), 1.5))
        painter.drawEllipse(QPointF(ep_px, ep_py), p_r_px, p_r_px)

        # S-wave wavefront (Red/Orange)
        painter.setPen(QPen(QColor(244, 63, 94, 160), 2.0))
        painter.drawEllipse(QPointF(ep_px, ep_py), s_r_px, s_r_px)

        # 3 Triangulation Station Circles
        for st in DEFAULT_STATIONS:
            st_px, st_py = km_to_px(st.x_km, st.y_km)
            dist_km = self.station_distance_km(st)
            r_circ_px = dist_km * scale

            # Draw Triangulation Circle
            circ_pen = QPen(st.color, 1.2, Qt.PenStyle.DashLine)
            painter.setPen(circ_pen)
            painter.drawEllipse(QPointF(st_px, st_py), r_circ_px, r_circ_px)

            # Draw Station Triangle marker
            painter.setBrush(QBrush(st.color))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            tri = QPainterPath()
            tri.moveTo(st_px, st_py - 6.0)
            tri.lineTo(st_px - 5.0, st_py + 4.0)
            tri.lineTo(st_px + 5.0, st_py + 4.0)
            tri.closeSubpath()
            painter.drawPath(tri)

            # Station label
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QPen(st.color))
            painter.drawText(st_px + 6.0, st_py - 4.0, st.label)

        # Epicenter marker (Flashing Red Star / Cross)
        painter.setBrush(QBrush(QColor("#f43f5e")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(QPointF(ep_px, ep_py), 5.5, 5.5)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(ep_px - 32.0, ep_py - 10.0, "TÂM CHẤN (Epicenter)")

    def _paint_seismogram_and_curves(self, painter: QPainter) -> None:
        """Renders synthetic 3-phase seismogram trace and S-P time lag indicator."""
        px = 384.0
        py = 74.0
        pw = 334.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#050811")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "2. Băng Địa Chấn Ký (Seismogram) & Độ Trễ S-P")

        # Select Station A for waveform demonstration
        st = DEFAULT_STATIONS[0]
        tp = self.p_arrival_time_s(st)
        ts = self.s_arrival_time_s(st)
        sp_lag = ts - tp

        # Trace viewport coordinates
        t_x0 = px + 28.0
        t_y0 = py + 75.0
        t_w = pw - 44.0
        t_h = 90.0
        cy = t_y0 + t_h / 2.0

        # Time axis scaling (0 to max(60s, ts * 1.4))
        max_time_s = max(60.0, ts * 1.35)

        # Centerline for trace
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawLine(QPointF(t_x0, cy), QPointF(t_x0 + t_w, cy))

        # Synthetic Seismogram Path
        trace_path = QPainterPath()
        num_trace_pts = 120
        random.seed(42)  # Deterministic seed for authentic stable waveform

        for i in range(num_trace_pts + 1):
            cur_t = (i / num_trace_pts) * max_time_s
            tx = t_x0 + (cur_t / max_time_s) * t_w

            if cur_t < tp:
                # Ambient micro-seismic background noise
                amp = (random.random() - 0.5) * 3.0
            elif cur_t < ts:
                # P-wave (high frequency, moderate amplitude)
                phase = (cur_t - tp) * 18.0
                amp = math.sin(phase) * 12.0 + (random.random() - 0.5) * 4.0
            else:
                # S-wave arrival & Surface waves (large amplitude, lower frequency)
                phase_s = (cur_t - ts) * 8.0
                envelope = math.exp(-(cur_t - ts) * 0.04)
                amp = math.sin(phase_s) * 32.0 * envelope + (random.random() - 0.5) * 6.0

            ty = cy + amp
            if i == 0:
                trace_path.moveTo(tx, ty)
            else:
                trace_path.lineTo(tx, ty)

        # Draw waveform
        painter.setPen(QPen(QColor("#38bdf8"), 1.5))
        painter.drawPath(trace_path)

        # P-wave arrival vertical marker
        p_x = t_x0 + (tp / max_time_s) * t_w
        painter.setPen(QPen(QColor("#10b981"), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(p_x, t_y0 - 8.0), QPointF(p_x, t_y0 + t_h + 8.0))

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(p_x - 14.0, t_y0 - 12.0, f"P (t={tp:.1f}s)")

        # S-wave arrival vertical marker
        s_x = t_x0 + (ts / max_time_s) * t_w
        painter.setPen(QPen(QColor("#f43f5e"), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(s_x, t_y0 - 8.0), QPointF(s_x, t_y0 + t_h + 8.0))

        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(s_x - 14.0, t_y0 - 12.0, f"S (t={ts:.1f}s)")

        # S-P Interval span indicator
        painter.setPen(QPen(QColor("#fbbf24"), 1.5))
        sp_y = t_y0 + t_h + 16.0
        painter.drawLine(QPointF(p_x, sp_y), QPointF(s_x, sp_y))
        painter.drawLine(QPointF(p_x, sp_y - 4.0), QPointF(p_x, sp_y + 4.0))
        painter.drawLine(QPointF(s_x, sp_y - 4.0), QPointF(s_x, sp_y + 4.0))

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText((p_x + s_x) / 2.0 - 24.0, sp_y + 14.0, f"Δt={sp_lag:.1f}s")

        # Bottom Formula Tag
        f_box = QRectF(t_x0, py + ph - 34.0, t_w, 22.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(f_box, 4.0, 4.0)

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#38bdf8")))
        d_val = self.station_distance_km(st)
        painter.drawText(f_box, Qt.AlignmentFlag.AlignCenter, f"d = Δt_SP · [vp·vs / (vp - vs)] = {d_val:.1f} km")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Renders live quantitative parameters, Omori distance law, and station metrics."""
        px = 22.0
        py = 336.0
        pw = self.CARD_W - 44.0
        ph = 142.0

        # Background container
        painter.setBrush(QBrush(QColor("#09101c")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Header of metrics
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 14.0, py + 20.0, "3. Thông Số Vật Lý Địa Chấn & Phương Trình Khoảng Cách Omori: d = k · Δt_{S-P}")

        # Column 1: Epicenter & Wave Speeds
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#e2e8f0")))
        col1_x = px + 14.0
        line1_y = py + 44.0
        dy = 22.0

        painter.drawText(col1_x, line1_y, f"• Tọa độ tâm chấn: X = {self._epicenter_x:+.1f} km, Y = {self._epicenter_y:+.1f} km")
        painter.drawText(col1_x, line1_y + dy, f"• Độ lớn chấn động: M_w = {self._magnitude:.1f} (Độ sâu chấn tiêu h = {self._depth_km:.1f} km)")
        painter.drawText(col1_x, line1_y + 2 * dy, f"• Vận tốc sóng P: vp = {self._vp:.2f} km/s (Sóng dọc, truyền qua rắn và lỏng)")
        painter.drawText(col1_x, line1_y + 3 * dy, f"• Vận tốc sóng S: vs = {self._vs:.2f} km/s (Sóng ngang, chỉ truyền qua chất rắn)")

        # Column 2: 3-Station Distance & Lag Readouts
        col2_x = px + 360.0
        st_a, st_b, st_c = DEFAULT_STATIONS
        dA, dta = self.station_distance_km(st_a), self.sp_time_lag_s(st_a)
        dB, dtb = self.station_distance_km(st_b), self.sp_time_lag_s(st_b)
        dC, dtc = self.station_distance_km(st_c), self.sp_time_lag_s(st_c)

        painter.drawText(col2_x, line1_y, f"• Trạm A (Tokyo): dA = {dA:.1f} km  ➔  Δt_SP = {dta:.1f} s")
        painter.drawText(col2_x, line1_y + dy, f"• Trạm B (Sendai): dB = {dB:.1f} km  ➔  Δt_SP = {dtb:.1f} s")
        painter.drawText(col2_x, line1_y + 2 * dy, f"• Trạm C (Sapporo): dC = {dC:.1f} km  ➔  Δt_SP = {dtc:.1f} s")
        painter.drawText(col2_x, line1_y + 3 * dy, f"• Hệ số Omori: k = (vp·vs)/(vp-vs) = {self.omori_factor_km_per_s:.2f} km/s")

        # Explanatory Educational Text at bottom
        desc_rect = QRectF(px + 12.0, py + ph + 8.0, pw - 24.0, 95.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive touch/mouse control buttons."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        labels = {
            "btn_preset": f"📋 Kịch Bản ({self._preset.badge_label})",
            "btn_x_down": "X - 20km",
            "btn_x_up": "X + 20km",
            "btn_y_down": "Y - 20km",
            "btn_y_up": "Y + 20km",
            "btn_vp_down": "vp - 0.5",
            "btn_vp_up": "vp + 0.5",
            "btn_reset": "🔄 Đặt Lại",
            "btn_stamp": "📌 Đóng Dấu Bảng",
        }

        for key, rect in self._buttons.items():
            is_hover = (key == self._hovered_btn)
            text = labels.get(key, key)

            if key == "btn_stamp":
                bg = QColor("#0284c7") if not is_hover else QColor("#0369a1")
                border = QColor("#38bdf8")
            elif key == "btn_preset":
                bg = QColor("#4338ca") if not is_hover else QColor("#3730a3")
                border = QColor("#818cf8")
            elif "up" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#38bdf8")
            elif "down" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#f43f5e")
            else:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#64748b")

            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handling
    # -------------------------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if self._hovered_btn != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._on_button_clicked(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _on_button_clicked(self, key: str) -> None:
        """Dispatches button actions."""
        if key == "btn_preset":
            keys = list(SEISMIC_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_x_down":
            self.set_epicenter(self._epicenter_x - 20.0, self._epicenter_y)
        elif key == "btn_x_up":
            self.set_epicenter(self._epicenter_x + 20.0, self._epicenter_y)
        elif key == "btn_y_down":
            self.set_epicenter(self._epicenter_x, self._epicenter_y - 20.0)
        elif key == "btn_y_up":
            self.set_epicenter(self._epicenter_x, self._epicenter_y + 20.0)
        elif key == "btn_vp_down":
            self.set_wave_speeds(self._vp - 0.5, self._vs)
        elif key == "btn_vp_up":
            self.set_wave_speeds(self._vp + 0.5, self._vs)
        elif key == "btn_reset":
            self.reset_defaults()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        st_a = DEFAULT_STATIONS[0]
        data = {
            "preset_code": self._active_preset_code,
            "event_name": self._preset.event_name,
            "epicenter_x": self._epicenter_x,
            "epicenter_y": self._epicenter_y,
            "magnitude": self._magnitude,
            "vp": self._vp,
            "vs": self._vs,
            "depth_km": self._depth_km,
            "dist_a_km": self.station_distance_km(st_a),
            "sp_lag_a_s": self.sp_time_lag_s(st_a),
        }
        self.stamp_requested.emit(data)
