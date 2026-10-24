"""
AirTeach AI — Interactive Analytical Chemistry: High-Performance Liquid Chromatography (HPLC).
Simulates reversed-phase liquid chromatography, Van Deemter column efficiency equation
H = A + B/u + C·u, Eddy diffusion, longitudinal diffusion, mass transfer resistance,
theoretical plate count N = L/H, backpressure ΔP, retention capacity factor k' = (t_R - t_0)/t_0,
and chromatographic resolution R_s = 2(t_R2 - t_R1)/(W1 + W2).
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
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class AnalytePeak:
    name_vi: str
    formula: str
    k_prime: float         # Capacity factor k'
    color: QColor          # Representative dye color for column visualization
    relative_height: float # Relative UV absorbance height (0.0 to 1.0)


@dataclass(frozen=True)
class HPLCPreset:
    code: str
    title_vi: str
    badge_label: str
    column_length_mm: float      # Column length L in mm (e.g. 150 mm)
    particle_size_um: float      # Packing particle diameter d_p in μm (e.g. 5.0 μm)
    flow_rate_ml_min: float      # Flow rate u in mL/min (e.g. 1.0 mL/min)
    mobile_phase_name: str       # e.g. "65:35 Water / Acetonitrile (0.1% Formic Acid)"
    analytes: list[AnalytePeak]
    description: str


HPLC_PRESETS: dict[str, HPLCPreset] = {
    "REVERSE_PHASE_ANALGESICS": HPLCPreset(
        code="REVERSE_PHASE_ANALGESICS",
        title_vi="Sắc ký Pha Đảo C18 — Phân tách Thuốc Giảm đau (Paracetamol, Cafein, Aspirin)",
        badge_label="ANALGESIC C18",
        column_length_mm=150.0,
        particle_size_um=5.0,
        flow_rate_ml_min=1.0,
        mobile_phase_name="70:30 Nước / Acetonitril (pH 2.8)",
        analytes=[
            AnalytePeak("Paracetamol", "C₈H₉NO₂", 0.85, QColor(56, 189, 248), 0.85),
            AnalytePeak("Cafein", "C₈H₁₀N₄O₂", 2.20, QColor(250, 204, 21), 0.70),
            AnalytePeak("Aspirin", "C₉H₈O₄", 4.60, QColor(244, 63, 94), 0.95),
        ],
        description="Phân tách hỗn hợp giảm đau kinh điển trên cột C18. Phân tử phân cực nhất (Paracetamol) rửa giải trước tiên, phân tử kỵ nước nhất (Aspirin) giữ lại lâu nhất. Độ phân giải R_s > 2.0 đạt chuẩn tách nền hoàn hảo.",
    ),
    "VAN_DEEMTER_UHPLC_SUB2UM": HPLCPreset(
        code="VAN_DEEMTER_UHPLC_SUB2UM",
        title_vi="Sắc ký Siêu Cao áp UHPLC — Hạt siêu mịn Sub-2μm (d_p = 1.7 μm)",
        badge_label="UHPLC SUB-2μm",
        column_length_mm=50.0,
        particle_size_um=1.7,
        flow_rate_ml_min=2.0,
        mobile_phase_name="60:40 Nước / Acetonitril",
        analytes=[
            AnalytePeak("Chất A", "C₁₀H₁₂O", 0.60, QColor(34, 197, 94), 0.90),
            AnalytePeak("Chất B", "C₁₂H₁₆O", 1.40, QColor(168, 85, 247), 0.80),
            AnalytePeak("Chất C", "C₁₄H₂₀O", 2.50, QColor(249, 115, 22), 0.75),
        ],
        description="Hạt nhồi siêu nhỏ 1.7 μm làm phẳng số hạng truyền khối C trong phương trình Van Deemter, cho phép vận hành ở lưu lượng cao (2.0 mL/min) với số đĩa lý thuyết N > 15,000 và áp suất lên tới 600 bar.",
    ),
    "ISOMER_CO_ELUTION_FAILURE": HPLCPreset(
        code="ISOMER_CO_ELUTION_FAILURE",
        title_vi="Hiện tượng Đồng Rửa giải (Co-Elution) — Độ Phân giải Kém (R_s < 1.0)",
        badge_label="CO-ELUTION",
        column_length_mm=100.0,
        particle_size_um=5.0,
        flow_rate_ml_min=1.8,
        mobile_phase_name="50:50 Metanol / Nước",
        analytes=[
            AnalytePeak("Đồng phân Ortho", "C₇H₇Cl", 1.80, QColor(244, 63, 94), 0.85),
            AnalytePeak("Đồng phân Para", "C₇H₇Cl", 1.90, QColor(234, 179, 8), 0.80),
        ],
        description="Hai đồng phân cấu trúc có hệ số chọn lọc α ≈ 1.08 quá gần nhau. Khi lưu lượng vượt quá tối ưu, do hiệu ứng mở rộng pic, hai đỉnh chồng lấn lên nhau (R_s = 0.85 < 1.5) không thể định lượng độc lập.",
    ),
    "GRADIENT_ELUTION_STEROIDS": HPLCPreset(
        code="GRADIENT_ELUTION_STEROIDS",
        title_vi="Rửa giải Gradient Nồng độ — Phân tách 4 Hormon Steroid Đa phân cực",
        badge_label="GRADIENT STEROID",
        column_length_mm=150.0,
        particle_size_um=3.0,
        flow_rate_ml_min=1.2,
        mobile_phase_name="Gradient 20% → 85% ACN / Nước",
        analytes=[
            AnalytePeak("Cortisol", "C₂₁H₃₀O₅", 0.70, QColor(56, 189, 248), 0.85),
            AnalytePeak("Cortisone", "C₂₁H₂₈O₅", 1.50, QColor(34, 197, 94), 0.80),
            AnalytePeak("Testosterone", "C₁₉H₂₈O₂", 2.80, QColor(250, 204, 21), 0.90),
            AnalytePeak("Progesterone", "C₂₁H₃₀O₂", 4.80, QColor(239, 68, 68), 0.75),
        ],
        description="Rửa giải gradient tăng dần tỷ lệ dung môi hữu cơ theo thời gian giúp nén các pic rửa giải muộn (Progesterone), duy trì độ sắc nhọn của đỉnh và rút ngắn tổng thời gian phân tích.",
    ),
    "PHARMACEUTICAL_IMPURITY_LOD": HPLCPreset(
        code="PHARMACEUTICAL_IMPURITY_LOD",
        title_vi="Kiểm nghiệm Tạp chất Thuốc Dược phẩm — Giới hạn Phát hiện LOD & LOQ",
        badge_label="IMPURITY 0.2%",
        column_length_mm=250.0,
        particle_size_um=3.0,
        flow_rate_ml_min=0.8,
        mobile_phase_name="65:35 Đệm Phosphat / ACN",
        analytes=[
            AnalytePeak("Dược chất Chính (API)", "C₁₆H₁₉N₃O₄S", 2.10, QColor(56, 189, 248), 1.00),
            AnalytePeak("Tạp chất Phân hủy (0.2%)", "C₁₄H₁₇NO₂", 3.20, QColor(244, 63, 94), 0.08),
        ],
        description="Phát hiện tạp chất vết ở mức 0.2% so với hoạt chất chính. Cột dài 250 mm cung cấp số đĩa lý thuyết cao vượt trội, đảm bảo pic tạp chất sắc nhọn có tỷ số tín hiệu trên nhiễu SNR > 10 đạt chuẩn định lượng LOQ.",
    ),
}


class InteractiveHPLCItem(QGraphicsObject):
    """
    Interactive High-Performance Liquid Chromatography (HPLC) Simulator.
    Simulates Van Deemter column kinetics H = A + B/u + C·u, theoretical plate count N = L/H,
    backpressure ΔP, column void time t_0, analyte retention capacity factors k',
    chromatographic resolution R_s, and UV absorbance chromatogram peaks.
    """

    preset_changed = Signal(str)
    chromatography_computed = Signal(float, float, float)  # N_plates, pressure_bar, resolution_rs
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0
    COLUMN_ID_MM = 4.6      # Standard analytical column inner diameter
    TOTAL_POROSITY = 0.65   # Total column porosity ε_t
    VISCOSITY_MPA_S = 1.0   # Mobile phase dynamic viscosity η (mPa·s)

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation Preset & Parameters
        self._active_preset_code: str = "REVERSE_PHASE_ANALGESICS"
        self._preset: HPLCPreset = HPLC_PRESETS[self._active_preset_code]

        self._col_length_mm: float = self._preset.column_length_mm
        self._dp_um: float = self._preset.particle_size_um
        self._flow_rate: float = self._preset.flow_rate_ml_min

        # Animation state
        self._sim_time_s: float = 0.0
        self._is_running: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # 30 fps animation timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Chromatography & Van Deemter Physics Equations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> HPLCPreset:
        return self._preset

    @property
    def column_length_mm(self) -> float:
        return self._col_length_mm

    @property
    def particle_size_um(self) -> float:
        return self._dp_um

    @property
    def flow_rate_ml_min(self) -> float:
        return self._flow_rate

    @property
    def linear_velocity_mm_s(self) -> float:
        """
        Calculates interstitial mobile phase linear velocity u (mm/s):
        u = F / (A_cross * ε_t) = (F * 1000 / 60) / (π * (d_c/2)² * ε_t)
        """
        f_mm3_s = (self._flow_rate * 1000.0) / 60.0
        r_mm = self.COLUMN_ID_MM / 2.0
        cross_area_mm2 = math.pi * (r_mm ** 2) * self.TOTAL_POROSITY
        return f_mm3_s / cross_area_mm2

    @property
    def van_deemter_coefficients(self) -> tuple[float, float, float]:
        """
        Calculates Van Deemter coefficients (A, B, C) for plate height H (in μm) vs u (mm/s):
        H = A + B/u + C·u
        - A (Eddy diffusion) ≈ 1.5 * d_p
        - B (Longitudinal diffusion) ≈ 2.0 * γ * D_m (scaled to velocity)
        - C (Mass transfer resistance) ≈ 0.05 * d_p² / D_m
        """
        dp = self._dp_um
        a = 1.5 * dp                                # Eddy diffusion (μm)
        b = 3.5                                     # Longitudinal diffusion (μm * mm/s)
        c = 0.025 * (dp ** 2)                       # Resistance to mass transfer (μm / (mm/s))
        return a, b, c

    @property
    def plate_height_um(self) -> float:
        """Calculates theoretical plate height H (μm) at current linear velocity."""
        u = max(0.05, self.linear_velocity_mm_s)
        a, b, c = self.van_deemter_coefficients
        return a + (b / u) + (c * u)

    @property
    def optimal_linear_velocity_mm_s(self) -> float:
        """Optimal linear velocity u_opt = √(B/C) minimizing plate height."""
        _, b, c = self.van_deemter_coefficients
        return math.sqrt(b / max(1e-5, c))

    @property
    def min_plate_height_um(self) -> float:
        """Minimum plate height H_min = A + 2√(B·C)."""
        a, b, c = self.van_deemter_coefficients
        return a + 2.0 * math.sqrt(b * c)

    @property
    def theoretical_plates_n(self) -> float:
        """Theoretical plate count N = L / H."""
        l_um = self._col_length_mm * 1000.0
        h_um = max(1.0, self.plate_height_um)
        return l_um / h_um

    @property
    def column_backpressure_bar(self) -> float:
        """
        Calculates column hydraulic backpressure ΔP (bar) using Kozeny-Carman Darcy equation:
        ΔP = (Φ * η * L * u) / d_p²
        """
        u_mm_s = self.linear_velocity_mm_s
        l_mm = self._col_length_mm
        dp_um = self._dp_um
        # Empirical proportionality constant for packed spherical C18 silica
        k_factor = 280.0
        pressure_bar = (k_factor * self.VISCOSITY_MPA_S * (l_mm / 100.0) * u_mm_s) / (dp_um ** 2)
        return max(5.0, pressure_bar)

    @property
    def column_void_time_s(self) -> float:
        """Column void time t_0 = L / u in seconds."""
        u = max(0.1, self.linear_velocity_mm_s)
        return self._col_length_mm / u

    def calc_retention_time_s(self, k_prime: float) -> float:
        """Calculates retention time t_R = t_0 * (1 + k')."""
        return self.column_void_time_s * (1.0 + k_prime)

    def calc_peak_width_s(self, k_prime: float) -> float:
        """Calculates baseline peak width W = 4 * t_R / √N."""
        t_r = self.calc_retention_time_s(k_prime)
        n = max(100.0, self.theoretical_plates_n)
        return (4.0 * t_r) / math.sqrt(n)

    @property
    def chromatographic_resolution_rs(self) -> float:
        """
        Calculates resolution R_s between the first two adjacent analyte peaks:
        R_s = 2(t_R2 - t_R1) / (W1 + W2).
        """
        analytes = self._preset.analytes
        if len(analytes) < 2:
            return 2.5
        k1 = analytes[0].k_prime
        k2 = analytes[1].k_prime
        t1 = self.calc_retention_time_s(k1)
        t2 = self.calc_retention_time_s(k2)
        w1 = self.calc_peak_width_s(k1)
        w2 = self.calc_peak_width_s(k2)
        return (2.0 * abs(t2 - t1)) / max(0.1, w1 + w2)

    # -------------------------------------------------------------------------
    # Animation & Parameter Controls
    # -------------------------------------------------------------------------
    def _on_sim_tick(self) -> None:
        if self._is_running:
            self._sim_time_s += 0.05
        self.update()

    def load_preset(self, preset_code: str) -> None:
        """Loads a curriculum HPLC preset."""
        if preset_code not in HPLC_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = HPLC_PRESETS[preset_code]
        self._col_length_mm = self._preset.column_length_mm
        self._dp_um = self._preset.particle_size_um
        self._flow_rate = self._preset.flow_rate_ml_min
        self._sim_time_s = 0.0

        self.preset_changed.emit(preset_code)
        self.chromatography_computed.emit(
            self.theoretical_plates_n, self.column_backpressure_bar, self.chromatographic_resolution_rs
        )
        self.update()

    def step_flow_rate(self, delta_ml: float) -> None:
        """Steps mobile phase flow rate in mL/min."""
        self._flow_rate = max(0.2, min(3.5, round(self._flow_rate + delta_ml, 2)))
        self.chromatography_computed.emit(
            self.theoretical_plates_n, self.column_backpressure_bar, self.chromatographic_resolution_rs
        )
        self.update()

    def step_particle_size(self, delta_um: float) -> None:
        """Steps packing particle size d_p in μm."""
        self._dp_um = max(1.5, min(10.0, round(self._dp_um + delta_um, 1)))
        self.chromatography_computed.emit(
            self.theoretical_plates_n, self.column_backpressure_bar, self.chromatographic_resolution_rs
        )
        self.update()

    def toggle_column_length(self) -> None:
        """Cycles column length through standard analytical lengths (50, 100, 150, 250 mm)."""
        lengths = [50.0, 100.0, 150.0, 250.0]
        curr_idx = lengths.index(self._col_length_mm) if self._col_length_mm in lengths else 2
        next_idx = (curr_idx + 1) % len(lengths)
        self._col_length_mm = lengths[next_idx]
        self.chromatography_computed.emit(
            self.theoretical_plates_n, self.column_backpressure_bar, self.chromatographic_resolution_rs
        )
        self.update()

    def toggle_animation(self) -> None:
        """Toggles real-time chromatographic flow animation."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Gestures
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_f_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_f_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_dp_down"] = QRectF(290.0, bar_y, 45.0, 34.0)
        self._buttons["btn_dp_up"] = QRectF(340.0, bar_y, 45.0, 34.0)
        self._buttons["btn_col_l"] = QRectF(392.0, bar_y, 65.0, 34.0)
        self._buttons["btn_p_analgesic"] = QRectF(464.0, bar_y, 85.0, 34.0)
        self._buttons["btn_p_uhplc"] = QRectF(556.0, bar_y, 80.0, 34.0)
        self._buttons["btn_p_coelute"] = QRectF(643.0, bar_y, 85.0, 34.0)
        self._buttons["btn_p_impurity"] = QRectF(735.0, bar_y, 80.0, 34.0)
        self._buttons["btn_play"] = QRectF(822.0, bar_y, 55.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(884.0, bar_y, 46.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(HPLC_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_f_down":
            self.step_flow_rate(-0.2)
        elif key == "btn_f_up":
            self.step_flow_rate(0.2)
        elif key == "btn_dp_down":
            self.step_particle_size(-0.5)
        elif key == "btn_dp_up":
            self.step_particle_size(0.5)
        elif key == "btn_col_l":
            self.toggle_column_length()
        elif key == "btn_p_analgesic":
            self.load_preset("REVERSE_PHASE_ANALGESICS")
        elif key == "btn_p_uhplc":
            self.load_preset("VAN_DEEMTER_UHPLC_SUB2UM")
        elif key == "btn_p_coelute":
            self.load_preset("ISOMER_CO_ELUTION_FAILURE")
        elif key == "btn_p_impurity":
            self.load_preset("PHARMACEUTICAL_IMPURITY_LOD")
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_stamp":
            self.stamp_diagram()
        self.update()

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
        """Emits signal requesting permanent vector stamping of current diagram."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "col_length_mm": self._col_length_mm,
            "dp_um": self._dp_um,
            "flow_rate": self._flow_rate,
            "plate_height_um": self.plate_height_um,
            "plates_n": self.theoretical_plates_n,
            "backpressure_bar": self.column_backpressure_bar,
            "resolution_rs": self.chromatographic_resolution_rs,
            "void_time_s": self.column_void_time_s,
        }
        self.stamp_requested.emit(stamp_data)

    # -------------------------------------------------------------------------
    # QGraphicsItem Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_column_diagram_panel(painter)
        self._paint_chromatogram_panel(painter)
        self._paint_van_deemter_panel(painter)
        self._paint_metrics_panel(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(9, 14, 25, 252))
        grad.setColorAt(1.0, QColor(5, 8, 16, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"SẮC KÝ LỎNG HIỆU NĂNG CAO HPLC & HIỆU SUẤT CỘT VAN DEEMTER: {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(
            22,
            48,
            f"Cột C18 L = {self._col_length_mm:.0f} mm | Hạt nhồi d_p = {self._dp_um:.1f} μm | Lưu lượng F = {self._flow_rate:.2f} mL/min | Áp suất ΔP = {self.column_backpressure_bar:.0f} bar | Đĩa lý thuyết N = {self.theoretical_plates_n:.0f} | R_s = {self.chromatographic_resolution_rs:.2f}",
        )

    def _paint_column_diagram_panel(self, painter: QPainter) -> None:
        """Renders stainless steel column cross-section with analyte migration bands."""
        cx = 20.0
        cy = 60.0
        cw = 440.0
        ch = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(cx, cy, cw, ch, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(cx + 12.0, cy + 20.0, "MÔ HÌNH CỘT SẮC KÝ & SỰ DI CHUYỂN CÁC DẢI CHẤT:")

        # Stainless steel column tube
        col_x = cx + 45.0
        col_y = cy + 60.0
        col_w = cw - 90.0
        col_h = 44.0

        col_grad = QLinearGradient(col_x, col_y, col_x, col_y + col_h)
        col_grad.setColorAt(0.0, QColor(71, 85, 105))
        col_grad.setColorAt(0.5, QColor(30, 41, 59))
        col_grad.setColorAt(1.0, QColor(15, 23, 42))
        painter.setBrush(QBrush(col_grad))
        painter.setPen(QPen(QColor(148, 163, 184), 1.5))
        painter.drawRoundedRect(col_x, col_y, col_w, col_h, 6.0)

        # End fittings
        painter.setBrush(QBrush(QColor(100, 116, 139)))
        painter.drawRect(col_x - 12.0, col_y + 6.0, 12.0, col_h - 12.0)
        painter.drawRect(col_x + col_w, col_y + 6.0, 12.0, col_h - 12.0)

        # Inlet & outlet tubing
        painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        painter.drawLine(cx + 10.0, col_y + col_h / 2.0, col_x - 12.0, col_y + col_h / 2.0)
        painter.drawLine(col_x + col_w + 12.0, col_y + col_h / 2.0, cx + cw - 10.0, col_y + col_h / 2.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(cx + 12.0, col_y + 14.0, "Vào: Bơm Cao áp")
        painter.drawText(cx + cw - 70.0, col_y + 14.0, "Ra: Đầu dò UV")

        # Live migration of analyte bands inside column
        # Progress of analyte band i along column: frac_i = (t / t_Ri) % 1.2
        cycle_period = 10.0
        sim_phase = (self._sim_time_s % cycle_period) / cycle_period

        for analyte in self._preset.analytes:
            # Migration velocity is inversely related to (1 + k')
            band_speed = 1.0 / (1.0 + analyte.k_prime)
            frac = (sim_phase * band_speed * 3.5) % 1.15

            if 0.0 <= frac <= 1.0:
                band_center_x = col_x + frac * col_w
                # Band width broadens as it travels (diffusion)
                band_w = max(10.0, 8.0 + frac * 22.0)
                band_rect = QRectF(band_center_x - band_w / 2.0, col_y + 3.0, band_w, col_h - 6.0)

                # Translucent gaussian band gradient
                b_grad = QLinearGradient(band_rect.left(), col_y, band_rect.right(), col_y)
                col_c = analyte.color
                b_grad.setColorAt(0.0, QColor(col_c.red(), col_c.green(), col_c.blue(), 10))
                b_grad.setColorAt(0.5, QColor(col_c.red(), col_c.green(), col_c.blue(), 210))
                b_grad.setColorAt(1.0, QColor(col_c.red(), col_c.green(), col_c.blue(), 10))

                painter.setBrush(QBrush(b_grad))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(band_rect)

        # Legend of analytes below column
        leg_y = cy + 130.0
        painter.setFont(QFont("Segoe UI", 7))
        for i, analyte in enumerate(self._preset.analytes):
            painter.setBrush(QBrush(analyte.color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(cx + 25.0, leg_y + i * 22.0, 10.0, 10.0)

            t_r = self.calc_retention_time_s(analyte.k_prime)
            painter.setPen(QPen(QColor(226, 232, 240)))
            painter.drawText(
                cx + 45.0,
                leg_y + i * 22.0 + 9.0,
                f"{analyte.name_vi} ({analyte.formula}) — k' = {analyte.k_prime:.2f} | t_R = {t_r:.1f} s ({t_r/60.0:.2f} min)",
            )

        # Mobile phase info
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(cx + 20.0, cy + ch - 12.0, f"Pha động: {self._preset.mobile_phase_name}")

    def _paint_chromatogram_panel(self, painter: QPainter) -> None:
        """Plots simulated UV Absorbance Chromatogram A_254 vs Retention Time."""
        sx = 480.0
        sy = 60.0
        sw = 450.0
        sh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(sx, sy, sw, sh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(sx + 12.0, sy + 20.0, "SẮC KÝ ĐỒ ĐẦU DÒ UV-VIS (ABSORBANCE VS THỜI GIAN t_R):")

        # Plot axes area
        px = sx + 45.0
        py = sy + 35.0
        pw = sw - 65.0
        ph = sh - 65.0

        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawLine(px, py + ph, px + pw, py + ph)  # X-axis
        painter.drawLine(px, py, px, py + ph)            # Y-axis

        # Max time window for chromatogram: 1.2 * max(t_R)
        analytes = self._preset.analytes
        max_k = max(a.k_prime for a in analytes)
        max_time_s = max(60.0, self.calc_retention_time_s(max_k) * 1.25)

        # Plot composite Gaussian absorbance curve
        curve_path = QPainterPath()
        steps = int(pw)
        t_0 = self.column_void_time_s

        for s in range(steps + 1):
            t_curr = (s / float(steps)) * max_time_s
            total_abs = 0.0

            # Void volume disturbance peak at t_0
            sigma_0 = max(0.5, t_0 * 0.02)
            total_abs += 0.06 * math.exp(-0.5 * ((t_curr - t_0) / sigma_0) ** 2)

            # Analyte peaks
            for a in analytes:
                t_ri = self.calc_retention_time_s(a.k_prime)
                w_half = self.calc_peak_width_s(a.k_prime) * (2.355 / 4.0)
                sigma_i = max(0.2, w_half / 2.355)
                # Gaussian: A = A0 * exp(-0.5 * ((t - t_R)/sigma)²)
                total_abs += a.relative_height * math.exp(-0.5 * ((t_curr - t_ri) / sigma_i) ** 2)

            # Clamp total absorbance
            total_abs = min(1.05, total_abs)
            gx = px + s
            gy = py + ph - (total_abs / 1.1) * ph
            if s == 0:
                curve_path.moveTo(gx, gy)
            else:
                curve_path.lineTo(gx, gy)

        # Fill under chromatogram
        fill_path = QPainterPath(curve_path)
        fill_path.lineTo(px + pw, py + ph)
        fill_path.lineTo(px, py + ph)
        fill_path.closeSubpath()
        grad = QLinearGradient(px, py, px, py + ph)
        grad.setColorAt(0.0, QColor(56, 189, 248, 120))
        grad.setColorAt(1.0, QColor(56, 189, 248, 10))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # Stroke
        painter.setPen(QPen(QColor(56, 189, 248), 1.8))
        painter.drawPath(curve_path)

        # Void time marker t_0
        t0_x = px + (t_0 / max_time_s) * pw
        painter.setPen(QPen(QColor(148, 163, 184), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(t0_x, py, t0_x, py + ph)
        painter.setFont(QFont("Segoe UI", 6))
        painter.drawText(t0_x - 8.0, py + 12.0, "t_0")

        # Peak peak labels & retention times
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        for a in analytes:
            t_ri = self.calc_retention_time_s(a.k_prime)
            px_val = px + (t_ri / max_time_s) * pw
            py_val = py + ph - (a.relative_height / 1.1) * ph

            painter.setPen(QPen(a.color))
            painter.drawText(px_val - 12.0, py_val - 6.0, f"{t_ri:.1f}s")

        # Resolution indicator between peak 1 & 2
        rs = self.chromatographic_resolution_rs
        rs_color = QColor(34, 197, 94) if rs >= 1.5 else (QColor(234, 179, 8) if rs >= 1.0 else QColor(239, 68, 68))
        rs_status = "Đạt chuẩn Tách nền (R_s ≥ 1.5)" if rs >= 1.5 else ("Tách một phần (1.0 ≤ R_s < 1.5)" if rs >= 1.0 else "Chồng lấn / Thất bại (R_s < 1.0)")
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(rs_color))
        painter.drawText(px + 45.0, py + 18.0, f"Độ phân giải R_s = {rs:.2f} ({rs_status})")

        # X-axis time labels
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px, py + ph + 14.0, "0s")
        painter.drawText(px + pw / 2.0 - 15.0, py + ph + 14.0, f"{max_time_s/2.0:.0f}s")
        painter.drawText(px + pw - 25.0, py + ph + 14.0, f"{max_time_s:.0f}s ({max_time_s/60.0:.1f}m)")

    def _paint_van_deemter_panel(self, painter: QPainter) -> None:
        """Plots the Van Deemter efficiency curve H(u) = A + B/u + C·u."""
        vx = 20.0
        vy = 315.0
        vw = 440.0
        vh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(vx, vy, vw, vh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(vx + 12.0, vy + 20.0, "ĐƯỜNG CONG HIỆU SUẤT VAN DEEMTER: H(u) = A + B/u + C·u:")

        # Plot axes
        px = vx + 45.0
        py = vy + 35.0
        pw = vw - 65.0
        ph = vh - 65.0

        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawLine(px, py + ph, px + pw, py + ph)  # X: linear velocity u
        painter.drawLine(px, py, px, py + ph)            # Y: plate height H

        a, b, c = self.van_deemter_coefficients
        u_max = 5.0  # mm/s
        h_max = max(40.0, a * 3.5)

        # Plot components & composite curve
        vd_path = QPainterPath()
        a_path = QPainterPath()
        steps = int(pw)

        for s in range(steps + 1):
            u_val = max(0.08, (s / float(steps)) * u_max)
            h_val = a + (b / u_val) + (c * u_val)

            gx = px + s
            gy = py + ph - min(1.0, h_val / h_max) * ph
            if s == 0:
                vd_path.moveTo(gx, gy)
            else:
                vd_path.lineTo(gx, gy)

        painter.setPen(QPen(QColor(56, 189, 248), 2.2))
        painter.drawPath(vd_path)

        # Constant A term line (Eddy diffusion)
        a_y = py + ph - min(1.0, a / h_max) * ph
        painter.setPen(QPen(QColor(148, 163, 184), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(px, a_y, px + pw, a_y)
        painter.setFont(QFont("Segoe UI", 6))
        painter.drawText(px + pw - 90.0, a_y - 4.0, f"A (Eddy) = {a:.1f} μm")

        # Optimal velocity marker
        u_opt = self.optimal_linear_velocity_mm_s
        h_min = self.min_plate_height_um
        opt_x = px + (u_opt / u_max) * pw
        opt_y = py + ph - (h_min / h_max) * ph

        painter.setPen(QPen(QColor(34, 197, 94), 1.2, Qt.PenStyle.DotLine))
        painter.drawLine(opt_x, py, opt_x, py + ph)
        painter.setBrush(QBrush(QColor(34, 197, 94)))
        painter.drawEllipse(QPointF(opt_x, opt_y), 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(34, 197, 94)))
        painter.drawText(opt_x - 18.0, opt_y - 6.0, f"H_min={h_min:.1f}μm")

        # Current Operating Point
        u_curr = self.linear_velocity_mm_s
        h_curr = self.plate_height_um
        curr_x = px + min(pw, (u_curr / u_max) * pw)
        curr_y = py + ph - min(ph, (h_curr / h_max) * ph)

        painter.setBrush(QBrush(QColor(239, 68, 68)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        painter.drawEllipse(QPointF(curr_x, curr_y), 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(239, 68, 68)))
        painter.drawText(curr_x + 8.0, curr_y + 3.0, f"Hiện tại (u={u_curr:.2f} mm/s)")

        # X-axis label
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px + pw - 90.0, py + ph + 14.0, "Vận tốc dòng u (mm/s)")
        painter.drawText(px - 35.0, py + 10.0, "H (μm)")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Displays quantitative chromatography metrics and analytical equations."""
        mx = 480.0
        my = 315.0
        mw = 450.0
        mh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 20.0, "THÔNG SỐ HIỆU SUẤT CỘT & ĐỘ PHÂN GIẢI SẮC KÝ:")

        metrics = [
            ("Chiều cao Đĩa Lý thuyết (Plate Height):", f"H = {self.plate_height_um:.2f} μm (H_min = {self.min_plate_height_um:.2f} μm)"),
            ("Tổng Số Đĩa Lý thuyết (Theoretical Plates):", f"N = L / H = {self.theoretical_plates_n:.0f} đĩa"),
            ("Vận tốc Tuyến tính Dòng Pha động:", f"u = {self.linear_velocity_mm_s:.2f} mm/s (u_opt = {self.optimal_linear_velocity_mm_s:.2f} mm/s)"),
            ("Áp suất Ngược Thủy lực Cột (Backpressure):", f"ΔP = {self.column_backpressure_bar:.1f} bar"),
            ("Thời gian Chết Cột (Dead Time t_0):", f"t_0 = L / u = {self.column_void_time_s:.2f} s ({self.column_void_time_s/60.0:.2f} min)"),
            ("Hệ số Khuếch tán Xoáy A (Eddy Diffusion):", f"A = 1.5 · d_p = {self._dp_um * 1.5:.2f} μm"),
            ("Hệ số Kháng Truyền khối C (Mass Transfer):", f"C = 0.025 · d_p² = {0.025 * (self._dp_um**2):.3f} μm·s/mm"),
            ("Độ Phân giải giữa 2 Pic Đầu (Resolution R_s):", f"R_s = {self.chromatographic_resolution_rs:.2f}"),
        ]

        row_y = my + 44.0
        line_h = 21.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 14.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(226, 232, 240)))
            painter.drawText(mx + 235.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_f_down", "F -", QColor(51, 65, 85)),
            ("btn_f_up", "F +", QColor(51, 65, 85)),
            ("btn_dp_down", "dp -", QColor(51, 65, 85)),
            ("btn_dp_up", "dp +", QColor(51, 65, 85)),
            ("btn_col_l", f"L:{self._col_length_mm:.0f}", QColor(51, 65, 85)),
            ("btn_p_analgesic", "Thuốc Đau", QColor(30, 41, 59)),
            ("btn_p_uhplc", "UHPLC", QColor(30, 41, 59)),
            ("btn_p_coelute", "Đồng rửa giải", QColor(30, 41, 59)),
            ("btn_p_impurity", "Tạp 0.2%", QColor(30, 41, 59)),
            ("btn_play", "Dừng" if self._is_running else "Chạy", QColor(245, 158, 11)),
            ("btn_stamp", "Ghim", QColor(16, 185, 129)),
        ]

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        for key, text, color in buttons_info:
            rect = self._buttons[key]
            hovered = self._hovered_btn == key
            btn_col = color.lighter(125) if hovered else color

            painter.setBrush(QBrush(btn_col))
            painter.setPen(QPen(QColor(255, 255, 255, 160) if hovered else QColor(0, 0, 0, 80), 1.0))
            painter.drawRoundedRect(rect, 6.0, 6.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
