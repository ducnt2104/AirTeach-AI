"""
AirTeach AI — Interactive Biochemistry & Molecular Genetics: Polymerase Chain Reaction (PCR) & Gel Electrophoresis.
Simulates 3-step thermal cycling (Denaturation 95°C, Annealing 55°C, Extension 72°C),
exponential DNA amplification N_n = N_0·(1+E)^n, cycle threshold C_t, plateau effect,
and agarose gel electrophoresis separation based on molecular weight (bp).
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
class GelBand:
    size_bp: int
    name: str
    intensity: float  # 0.0 to 1.0


@dataclass(frozen=True)
class PCRPreset:
    code: str
    title_vi: str
    badge_label: str
    target_gene: str
    amplicon_size_bp: int
    initial_copies: float       # N_0 template copy number
    efficiency: float           # E (0.85 to 1.00)
    annealing_temp_c: float     # Optimal annealing temp T_a (°C)
    ladder_bands: list[int]
    description: str


PCR_PRESETS: dict[str, PCRPreset] = {
    "COVID19_RTPCR_DIAGNOSTICS": PCRPreset(
        code="COVID19_RTPCR_DIAGNOSTICS",
        title_vi="Chẩn đoán RT-qPCR SARS-CoV-2 — Đoạn Gen N (150 bp) & Ngưỡng Chu kỳ C_t",
        badge_label="SARS-CoV-2 N",
        target_gene="Gen N nucleocapsid",
        amplicon_size_bp=150,
        initial_copies=10000.0,
        efficiency=0.98,
        annealing_temp_c=58.0,
        ladder_bands=[1500, 1000, 750, 500, 250, 100],
        description="Quy trình chẩn đoán RT-qPCR chuẩn WHO. Nồng độ virus ban đầu cao (N₀ = 10,000 bản sao) giúp tín hiệu huỳnh quang vượt ngưỡng nền rất sớm tại chu kỳ C_t ≈ 18.5, khẳng định mẫu dương tính rõ rệt.",
    ),
    "FORENSIC_STR_DNA_PROFILING": PCRPreset(
        code="FORENSIC_STR_DNA_PROFILING",
        title_vi="Giám định ADN Pháp y — Đoạn Lặp Ngắn STR Locus TH01 (200 bp)",
        badge_label="FORENSIC STR",
        target_gene="STR Locus TH01",
        amplicon_size_bp=200,
        initial_copies=200.0,
        efficiency=0.95,
        annealing_temp_c=60.0,
        ladder_bands=[1500, 1000, 750, 500, 250, 100],
        description="Khuếch đại mẫu ADN vết tại hiện trường với lượng khởi đầu cực thấp (200 bản sao). Sau 32 chu kỳ nhân bản 2³², lượng sản phẩm đạt mức microgram để giải mã điện di mao quản phân định danh tính.",
    ),
    "PLASMID_RESTRICTION_CLONING": PCRPreset(
        code="PLASMID_RESTRICTION_CLONING",
        title_vi="Nhân dòng Phân tử Plasmid — Khuếch đại Gen Mục tiêu (850 bp)",
        badge_label="CLONING 850bp",
        target_gene="Gen GFP chèn",
        amplicon_size_bp=850,
        initial_copies=1000.0,
        efficiency=0.92,
        annealing_temp_c=55.0,
        ladder_bands=[2000, 1500, 1000, 750, 500, 250],
        description="Nhân bản đoạn gen mã hóa protein huỳnh quang xanh GFP (850 bp) từ plasmid. Sau khi nhân bản, sản phẩm được cắt bằng enzyme giới hạn EcoRI/BamHI để ghép vào vector biểu hiện.",
    ),
    "PRIMER_DIMER_NON_SPECIFIC": PCRPreset(
        code="PRIMER_DIMER_NON_SPECIFIC",
        title_vi="Lỗi Nhiệt độ Bắt cặp Quá Thấp — Xuất hiện Pic Primer-Dimer (50 bp)",
        badge_label="PRIMER-DIMER",
        target_gene="Gen Đích 600 bp + Dimer",
        amplicon_size_bp=600,
        initial_copies=500.0,
        efficiency=0.86,
        annealing_temp_c=48.0,  # Suboptimal low annealing temp
        ladder_bands=[1500, 1000, 750, 500, 250, 100],
        description="Khi hạ nhiệt độ bắt cặp xuống 48°C (thấp hơn Tm - 5°C), các đoạn mồi tự lai ghép với nhau tạo thành vạch mồi phụ 'Primer-Dimer' ở đáy bản gel (50 bp), làm giảm mạnh hiệu suất nhân gen chính.",
    ),
    "REAL_TIME_QPCR_SYBR_GREEN": PCRPreset(
        code="REAL_TIME_QPCR_SYBR_GREEN",
        title_vi="Khuếch đại Định lượng qPCR — Đường cong Tích lũy Huỳnh quang SYBR Green",
        badge_label="qPCR DYNAMICS",
        target_gene="Gen Chuẩn GAPDH",
        amplicon_size_bp=300,
        initial_copies=2500.0,
        efficiency=0.96,
        annealing_temp_c=60.0,
        ladder_bands=[1500, 1000, 750, 500, 250, 100],
        description="Thuốc nhuộm xen huỳnh quang SYBR Green chỉ phát sáng khi liên kết với chuỗi ADN sợi kép. Đồ thị cường độ huỳnh quang theo chu kỳ thể hiện pha trễ, pha hàm mũ lý tưởng và pha bão hòa plateau.",
    ),
}


class InteractivePCRItem(QGraphicsObject):
    """
    Interactive Polymerase Chain Reaction (PCR) & Agarose Gel Electrophoresis.
    Provides thermal cycler simulation (95°C / 55°C / 72°C), exponential DNA copy scaling,
    cycle threshold C_t calculation, and realistic agarose gel band migration.
    """

    preset_changed = Signal(str)
    pcr_computed = Signal(int, float, float)  # cycle_n, copies, ct_threshold
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0
    MAX_COPIES_PLATEAU = 2.0e11
    DETECTION_THRESHOLD = 1.0e7

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation Preset & Parameters
        self._active_preset_code: str = "COVID19_RTPCR_DIAGNOSTICS"
        self._preset: PCRPreset = PCR_PRESETS[self._active_preset_code]

        self._cycle_n: int = 25
        self._anneal_temp_c: float = self._preset.annealing_temp_c
        self._current_phase_idx: int = 1  # 0: Denat, 1: Anneal, 2: Extend

        # Animation state
        self._is_running: bool = True
        self._sim_tick: int = 0

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
    # Molecular Genetics & PCR Mathematical Equations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> PCRPreset:
        return self._preset

    @property
    def cycle_number(self) -> int:
        return self._cycle_n

    @property
    def annealing_temp_c(self) -> float:
        return self._anneal_temp_c

    @property
    def efficiency_factor(self) -> float:
        """
        Calculates effective efficiency E considering deviation from optimal annealing temperature:
        Temperature penalty: 5% loss per degree Celsius deviation.
        """
        temp_diff = abs(self._anneal_temp_c - self._preset.annealing_temp_c)
        penalty = min(0.35, temp_diff * 0.05)
        return max(0.50, self._preset.efficiency - penalty)

    @property
    def dna_copies(self) -> float:
        """
        Calculates DNA amplicon copy count using logistic saturation equation:
        N(n) = N_max / [1 + ((N_max - N_0)/N_0) * (1 + E)^(-n)]
        """
        n0 = self._preset.initial_copies
        e = self.efficiency_factor
        n_max = self.MAX_COPIES_PLATEAU
        denom = 1.0 + ((n_max - n0) / n0) * ((1.0 + e) ** (-self._cycle_n))
        return n_max / denom

    @property
    def cycle_threshold_ct(self) -> float:
        """
        Calculates theoretical Cycle Threshold C_t where copies reach DETECTION_THRESHOLD (10^7):
        C_t = ln(N_thresh / N_0) / ln(1 + E).
        """
        n0 = self._preset.initial_copies
        e = self.efficiency_factor
        if n0 >= self.DETECTION_THRESHOLD:
            return 1.0
        return math.log(self.DETECTION_THRESHOLD / n0) / math.log(1.0 + e)

    @property
    def amplification_fold(self) -> float:
        """Fold increase: N(n) / N_0."""
        return self.dna_copies / max(1.0, self._preset.initial_copies)

    def calc_copies_at_cycle(self, n: int) -> float:
        """Calculates copy count at arbitrary cycle n."""
        n0 = self._preset.initial_copies
        e = self.efficiency_factor
        n_max = self.MAX_COPIES_PLATEAU
        denom = 1.0 + ((n_max - n0) / n0) * ((1.0 + e) ** (-n))
        return n_max / denom

    # -------------------------------------------------------------------------
    # Animation & Parameter Controls
    # -------------------------------------------------------------------------
    def _on_sim_tick(self) -> None:
        if self._is_running:
            self._sim_tick += 1
            # Cycle through 3 phases every ~60 frames
            if self._sim_tick % 60 == 0:
                self._current_phase_idx = (self._current_phase_idx + 1) % 3
        self.update()

    def load_preset(self, preset_code: str) -> None:
        """Loads a curriculum molecular genetics preset."""
        if preset_code not in PCR_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PCR_PRESETS[preset_code]
        self._anneal_temp_c = self._preset.annealing_temp_c
        self._cycle_n = 25
        self.preset_changed.emit(preset_code)
        self.pcr_computed.emit(self._cycle_n, self.dna_copies, self.cycle_threshold_ct)
        self.update()

    def step_cycles(self, delta: int) -> None:
        """Steps PCR cycle number n (clamped between 0 and 40)."""
        self._cycle_n = max(0, min(40, self._cycle_n + delta))
        self.pcr_computed.emit(self._cycle_n, self.dna_copies, self.cycle_threshold_ct)
        self.update()

    def step_annealing_temp(self, delta_c: float) -> None:
        """Steps annealing temperature T_a."""
        self._anneal_temp_c = max(45.0, min(68.0, round(self._anneal_temp_c + delta_c, 1)))
        self.pcr_computed.emit(self._cycle_n, self.dna_copies, self.cycle_threshold_ct)
        self.update()

    def toggle_animation(self) -> None:
        """Toggles real-time phase animation."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Gestures
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_n_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_n_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_ta_down"] = QRectF(290.0, bar_y, 45.0, 34.0)
        self._buttons["btn_ta_up"] = QRectF(340.0, bar_y, 45.0, 34.0)
        self._buttons["btn_p_covid"] = QRectF(392.0, bar_y, 82.0, 34.0)
        self._buttons["btn_p_forensic"] = QRectF(480.0, bar_y, 80.0, 34.0)
        self._buttons["btn_p_cloning"] = QRectF(566.0, bar_y, 80.0, 34.0)
        self._buttons["btn_p_dimer"] = QRectF(652.0, bar_y, 85.0, 34.0)
        self._buttons["btn_p_sybr"] = QRectF(743.0, bar_y, 75.0, 34.0)
        self._buttons["btn_play"] = QRectF(824.0, bar_y, 55.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(885.0, bar_y, 45.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(PCR_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_n_down":
            self.step_cycles(-5)
        elif key == "btn_n_up":
            self.step_cycles(5)
        elif key == "btn_ta_down":
            self.step_annealing_temp(-1.0)
        elif key == "btn_ta_up":
            self.step_annealing_temp(1.0)
        elif key == "btn_p_covid":
            self.load_preset("COVID19_RTPCR_DIAGNOSTICS")
        elif key == "btn_p_forensic":
            self.load_preset("FORENSIC_STR_DNA_PROFILING")
        elif key == "btn_p_cloning":
            self.load_preset("PLASMID_RESTRICTION_CLONING")
        elif key == "btn_p_dimer":
            self.load_preset("PRIMER_DIMER_NON_SPECIFIC")
        elif key == "btn_p_sybr":
            self.load_preset("REAL_TIME_QPCR_SYBR_GREEN")
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
            "target_gene": self._preset.target_gene,
            "amplicon_bp": self._preset.amplicon_size_bp,
            "cycle_n": self._cycle_n,
            "dna_copies": self.dna_copies,
            "ct_threshold": self.cycle_threshold_ct,
            "efficiency": self.efficiency_factor,
            "annealing_temp_c": self._anneal_temp_c,
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
        self._paint_thermal_cycle_panel(painter)
        self._paint_amplification_curve_panel(painter)
        self._paint_gel_electrophoresis_panel(painter)
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
        painter.drawText(22, 30, f"SINH HỌC PHÂN TỬ: PHẢN ỨNG CHUỖI POLYMERASE (PCR) & ĐIỆN DI GEL: {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        copies_sci = f"{self.dna_copies:.2e}"
        painter.drawText(
            22,
            48,
            f"Mục tiêu: {self._preset.target_gene} ({self._preset.amplicon_size_bp} bp) | Chu kỳ n = {self._cycle_n} | Số bản sao N = {copies_sci} ({self.amplification_fold:.1e}x) | Ngưỡng C_t = {self.cycle_threshold_ct:.1f} | T_a = {self._anneal_temp_c:.1f}°C (E = {self.efficiency_factor*100:.1f}%)",
        )

    def _paint_thermal_cycle_panel(self, painter: QPainter) -> None:
        """Renders 3-step thermal cycler temperature curve and DNA strand status."""
        cx = 20.0
        cy = 60.0
        cw = 440.0
        ch = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(cx, cy, cw, ch, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(cx + 12.0, cy + 20.0, "CHU TRÌNH NHIỆT PCR 3 BƯỚC (THERMAL CYCLER):")

        # 3 Stages Diagram
        stages = [
            ("1. Biến tính (Denaturation)", "95°C", "Tách ADN sợi đôi", QColor(239, 68, 68), cx + 18.0, cy + 45.0, 125.0, 75.0),
            ("2. Bắt cặp (Annealing)", f"{self._anneal_temp_c:.0f}°C", "Mồi gắn đặc hiệu", QColor(56, 189, 248), cx + 158.0, cy + 85.0, 125.0, 75.0),
            ("3. Kéo dài (Extension)", "72°C", "Taq pol tổng hợp", QColor(34, 197, 94), cx + 298.0, cy + 60.0, 125.0, 75.0),
        ]

        for i, (name, temp_str, desc, col, sx, sy, sw, sh) in enumerate(stages):
            is_active = self._current_phase_idx == i
            bg_col = col.lighter(130) if is_active else col
            painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 40 if not is_active else 120)))
            painter.setPen(QPen(col if is_active else col.darker(140), 2.0 if is_active else 1.0))
            painter.drawRoundedRect(sx, sy, sw, sh, 6.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(col))
            painter.drawText(sx + 8.0, sy + 18.0, name)

            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            painter.drawText(sx + 8.0, sy + 42.0, temp_str)

            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QPen(QColor(226, 232, 240)))
            painter.drawText(sx + 8.0, sy + 62.0, desc)

        # DNA Strand visualizer below thermal steps
        strand_y = cy + 185.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(cx + 18.0, strand_y - 8.0, "TRẠNG THÁI PHÂN TỬ ADN:")

        # Draw DNA strands according to active phase
        phase_idx = self._current_phase_idx
        if phase_idx == 0:  # Denaturation: 2 single strands separated
            painter.setPen(QPen(QColor(239, 68, 68), 2.5))
            painter.drawLine(cx + 30.0, strand_y + 12.0, cx + 410.0, strand_y + 12.0)
            painter.drawLine(cx + 30.0, strand_y + 35.0, cx + 410.0, strand_y + 35.0)
            painter.setFont(QFont("Segoe UI", 6))
            painter.drawText(cx + 30.0, strand_y + 48.0, "5' ─── Hai sợi đơn tách rời do đứt liên kết Hydro ở 95°C ─── 3'")
        elif phase_idx == 1:  # Annealing: Primers attached
            painter.setPen(QPen(QColor(148, 163, 184), 2.0))
            painter.drawLine(cx + 30.0, strand_y + 12.0, cx + 410.0, strand_y + 12.0)
            painter.drawLine(cx + 30.0, strand_y + 35.0, cx + 410.0, strand_y + 35.0)
            # Primers (blue arrows)
            painter.setPen(QPen(QColor(56, 189, 248), 3.5))
            painter.drawLine(cx + 40.0, strand_y + 18.0, cx + 110.0, strand_y + 18.0)
            painter.drawLine(cx + 330.0, strand_y + 29.0, cx + 400.0, strand_y + 29.0)
            painter.setFont(QFont("Segoe UI", 6))
            painter.drawText(cx + 40.0, strand_y + 48.0, f"Mồi xuôi & ngược bắt cặp đặc hiệu ở T_a = {self._anneal_temp_c:.0f}°C")
        else:  # Extension: Taq synthesis
            painter.setPen(QPen(QColor(34, 197, 94), 2.5))
            painter.drawLine(cx + 30.0, strand_y + 12.0, cx + 410.0, strand_y + 12.0)
            painter.drawLine(cx + 30.0, strand_y + 18.0, cx + 320.0, strand_y + 18.0)
            painter.drawLine(cx + 30.0, strand_y + 35.0, cx + 410.0, strand_y + 35.0)
            painter.drawLine(cx + 120.0, strand_y + 29.0, cx + 410.0, strand_y + 29.0)
            painter.setFont(QFont("Segoe UI", 6))
            painter.drawText(cx + 30.0, strand_y + 48.0, "Taq DNA Polymerase kéo dài chuỗi bổ sung 5' → 3' ở 72°C")

    def _paint_amplification_curve_panel(self, painter: QPainter) -> None:
        """Plots semi-logarithmic DNA amplification curve log10(N) vs Cycle Number."""
        ax = 480.0
        ay = 60.0
        aw = 450.0
        ah = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(ax, ay, aw, ah, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(ax + 12.0, ay + 20.0, "ĐỒ THỊ KHUẾCH ĐẠI RT-qPCR: log₁₀(BẢN SAO) VS CHU KỲ (n):")

        # Plot axes
        px = ax + 45.0
        py = ay + 35.0
        pw = aw - 65.0
        ph = ah - 65.0

        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawLine(px, py + ph, px + pw, py + ph)  # X-axis: cycles 0 to 40
        painter.drawLine(px, py, px, py + ph)            # Y-axis: log10(copies) from 2 to 12

        log_min = 2.0
        log_max = 12.0
        log_span = log_max - log_min

        # Threshold line (10^7 copies, log = 7.0)
        thresh_y = py + ph - ((7.0 - log_min) / log_span) * ph
        painter.setPen(QPen(QColor(244, 63, 94), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(px, thresh_y, px + pw, thresh_y)
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(244, 63, 94)))
        painter.drawText(px + pw - 90.0, thresh_y - 4.0, "Ngưỡng C_t (10⁷ copies)")

        # Plot Sigmoidal Curve N(n)
        curve_path = QPainterPath()
        steps = int(pw)

        for s in range(steps + 1):
            n_curr = (s / float(steps)) * 40.0
            copies_curr = self.calc_copies_at_cycle(int(n_curr))
            log_c = math.log10(max(1.0, copies_curr))
            gx = px + s
            gy = py + ph - ((log_c - log_min) / log_span) * ph
            if s == 0:
                curve_path.moveTo(gx, gy)
            else:
                curve_path.lineTo(gx, gy)

        painter.setPen(QPen(QColor(56, 189, 248), 2.2))
        painter.drawPath(curve_path)

        # Mark Cycle Threshold C_t on X-axis
        ct = self.cycle_threshold_ct
        if 0.0 <= ct <= 40.0:
            ct_x = px + (ct / 40.0) * pw
            painter.setPen(QPen(QColor(250, 204, 21), 1.2, Qt.PenStyle.DotLine))
            painter.drawLine(ct_x, py, ct_x, py + ph)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(250, 204, 21)))
            painter.drawText(ct_x - 14.0, py + ph + 13.0, f"C_t={ct:.1f}")

        # Current cycle operating point
        curr_x = px + (self._cycle_n / 40.0) * pw
        curr_log = math.log10(max(1.0, self.dna_copies))
        curr_y = py + ph - ((curr_log - log_min) / log_span) * ph

        painter.setBrush(QBrush(QColor(34, 197, 94)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.4))
        painter.drawEllipse(QPointF(curr_x, curr_y), 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(34, 197, 94)))
        painter.drawText(curr_x + 8.0, curr_y + 3.0, f"n={self._cycle_n}")

        # X-axis Cycle labels
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px, py + ph + 14.0, "0")
        painter.drawText(px + pw / 2.0 - 6.0, py + ph + 14.0, "20")
        painter.drawText(px + pw - 12.0, py + ph + 14.0, "40")
        painter.drawText(px - 32.0, py + 10.0, "10¹²")
        painter.drawText(px - 32.0, py + ph, "10²")

    def _paint_gel_electrophoresis_panel(self, painter: QPainter) -> None:
        """Renders realistic agarose gel electrophoresis slab under UV transillumination."""
        gx = 20.0
        gy = 315.0
        gw = 440.0
        gh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(gx, gy, gw, gh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(gx + 12.0, gy + 20.0, "ĐIỆN DI GEL AGAROSE (302 nm UV TRANSILLUMINATOR):")

        # Gel Slab (translucent deep blue)
        gel_x = gx + 50.0
        gel_y = gy + 35.0
        gel_w = gw - 100.0
        gel_h = gh - 55.0

        gel_grad = QLinearGradient(gel_x, gel_y, gel_x, gel_y + gel_h)
        gel_grad.setColorAt(0.0, QColor(10, 20, 45))
        gel_grad.setColorAt(1.0, QColor(5, 10, 25))
        painter.setBrush(QBrush(gel_grad))
        painter.setPen(QPen(QColor(30, 58, 138), 1.5))
        painter.drawRect(gel_x, gel_y, gel_w, gel_h)

        # Electrodes: Cathode (-) at top, Anode (+) at bottom
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(239, 68, 68)))
        painter.drawText(gel_x - 30.0, gel_y + 18.0, "(-) K")
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(gel_x - 30.0, gel_y + gel_h - 10.0, "(+) A")

        # Loading Wells at top
        n_lanes = 4
        lane_w = gel_w / float(n_lanes)
        for l in range(n_lanes):
            well_x = gel_x + l * lane_w + lane_w * 0.25
            painter.setBrush(QBrush(QColor(2, 6, 23)))
            painter.setPen(QPen(QColor(71, 85, 105), 1.0))
            painter.drawRect(well_x, gel_y + 4.0, lane_w * 0.5, 10.0)

        # Lane labels
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(gel_x + 0.15 * lane_w, gel_y - 4.0, "1: Thang ADN")
        painter.drawText(gel_x + 1.15 * lane_w, gel_y - 4.0, "2: PCR Đích")
        painter.drawText(gel_x + 2.15 * lane_w, gel_y - 4.0, "3: Đối chứng (-)")
        painter.drawText(gel_x + 3.15 * lane_w, gel_y - 4.0, "4: Bệnh phẩm")

        # Migration function: Y(bp) ∝ -log10(bp)
        # 100 bp migrates furthest (near bottom); 2000 bp migrates least (near top)
        def get_band_y(bp: int) -> float:
            log_bp = math.log10(max(40, bp))
            log_min = math.log10(40)
            log_max = math.log10(2500)
            frac = (log_max - log_bp) / (log_max - log_min)
            return gel_y + 25.0 + frac * (gel_h - 45.0)

        # Lane 1: DNA Ladder
        l1_x = gel_x + 0.25 * lane_w
        for bp in self._preset.ladder_bands:
            by = get_band_y(bp)
            self._draw_fluorescent_band(painter, l1_x, by, lane_w * 0.5, 3.5, 0.75, QColor(250, 204, 21))
            painter.setFont(QFont("Segoe UI", 5))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(l1_x - 22.0, by + 3.0, f"{bp}")

        # Lane 2: Target PCR Product
        # Band brightness depends on cycle number (log copies)
        log_copies = math.log10(max(1.0, self.dna_copies))
        band_intensity = min(1.0, max(0.0, (log_copies - 4.0) / 6.0))

        if band_intensity > 0.05:
            l2_x = gel_x + 1.25 * lane_w
            target_bp = self._preset.amplicon_size_bp
            by = get_band_y(target_bp)
            self._draw_fluorescent_band(painter, l2_x, by, lane_w * 0.5, 4.5, band_intensity, QColor(56, 189, 248))
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(56, 189, 248)))
            painter.drawText(l2_x + lane_w * 0.5 + 4.0, by + 3.0, f"{target_bp} bp")

        # Primer-Dimer band if present (e.g. at 50 bp)
        if self._active_preset_code == "PRIMER_DIMER_NON_SPECIFIC":
            dimer_y = get_band_y(50)
            l2_x = gel_x + 1.25 * lane_w
            self._draw_fluorescent_band(painter, l2_x, dimer_y, lane_w * 0.5, 3.0, 0.85, QColor(244, 63, 94))
            painter.setFont(QFont("Segoe UI", 5))
            painter.setPen(QPen(QColor(244, 63, 94)))
            painter.drawText(l2_x + lane_w * 0.5 + 4.0, dimer_y + 3.0, "Dimer 50bp")

        # Lane 4: Patient sample
        if self._cycle_n >= 20 and self._preset.initial_copies > 100:
            l4_x = gel_x + 3.25 * lane_w
            by = get_band_y(self._preset.amplicon_size_bp)
            self._draw_fluorescent_band(painter, l4_x, by, lane_w * 0.5, 4.0, band_intensity * 0.9, QColor(34, 197, 94))

    def _draw_fluorescent_band(
        self, painter: QPainter, x: float, y: float, w: float, h: float, intensity: float, tint: QColor
    ) -> None:
        """Renders glowing DNA band under UV staining."""
        alpha = int(max(20, min(255, intensity * 255)))
        b_col = QColor(tint.red(), tint.green(), tint.blue(), alpha)
        painter.setBrush(QBrush(b_col))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y - h / 2.0, w, h, 2.0, 2.0)

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Displays rigorous analytical equations and quantitative metrics."""
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
        painter.drawText(mx + 12.0, my + 20.0, "THÔNG SỐ ĐỘNG HỌC KHUẾCH ĐẠI & ĐIỆN DI:")

        metrics = [
            ("Kích thước Đoạn gen Mục tiêu (bp):", f"{self._preset.amplicon_size_bp} bp ({self._preset.target_gene})"),
            ("Số Bản sao Khuếch đại Hiện tại N(n):", f"{self.dna_copies:.2e} bản sao ({self.amplification_fold:.1e}x)"),
            ("Số Bản sao Mẫu Ban đầu (N_0):", f"{self._preset.initial_copies:,.0f} bản sao"),
            ("Hiệu suất Phản ứng Chuỗi (Efficiency E):", f"E = {self.efficiency_factor*100:.1f}% (T_a = {self._anneal_temp_c:.1f}°C)"),
            ("Ngưỡng Chu kỳ Định lượng (C_t):", f"C_t = {self.cycle_threshold_ct:.2f} ({'Dương tính mạnh' if self.cycle_threshold_ct < 25 else 'Mẫu âm tính hoặc vết'})"),
            ("Số Chu kỳ Hoàn thành Hiện tại:", f"n = {self._cycle_n} / 40 chu kỳ"),
            ("Tốc độ Điện di DNA trong Gel (Mobility):", f"v ∝ 1 / log₁₀(bp) hướng về Cực Dương (+) Anot"),
            ("Độ Sáng Vạch Sản phẩm trên Bản Gel:", f"{'Rõ nét định lượng' if self.dna_copies > 1e8 else 'Mờ nhạt / Dưới ngưỡng phát hiện'}"),
        ]

        row_y = my + 44.0
        line_h = 21.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 14.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(226, 232, 240)))
            painter.drawText(mx + 225.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_n_down", "n -5", QColor(51, 65, 85)),
            ("btn_n_up", "n +5", QColor(51, 65, 85)),
            ("btn_ta_down", "Ta -1", QColor(51, 65, 85)),
            ("btn_ta_up", "Ta +1", QColor(51, 65, 85)),
            ("btn_p_covid", "Covid RT", QColor(30, 41, 59)),
            ("btn_p_forensic", "Pháp y", QColor(30, 41, 59)),
            ("btn_p_cloning", "Plasmid", QColor(30, 41, 59)),
            ("btn_p_dimer", "Mồi phụ", QColor(30, 41, 59)),
            ("btn_p_sybr", "SYBR", QColor(30, 41, 59)),
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
