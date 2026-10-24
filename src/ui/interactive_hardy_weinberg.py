"""
AirTeach AI — Interactive Biology & Population Genetics: Hardy-Weinberg Equilibrium Simulator.
Simulates p + q = 1 and p² + 2pq + q² = 1, Wright-Fisher stochastic genetic drift,
natural selection with differential genotype fitness (w_AA, w_Aa, w_aa), heterozygote advantage
(sickle-cell anemia balanced polymorphism), population bottlenecks, and Chi-square goodness-of-fit tests.
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
class HWPreset:
    code: str
    title_vi: str
    badge_label: str
    initial_p: float           # Frequency of dominant allele A
    pop_size: int              # N individuals (0 for infinite/deterministic)
    w_aa_dom: float            # Relative fitness w_AA
    w_aa_het: float            # Relative fitness w_Aa
    w_aa_rec: float            # Relative fitness w_aa
    inbreeding_f: float        # Fixation index F (0.0 for panmixia)
    description: str


HW_PRESETS: dict[str, HWPreset] = {
    "CLASSIC_IDEAL_EQUILIBRIUM": HWPreset(
        code="CLASSIC_IDEAL_EQUILIBRIUM",
        title_vi="Cân bằng Hardy-Weinberg Lý tưởng — Quần thể Vô hạn (N → ∞)",
        badge_label="HWE IDEAL",
        initial_p=0.70,
        pop_size=0,  # Infinite population
        w_aa_dom=1.00,
        w_aa_het=1.00,
        w_aa_rec=1.00,
        inbreeding_f=0.00,
        description="Quần thể ngẫu phối vô hạn, không chọn lọc tự nhiên, không đột biến, không di nhập gen. Tần số alen p=0.7, q=0.3 và tần số kiểu gen p²=0.49, 2pq=0.42, q²=0.09 duy trì bất biến qua mọi thế hệ (χ² = 0.00).",
    ),
    "SICKLE_CELL_OVERDOMINANCE": HWPreset(
        code="SICKLE_CELL_OVERDOMINANCE",
        title_vi="Ưu thế Dị hợp tử (Overdominance) — Hồng cầu Hình liềm & Sốt rét",
        badge_label="HET ADVANTAGE",
        initial_p=0.50,
        pop_size=1000,
        w_aa_dom=0.88,  # HbA/HbA susceptible to severe malaria
        w_aa_het=1.00,  # HbA/HbS resistant to malaria, normal blood
        w_aa_rec=0.14,  # HbS/HbS severe sickle-cell anemia
        inbreeding_f=0.00,
        description="Chọn lọc cân bằng (Balanced polymorphism): Dị hợp tử HbA/HbS có độ thích nghi cao nhất. Quần thể tự động hội tụ về tần số cân bằng ổn định p̂ = (1 - w_aa)/(2 - w_AA - w_aa) ≈ 0.88 bất kể tần số ban đầu.",
    ),
    "RECESSIVE_LETHAL_SELECTION": HWPreset(
        code="RECESSIVE_LETHAL_SELECTION",
        title_vi="Chọn lọc Đào thải Đồng hợp tử Lặn (w_aa = 0) — Bệnh Di truyền",
        badge_label="RECESSIVE LETHAL",
        initial_p=0.50,
        pop_size=1000,
        w_aa_dom=1.00,
        w_aa_het=1.00,
        w_aa_rec=0.00,  # Lethal recessive allele
        inbreeding_f=0.00,
        description="Alen lặn gây chết hoàn toàn ở trạng thái đồng hợp (w_aa=0, như xơ nang, Tay-Sachs). Tần số alen q giảm nhanh ban đầu nhưng không bao giờ bị triệt tiêu hoàn toàn vì được 'ẩn náu' an toàn trong các dị hợp tử 2pq.",
    ),
    "GENETIC_DRIFT_BOTTLENECK": HWPreset(
        code="GENETIC_DRIFT_BOTTLENECK",
        title_vi="Phiêu bạt Di truyền & Thắt cổ chai Quần thể (N = 25)",
        badge_label="DRIFT BOTTLENECK",
        initial_p=0.50,
        pop_size=25,   # Small population bottleneck
        w_aa_dom=1.00,
        w_aa_het=1.00,
        w_aa_rec=1.00,
        inbreeding_f=0.00,
        description="Quần thể quy mô nhỏ (N=25) chịu sự thăng giáng ngẫu nhiên do sai số lấy mẫu ngẫu phối (Wright-Fisher model). Tần số alen biến động mạnh theo bước đi ngẫu nhiên, dẫn đến mất tính đa dạng và cố định alen (p=1 hoặc p=0).",
    ),
    "INBREEDING_CONSANGUINITY_F": HWPreset(
        code="INBREEDING_CONSANGUINITY_F",
        title_vi="Nội phối / Giao phối Cận huyết (Hệ số F = 0.25)",
        badge_label="INBREEDING F=0.25",
        initial_p=0.60,
        pop_size=1000,
        w_aa_dom=1.00,
        w_aa_het=1.00,
        w_aa_rec=1.00,
        inbreeding_f=0.25,  # Brother-sister mating coefficient
        description="Giao phối đồng huyết làm tăng vọt tần số đồng hợp tử f(AA)=p²+Fpq và f(aa)=q²+Fpq, đồng thời suy giảm mạnh tỷ lệ dị hợp tử f(Aa)=2pq(1-F). Gây bộc lộ các bệnh di truyền lặn nghiêm trọng.",
    ),
}


class InteractiveHardyWeinbergItem(QGraphicsObject):
    """
    Interactive Hardy-Weinberg Population Genetics & Evolution Simulator.
    Provides real-time p² + 2pq + q² parabolas, Wright-Fisher stochastic drift,
    natural selection trajectories, Chi-square goodness-of-fit, and 100-cell organism grid.
    """

    preset_changed = Signal(str)
    generation_advanced = Signal(int, float, float)  # gen, p, q
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation Preset
        self._active_preset_code: str = "CLASSIC_IDEAL_EQUILIBRIUM"
        self._preset: HWPreset = HW_PRESETS[self._active_preset_code]

        # Population Genetic State
        self._p: float = self._preset.initial_p
        self._pop_size: int = self._preset.pop_size
        self._w_aa: float = self._preset.w_aa_dom
        self._w_het: float = self._preset.w_aa_het
        self._w_rec: float = self._preset.w_aa_rec
        self._inbreeding_f: float = self._preset.inbreeding_f

        # Multi-generation tracking
        self._generation: int = 0
        self._p_history: list[float] = [self._p]
        self._observed_counts: tuple[int, int, int] = (0, 0, 0)  # (N_AA, N_Aa, N_aa)
        self._sample_population()

        # Playback animation
        self._is_running: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Evolution generation timer (advance generation every 120 ms)
        self._timer = QTimer()
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Properties & Population Genetics Equations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> HWPreset:
        return self._preset

    @property
    def p(self) -> float:
        return self._p

    @property
    def q(self) -> float:
        return 1.0 - self._p

    @property
    def expected_hwe_frequencies(self) -> tuple[float, float, float]:
        """Calculates expected theoretical HWE frequencies (p², 2pq, q²)."""
        p = self._p
        q = self.q
        return (p**2, 2.0 * p * q, q**2)

    @property
    def inbreeding_adjusted_frequencies(self) -> tuple[float, float, float]:
        """
        Calculates expected genotype frequencies under inbreeding coefficient F:
        f(AA) = p² + F*p*q
        f(Aa) = 2*p*q*(1 - F)
        f(aa) = q² + F*p*q
        """
        p = self._p
        q = self.q
        f = self._inbreeding_f
        f_aa = p**2 + f * p * q
        f_het = 2.0 * p * q * (1.0 - f)
        f_rec = q**2 + f * p * q
        return (f_aa, f_het, f_rec)

    @property
    def mean_fitness(self) -> float:
        """Calculates population mean relative fitness w̄ = p² w_AA + 2pq w_Aa + q² w_aa."""
        f_aa, f_het, f_rec = self.inbreeding_adjusted_frequencies
        return f_aa * self._w_aa + f_het * self._w_het + f_rec * self._w_rec

    @property
    def chi_square_statistic(self) -> float:
        """
        Calculates Chi-square goodness-of-fit test statistic:
        χ² = Σ (O_i - E_i)² / E_i
        Degrees of freedom = 1. Critical value at α=0.05 is 3.841.
        """
        n_tot = sum(self._observed_counts)
        if n_tot <= 0:
            return 0.0

        p = self._p
        q = self.q
        exp_aa = n_tot * (p**2)
        exp_het = n_tot * (2.0 * p * q)
        exp_rec = n_tot * (q**2)

        o_aa, o_het, o_rec = self._observed_counts

        chi_sq = 0.0
        if exp_aa > 0:
            chi_sq += ((o_aa - exp_aa) ** 2) / exp_aa
        if exp_het > 0:
            chi_sq += ((o_het - exp_het) ** 2) / exp_het
        if exp_rec > 0:
            chi_sq += ((o_rec - exp_rec) ** 2) / exp_rec

        return chi_sq

    def is_in_hwe(self) -> bool:
        """Checks if current population is consistent with HWE (χ² < 3.841, p > 0.05)."""
        return self.chi_square_statistic < 3.841

    # -------------------------------------------------------------------------
    # Stochastic Sampling & Evolution Step
    # -------------------------------------------------------------------------
    def _sample_population(self) -> None:
        """Samples individuals according to genotype frequencies and population size."""
        f_aa, f_het, f_rec = self.inbreeding_adjusted_frequencies
        n = 100 if self._pop_size == 0 else min(1000, self._pop_size)

        if self._pop_size == 0:
            # Deterministic perfect sample
            n_aa = int(round(f_aa * n))
            n_het = int(round(f_het * n))
            n_rec = max(0, n - n_aa - n_het)
        else:
            # Stochastic multinomial sampling with natural selection weighting
            w_bar = max(1e-6, self.mean_fitness)
            sel_aa = (f_aa * self._w_aa) / w_bar
            sel_het = (f_het * self._w_het) / w_bar
            sel_rec = (f_rec * self._w_rec) / w_bar

            # Draw n individuals
            weights = [sel_aa, sel_het, sel_rec]
            draws = random.choices([0, 1, 2], weights=weights, k=n)
            n_aa = draws.count(0)
            n_het = draws.count(1)
            n_rec = draws.count(2)

        self._observed_counts = (n_aa, n_het, n_rec)

    def advance_generation(self) -> None:
        """
        Advances one generation under natural selection and genetic drift:
        Deterministic update:
        p' = (p² w_AA + pq w_Aa) / w̄
        Stochastic drift (Wright-Fisher):
        Draw 2N alleles from binomial(2N, p')
        """
        p = self._p
        q = 1.0 - p
        w_bar = max(1e-6, self.mean_fitness)

        # Expected p after natural selection
        p_next_det = (p**2 * self._w_aa + p * q * self._w_het) / w_bar
        p_next_det = max(0.0, min(1.0, p_next_det))

        if self._pop_size == 0:
            self._p = p_next_det
        else:
            # Wright-Fisher binomial sampling with 2N alleles
            total_alleles = 2 * self._pop_size
            # Use random.binomial approximation
            p_curr = p_next_det
            if p_curr <= 0.0:
                num_a = 0
            elif p_curr >= 1.0:
                num_a = total_alleles
            else:
                # Binomial sampling
                num_a = sum(1 for _ in range(total_alleles) if random.random() < p_curr)
            self._p = num_a / float(total_alleles)

        self._generation += 1
        self._p_history.append(self._p)
        if len(self._p_history) > 120:
            self._p_history.pop(0)

        self._sample_population()
        self.generation_advanced.emit(self._generation, self._p, self.q)
        self.update()

    def reset_simulation(self, reset_to_preset: bool = False) -> None:
        """Resets simulation to generation 0."""
        self._generation = 0
        if reset_to_preset:
            self._p = self._preset.initial_p
        self._p_history = [self._p]
        self._sample_population()
        self.update()

    def _on_sim_tick(self) -> None:
        if self._is_running:
            self.advance_generation()

    # -------------------------------------------------------------------------
    # Presets & Control Methods
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a population genetics preset."""
        if preset_code not in HW_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = HW_PRESETS[preset_code]
        self._p = self._preset.initial_p
        self._pop_size = self._preset.pop_size
        self._w_aa = self._preset.w_aa_dom
        self._w_het = self._preset.w_aa_het
        self._w_rec = self._preset.w_aa_rec
        self._inbreeding_f = self._preset.inbreeding_f
        self.reset_simulation(reset_to_preset=True)
        self.preset_changed.emit(preset_code)

    def step_p(self, delta_p: float) -> None:
        """Adjusts initial allele frequency p."""
        self._p = max(0.01, min(0.99, self._p + delta_p))
        self.reset_simulation(reset_to_preset=False)

    def toggle_animation(self) -> None:
        """Toggles evolution playback."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Gestures
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_p_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_p_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_step_gen"] = QRectF(290.0, bar_y, 90.0, 34.0)
        self._buttons["btn_reset"] = QRectF(386.0, bar_y, 75.0, 34.0)
        self._buttons["btn_p_sickle"] = QRectF(467.0, bar_y, 95.0, 34.0)
        self._buttons["btn_p_drift"] = QRectF(568.0, bar_y, 95.0, 34.0)
        self._buttons["btn_p_inbreed"] = QRectF(669.0, bar_y, 95.0, 34.0)
        self._buttons["btn_play"] = QRectF(770.0, bar_y, 65.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(841.0, bar_y, 89.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(HW_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_p_down":
            self.step_p(-0.05)
        elif key == "btn_p_up":
            self.step_p(0.05)
        elif key == "btn_step_gen":
            self.advance_generation()
        elif key == "btn_reset":
            self.reset_simulation(reset_to_preset=True)
        elif key == "btn_p_sickle":
            self.load_preset("SICKLE_CELL_OVERDOMINANCE")
        elif key == "btn_p_drift":
            self.load_preset("GENETIC_DRIFT_BOTTLENECK")
        elif key == "btn_p_inbreed":
            self.load_preset("INBREEDING_CONSANGUINITY_F")
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
        f_aa, f_het, f_rec = self.inbreeding_adjusted_frequencies
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "generation": self._generation,
            "p": self._p,
            "q": self.q,
            "f_aa": f_aa,
            "f_het": f_het,
            "f_rec": f_rec,
            "pop_size": self._pop_size,
            "chi_square": self.chi_square_statistic,
            "is_hwe": self.is_in_hwe(),
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
        self._paint_hwe_parabola_chart(painter)
        self._paint_evolution_trajectory_chart(painter)
        self._paint_organism_population_grid(painter)
        self._paint_analytical_metrics(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(10, 14, 26, 252))
        grad.setColorAt(1.0, QColor(6, 9, 18, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"DI TRUYỀN HỌC QUẦN THỂ: CÂN BẰNG HARDY-WEINBERG – {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        pop_str = "Vô hạn (N → ∞)" if self._pop_size == 0 else f"N = {self._pop_size}"
        painter.drawText(22, 48, f"Phương trình: p + q = 1 | p² + 2pq + q² = 1 | Thế hệ: {self._generation} | Kích thước: {pop_str} | Kiểm định χ² = {self.chi_square_statistic:.2f} (df=1)")

    def _paint_hwe_parabola_chart(self, painter: QPainter) -> None:
        """Paints Hardy-Weinberg parabolas: p² (AA), 2pq (Aa), q² (aa) vs p."""
        cx = 20.0
        cy = 60.0
        cw = 380.0
        ch = 250.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(cx, cy, cw, ch, 8.0)

        # Inner plot area
        px0 = cx + 45.0
        py0 = cy + ch - 35.0
        pw = cw - 65.0
        ph = ch - 60.0

        # Axes
        painter.setPen(QPen(QColor(100, 116, 139), 1.2))
        painter.drawLine(px0, py0, px0 + pw, py0)
        painter.drawLine(px0, py0, px0, py0 - ph)

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px0 - 25.0, py0 + 4.0, "0.0")
        painter.drawText(px0 - 25.0, py0 - ph / 2.0 + 4.0, "0.5")
        painter.drawText(px0 - 25.0, py0 - ph + 4.0, "1.0")

        painter.drawText(px0 - 4.0, py0 + 16.0, "0.0")
        painter.drawText(px0 + pw / 2.0 - 10.0, py0 + 16.0, "p=0.5")
        painter.drawText(px0 + pw - 10.0, py0 + 16.0, "1.0")

        # Theoretical Parabolas
        pts_aa = QPainterPath()
        pts_het = QPainterPath()
        pts_rec = QPainterPath()

        num_steps = 50
        for s in range(num_steps + 1):
            cur_p = s / float(num_steps)
            cur_q = 1.0 - cur_p
            x = px0 + cur_p * pw

            # p² (AA)
            y_aa = py0 - (cur_p**2) * ph
            # 2pq (Aa)
            y_het = py0 - (2.0 * cur_p * cur_q) * ph
            # q² (aa)
            y_rec = py0 - (cur_q**2) * ph

            if s == 0:
                pts_aa.moveTo(x, y_aa)
                pts_het.moveTo(x, y_het)
                pts_rec.moveTo(x, y_rec)
            else:
                pts_aa.lineTo(x, y_aa)
                pts_het.lineTo(x, y_het)
                pts_rec.lineTo(x, y_rec)

        # Draw curves
        painter.setPen(QPen(QColor(239, 68, 68), 2.2))
        painter.drawPath(pts_aa)

        painter.setPen(QPen(QColor(168, 85, 247), 2.2))
        painter.drawPath(pts_het)

        painter.setPen(QPen(QColor(6, 182, 212), 2.2))
        painter.drawPath(pts_rec)

        # Current p line
        curr_x = px0 + self._p * pw
        painter.setPen(QPen(QColor(245, 158, 11), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(curr_x, py0, curr_x, py0 - ph)

        # Current genotype dots
        f_aa, f_het, f_rec = self.inbreeding_adjusted_frequencies
        painter.setBrush(QBrush(QColor(239, 68, 68)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(curr_x, py0 - f_aa * ph), 4.5, 4.5)

        painter.setBrush(QBrush(QColor(168, 85, 247)))
        painter.drawEllipse(QPointF(curr_x, py0 - f_het * ph), 4.5, 4.5)

        painter.setBrush(QBrush(QColor(6, 182, 212)))
        painter.drawEllipse(QPointF(curr_x, py0 - f_rec * ph), 4.5, 4.5)

        # Legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(239, 68, 68)))
        painter.drawText(cx + 45.0, cy + 20.0, "■ p² (AA)")
        painter.setPen(QPen(QColor(168, 85, 247)))
        painter.drawText(cx + 125.0, cy + 20.0, "■ 2pq (Aa)")
        painter.setPen(QPen(QColor(6, 182, 212)))
        painter.drawText(cx + 215.0, cy + 20.0, "■ q² (aa)")

    def _paint_evolution_trajectory_chart(self, painter: QPainter) -> None:
        """Paints multi-generation trajectory of allele frequency p(t)."""
        tx = 20.0
        ty = 320.0
        tw = 380.0
        th = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(tx, ty, tw, th, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(tx + 12.0, ty + 20.0, "QUỸ ĐẠO TIẾN HÓA TẦN SỐ ALEN p(t) QUA THẾ HỆ:")

        px0 = tx + 45.0
        py0 = ty + th - 30.0
        pw = tw - 65.0
        ph = th - 55.0

        # Axes
        painter.setPen(QPen(QColor(100, 116, 139), 1.0))
        painter.drawLine(px0, py0, px0 + pw, py0)
        painter.drawLine(px0, py0, px0, py0 - ph)

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px0 - 25.0, py0 + 3.0, "0.0")
        painter.drawText(px0 - 25.0, py0 - ph + 3.0, "1.0")
        painter.drawText(px0 - 25.0, py0 - ph / 2.0 + 3.0, "0.5")

        # p = 0.5 reference line
        painter.setPen(QPen(QColor(51, 65, 85), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(px0, py0 - ph * 0.5, px0 + pw, py0 - ph * 0.5)

        # Plot p_history
        if len(self._p_history) > 1:
            t_path = QPainterPath()
            n_pts = len(self._p_history)
            for idx, p_val in enumerate(self._p_history):
                gx = px0 + (idx / float(max(1, n_pts - 1))) * pw
                gy = py0 - p_val * ph
                if idx == 0:
                    t_path.moveTo(gx, gy)
                else:
                    t_path.lineTo(gx, gy)

            painter.setPen(QPen(QColor(245, 158, 11), 2.2))
            painter.drawPath(t_path)

            # Tip current value
            last_x = px0 + pw
            last_y = py0 - self._p * ph
            painter.setBrush(QBrush(QColor(245, 158, 11)))
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.drawEllipse(QPointF(last_x, last_y), 4.0, 4.0)

    def _paint_organism_population_grid(self, painter: QPainter) -> None:
        """Paints 100 individual organisms colored by genotype (AA: Red, Aa: Purple, aa: Cyan)."""
        gx = 415.0
        gy = 60.0
        gw = 240.0
        gh = 250.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(gx, gy, gw, gh, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(gx + 12.0, gy + 20.0, "MẪU QUẦN THỂ 100 CÁ THỂ:")

        o_aa, o_het, o_rec = self._observed_counts
        tot = max(1, o_aa + o_het + o_rec)
        # Normalize to 100 cells
        n_aa = int(round((o_aa / tot) * 100))
        n_het = int(round((o_het / tot) * 100))
        n_rec = max(0, 100 - n_aa - n_het)

        cell_colors = [QColor(239, 68, 68)] * n_aa + [QColor(168, 85, 247)] * n_het + [QColor(6, 182, 212)] * n_rec
        random.seed(42)  # Fixed aesthetic arrangement
        random.shuffle(cell_colors)

        cols = 10
        rows = 10
        start_x = gx + 18.0
        start_y = gy + 32.0
        size = 18.0
        gap = 2.5

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx < len(cell_colors):
                    col = cell_colors[idx]
                    painter.setBrush(QBrush(col))
                    painter.setPen(Qt.PenStyle.NoPen)
                    cell_rect = QRectF(start_x + c * (size + gap), start_y + r * (size + gap), size, size)
                    painter.drawRoundedRect(cell_rect, 3.0, 3.0)

    def _paint_analytical_metrics(self, painter: QPainter) -> None:
        """Paints analytical allele frequencies, chi-square, and HWE validation."""
        mx = 665.0
        my = 60.0
        mw = 265.0
        mh = 500.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(mx + 12.0, my + 22.0, "THÔNG SỐ DI TRUYỀN HỌC QUẦN THỂ:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(226, 232, 240)))

        f_aa, f_het, f_rec = self.inbreeding_adjusted_frequencies
        o_aa, o_het, o_rec = self._observed_counts
        tot = max(1, o_aa + o_het + o_rec)
        chi_sq = self.chi_square_statistic
        is_eq = self.is_in_hwe()

        status_text = "ĐẠT CÂN BẰNG HWE (p > 0.05)" if is_eq else "LỆCH CÂN BẰNG HWE (p < 0.05)"
        status_col = QColor(16, 185, 129) if is_eq else QColor(239, 68, 68)

        lines = [
            f"• Thế hệ: {self._generation}",
            f"• Tần số alen A (p): {self._p:.4f}",
            f"• Tần số alen a (q): {self.q:.4f}",
            f"• Tần số lý thuyết HWE:",
            f"   - f(AA) = p² = {self._p**2:.4f}",
            f"   - f(Aa) = 2pq = {2.0 * self._p * self.q:.4f}",
            f"   - f(aa) = q² = {self.q**2:.4f}",
            f"• Mẫu quan sát thực tế (N = {tot}):",
            f"   - AA: {o_aa} cá thể ({o_aa/tot*100:.1f}%)",
            f"   - Aa: {o_het} cá thể ({o_het/tot*100:.1f}%)",
            f"   - aa: {o_rec} cá thể ({o_rec/tot*100:.1f}%)",
            f"• Thích nghi tương đối (w_AA, w_Aa, w_aa):",
            f"   = ({self._w_aa:.2f}, {self._w_het:.2f}, {self._w_rec:.2f})",
            f"• Hệ số nội phối F: {self._inbreeding_f:.2f}",
            f"• Độ thích nghi trung bình w̄: {self.mean_fitness:.4f}",
            f"• Thống kê Chi-bình phương:",
            f"   χ² = {chi_sq:.3f} (Ngưỡng 3.841, α=0.05)",
        ]
        curr_y = my + 42.0
        for line in lines:
            painter.drawText(mx + 12.0, curr_y, line)
            curr_y += 18.0

        # Status badge
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(status_col))
        painter.drawText(mx + 12.0, curr_y + 8.0, status_text)

        # Description
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(148, 163, 184)))
        desc_rect = QRectF(mx + 12.0, curr_y + 24.0, mw - 24.0, 65.0)
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        self._layout_buttons()
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # 1. Preset Selector
        self._draw_btn(painter, self._buttons["btn_preset"], "btn_preset", f"Preset: {self._preset.badge_label}")

        # 2. Allele frequency steppers
        self._draw_btn(painter, self._buttons["btn_p_down"], "btn_p_down", "-p")
        self._draw_btn(painter, self._buttons["btn_p_up"], "btn_p_up", "+p")

        # 3. Generation controls
        self._draw_btn(painter, self._buttons["btn_step_gen"], "btn_step_gen", "⏭ 1 Thế hệ")
        self._draw_btn(painter, self._buttons["btn_reset"], "btn_reset", "↺ Đặt lại")

        # 4. Quick Presets
        self._draw_btn(painter, self._buttons["btn_p_sickle"], "btn_p_sickle", "Hồng cầu liềm")
        self._draw_btn(painter, self._buttons["btn_p_drift"], "btn_p_drift", "Thắt cổ chai N=25")
        self._draw_btn(painter, self._buttons["btn_p_inbreed"], "btn_p_inbreed", "Cận huyết F=0.25")

        # 5. Play / Pause
        play_label = "⏸ Dừng" if self._is_running else "▶ Tiếp"
        self._draw_btn(painter, self._buttons["btn_play"], "btn_play", play_label)

        # 6. Stamp Diagram
        self._draw_btn(painter, self._buttons["btn_stamp"], "btn_stamp", "📌 Chèn Vector")

    def _draw_btn(self, painter: QPainter, rect: QRectF, key: str, text: str) -> None:
        is_hovered = self._hovered_btn == key
        bg_col = QColor(30, 41, 59, 220) if not is_hovered else QColor(51, 65, 85, 240)
        border_col = QColor(56, 189, 248) if is_hovered else QColor(71, 85, 105)

        painter.setBrush(QBrush(bg_col))
        painter.setPen(QPen(border_col, 1.2))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setPen(QPen(QColor(248, 250, 252) if is_hovered else QColor(203, 213, 225)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # Mouse fallbacks
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        new_hover = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                new_hover = key
                break
        if new_hover != self._hovered_btn:
            self._hovered_btn = new_hover
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._handle_button_click(key)
                event.accept()
                return
        super().mousePressEvent(event)
