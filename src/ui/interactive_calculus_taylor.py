"""
AirTeach AI – Interactive Mathematics: Riemann Integrals, Numerical Quadrature & Taylor Series Approximations
=============================================================================================================
Provides an authentic mathematical analysis and calculus simulation:
  - Riemann Integral Sums & Quadratures:
      - Partition of [a, b] into n subintervals: Δx = (b - a) / n, x_i = a + i * Δx.
      - Left Riemann Sum: L_n = Δx * sum_{i=0}^{n-1} f(x_i)
      - Right Riemann Sum: R_n = Δx * sum_{i=1}^{n} f(x_i)
      - Midpoint Rule: M_n = Δx * sum_{i=1}^{n} f((x_{i-1} + x_i) / 2)  [Error O(Δx²)]
      - Trapezoidal Rule: T_n = (Δx / 2) * [f(a) + 2 * sum_{i=1}^{n-1} f(x_i) + f(b)]  [Error O(Δx²)]
      - Simpson's 1/3 Rule: S_n = (Δx / 3) * [f(x_0) + 4 * sum_{odd} f(x_i) + 2 * sum_{even} f(x_i) + f(x_n)]  [Error O(Δx⁴)]
  - Taylor & Maclaurin Polynomial Approximations:
      P_N(x) = sum_{k=0}^N (f^(k)(x_0) / k!) * (x - x_0)^k
      - sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
      - cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
      - exp(x) = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
  - 5 Curriculum Presets:
      1. RIEMANN_SINE_INTEGRAL: Integral of sin(x) from 0 to π (Exact = 2.0), comparing L, R, M, T, Simpson.
      2. RIEMANN_PARABOLA_AREA: Parabolic area f(x) = 4 - x² on [-2, 2] (Exact = 32/3 ≈ 10.667).
      3. TAYLOR_SIN_APPROXIMATION: Maclaurin expansion of sin(x) displaying polynomial degrees N = 1, 3, 5, 7, 9.
      4. TAYLOR_EXP_APPROXIMATION: Maclaurin expansion of exp(x) displaying polynomial degrees N = 1, 2, 3, 4, 5, 6.
      5. SIMPSON_HIGH_PRECISION: Comparison showing Simpson's 1/3 rule error dropping by O(n⁻⁴).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Callable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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


@dataclass(frozen=True)
class CalculusPreset:
    code: str
    title_vi: str
    badge_label: str
    func_name: str
    is_taylor_mode: bool
    a: float
    b: float
    default_n: int
    taylor_degree: int
    exact_integral: float
    description: str


PRESETS: dict[str, CalculusPreset] = {
    "RIEMANN_SINE_INTEGRAL": CalculusPreset(
        code="RIEMANN_SINE_INTEGRAL",
        title_vi="Tích phân Riemann Hàm Sin: ∫₀^π sin(x)dx = 2.0",
        badge_label="RIEMANN SIN",
        func_name="f(x) = sin(x)",
        is_taylor_mode=False,
        a=0.0,
        b=math.pi,
        default_n=8,
        taylor_degree=5,
        exact_integral=2.0,
        description="Tính diện tích hình thang cong giới hạn bởi đồ thị y = sin(x) trên [0, π]. So sánh tổng Riemann trái, phải, điểm giữa, hình thang và Simpson 1/3.",
    ),
    "RIEMANN_PARABOLA_AREA": CalculusPreset(
        code="RIEMANN_PARABOLA_AREA",
        title_vi="Diện tích Hình Parabol Archimedes: ∫₋₂^² (4 - x²)dx = 32/3",
        badge_label="PARABOL 32/3",
        func_name="f(x) = 4 - x²",
        is_taylor_mode=False,
        a=-2.0,
        b=2.0,
        default_n=8,
        taylor_degree=3,
        exact_integral=32.0 / 3.0,
        description="Bài toán cầu phương Parabol kinh điển của Archimedes. Công thức Simpson 1/3 cho kết quả chính xác tuyệt đối ngay từ n = 2 phân khoảng.",
    ),
    "TAYLOR_SIN_APPROXIMATION": CalculusPreset(
        code="TAYLOR_SIN_APPROXIMATION",
        title_vi="Khai triển Đa thức Taylor/Maclaurin của sin(x)",
        badge_label="TAYLOR SIN(x)",
        func_name="f(x) = sin(x)",
        is_taylor_mode=True,
        a=-2.0 * math.pi,
        b=2.0 * math.pi,
        default_n=10,
        taylor_degree=5,
        exact_integral=0.0,
        description="Xấp xỉ hàm tuần hoàn sin(x) bằng chuỗi đa thức Maclaurin bậc lẻ P_N(x) = x - x³/3! + x⁵/5! - x⁷/7!. Bán kính và miền hội tụ mở rộng theo bậc N.",
    ),
    "TAYLOR_EXP_APPROXIMATION": CalculusPreset(
        code="TAYLOR_EXP_APPROXIMATION",
        title_vi="Khai triển Taylor Đa thức Hàm Mũ: exp(x)",
        badge_label="TAYLOR EXP(x)",
        func_name="f(x) = exp(x)",
        is_taylor_mode=True,
        a=-3.0,
        b=3.0,
        default_n=10,
        taylor_degree=4,
        exact_integral=math.exp(3.0) - math.exp(-3.0),
        description="Chuỗi hàm mũ e^x = 1 + x + x²/2! + x³/3! + ... hội tụ tuyệt đối trên toàn bộ trục số thực R với tốc độ hội tụ siêu nhanh.",
    ),
    "SIMPSON_HIGH_PRECISION": CalculusPreset(
        code="SIMPSON_HIGH_PRECISION",
        title_vi="Độ chính xác Cao của Công thức Cầu phương Simpson 1/3",
        badge_label="SIMPSON 1/3",
        func_name="f(x) = sin(x)",
        is_taylor_mode=False,
        a=0.0,
        b=math.pi,
        default_n=16,
        taylor_degree=5,
        exact_integral=2.0,
        description="Công thức Simpson 1/3 xấp xỉ từng đoạn bằng đường cong Parabol bậc 2, sai số lý thuyết O(n⁻⁴) giảm cực kỳ nhanh chóng so với Riemann bậc 1.",
    ),
}


class InteractiveCalculusTaylorItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Mathematical Analysis & Calculus:
    Riemann Integral Sums (Left, Right, Midpoint, Trapezoidal, Simpson) and Taylor Series.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    preset_changed = Signal(str)

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

        # Simulation Preset & Mathematical Configuration
        self._active_preset_code: str = "RIEMANN_SINE_INTEGRAL"
        self._preset: CalculusPreset = PRESETS[self._active_preset_code]

        self._a: float = self._preset.a
        self._b: float = self._preset.b
        self._n_partitions: int = self._preset.default_n
        self._taylor_n: int = self._preset.taylor_degree
        self._is_taylor_mode: bool = self._preset.is_taylor_mode

        # Riemann rule: 0 = Midpoint, 1 = Left, 2 = Right, 3 = Trapezoid, 4 = Simpson
        self._riemann_rule: int = 0

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

    # -------------------------------------------------------------------------
    # Mathematical Calculations (Zero Mock Data)
    # -------------------------------------------------------------------------
    def eval_f(self, x: float) -> float:
        """Evaluates authentic mathematical function f(x)."""
        code = self._active_preset_code
        if code in ["RIEMANN_SINE_INTEGRAL", "TAYLOR_SIN_APPROXIMATION", "SIMPSON_HIGH_PRECISION"]:
            return math.sin(x)
        elif code == "RIEMANN_PARABOLA_AREA":
            return 4.0 - x ** 2
        elif code == "TAYLOR_EXP_APPROXIMATION":
            return math.exp(min(5.0, max(-5.0, x)))
        return math.sin(x)

    def eval_taylor(self, x: float, degree: int) -> float:
        """Evaluates authentic Taylor polynomial P_N(x) around x_0 = 0."""
        code = self._active_preset_code
        if code in ["TAYLOR_SIN_APPROXIMATION", "RIEMANN_SINE_INTEGRAL", "SIMPSON_HIGH_PRECISION"]:
            # sin(x) = sum_{k=0}^N (-1)^k * x^(2k+1) / (2k+1)!
            val = 0.0
            for k in range(0, degree // 2 + 1):
                p = 2 * k + 1
                if p <= degree:
                    term = ((-1.0) ** k) * (x ** p) / float(math.factorial(p))
                    val += term
            return val
        elif code == "TAYLOR_EXP_APPROXIMATION":
            # exp(x) = sum_{k=0}^N x^k / k!
            val = 0.0
            for k in range(0, degree + 1):
                term = (x ** k) / float(math.factorial(k))
                val += term
            return val
        elif code == "RIEMANN_PARABOLA_AREA":
            # 4 - x^2 is already exact at degree >= 2
            return 4.0 - (x ** 2 if degree >= 2 else 0.0)
        return math.sin(x)

    def compute_riemann_sum(self, rule_idx: int) -> float:
        """Calculates authentic numerical quadrature integral."""
        a, b, n = self._a, self._b, self._n_partitions
        if n < 1:
            return 0.0
        dx = (b - a) / float(n)

        if rule_idx == 1:  # Left sum
            return dx * sum(self.eval_f(a + i * dx) for i in range(n))
        elif rule_idx == 2:  # Right sum
            return dx * sum(self.eval_f(a + i * dx) for i in range(1, n + 1))
        elif rule_idx == 0:  # Midpoint sum
            return dx * sum(self.eval_f(a + (i + 0.5) * dx) for i in range(n))
        elif rule_idx == 3:  # Trapezoidal rule
            s = 0.5 * (self.eval_f(a) + self.eval_f(b))
            s += sum(self.eval_f(a + i * dx) for i in range(1, n))
            return dx * s
        elif rule_idx == 4:  # Simpson's 1/3 rule (requires even n)
            n_even = n if n % 2 == 0 else n + 1
            dx_even = (b - a) / float(n_even)
            s = self.eval_f(a) + self.eval_f(b)
            for i in range(1, n_even):
                coeff = 4.0 if i % 2 == 1 else 2.0
                s += coeff * self.eval_f(a + i * dx_even)
            return (dx_even / 3.0) * s
        return 0.0

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> CalculusPreset:
        return self._preset

    @property
    def is_taylor_mode(self) -> bool:
        return self._is_taylor_mode

    @property
    def n_partitions(self) -> int:
        return self._n_partitions

    @property
    def taylor_degree(self) -> int:
        return self._taylor_n

    @property
    def riemann_rule(self) -> int:
        return self._riemann_rule

    def load_preset(self, preset_code: str) -> None:
        """Loads curriculum calculus preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._a = self._preset.a
        self._b = self._preset.b
        self._n_partitions = self._preset.default_n
        self._taylor_n = self._preset.taylor_degree
        self._is_taylor_mode = self._preset.is_taylor_mode
        self.preset_changed.emit(preset_code)
        self.update()

    def set_partitions(self, n: int) -> None:
        self._n_partitions = max(2, min(64, int(n)))
        self.update()

    def set_taylor_degree(self, n: int) -> None:
        self._taylor_n = max(1, min(11, int(n)))
        self.update()

    def cycle_riemann_rule(self) -> None:
        """Cycles between Midpoint (0), Left (1), Right (2), Trapezoid (3), Simpson (4)."""
        self._riemann_rule = (self._riemann_rule + 1) % 5
        self.update()

    def toggle_mode(self) -> None:
        """Toggles between Riemann Quadrature view and Taylor Polynomial view."""
        self._is_taylor_mode = not self._is_taylor_mode
        self.update()

    def stamp_to_board(self) -> None:
        self.stamp_requested.emit()

    def cleanup(self) -> None:
        """Cleans up internal state and UI interactive button rects."""
        self._btn_rects.clear()

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
        painter.setBrush(QBrush(QColor("#080c14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Graph & Areas / Polynomials)
        left_w = 430.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_math_viewport(painter, view_rect)

        # 4. Right Telemetry & Controls
        right_rect = QRectF(view_rect.right() + 10.0, 48.0, self.WIDTH - view_rect.right() - 24.0, 310.0)
        self._draw_control_panel(painter, right_rect)

        # 5. Bottom Calculus Theory Banner
        bot_rect = QRectF(14.0, view_rect.bottom() + 10.0, self.WIDTH - 28.0, self.HEIGHT - view_rect.bottom() - 20.0)
        self._draw_theory_banner(painter, bot_rect)

        painter.restore()

    # -------------------------------------------------------------------------
    # Sub-Drawing Modules
    # -------------------------------------------------------------------------
    def _draw_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "GIẢI TÍCH TOÁN HỌC: TÍCH PHÂN RIEMANN & CHUỖI TAYLOR")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Tổng Riemann • Quy tắc Hình thang & Simpson 1/3 • Khai triển Đa thức Taylor • Bán kính hội tụ")

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

    def _draw_math_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        # Coordinate axes & domain bounds
        a, b = self._a, self._b
        dom_w = b - a
        if dom_w <= 0:
            dom_w = 1.0

        # Determine y range
        if self._preset.code == "RIEMANN_PARABOLA_AREA":
            y_min, y_max = -1.0, 5.0
        elif self._preset.code == "TAYLOR_EXP_APPROXIMATION":
            y_min, y_max = -1.0, 15.0
        else:
            y_min, y_max = -1.5, 1.5

        def to_canvas(x: float, y: float) -> QPointF:
            nx = (x - a) / dom_w
            px = rect.left() + 35.0 + nx * (rect.width() - 55.0)
            ny = (y - y_min) / (y_max - y_min)
            py = rect.bottom() - 30.0 - ny * (rect.height() - 55.0)
            return QPointF(px, py)

        # Zero axes lines
        p_y0_left = to_canvas(a, 0.0)
        p_y0_right = to_canvas(b, 0.0)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(p_y0_left, p_y0_right)

        if a <= 0.0 <= b:
            p_x0_bot = to_canvas(0.0, y_min)
            p_x0_top = to_canvas(0.0, y_max)
            painter.drawLine(p_x0_bot, p_x0_top)

        if not self._is_taylor_mode:
            # 1. RIEMANN INTEGRAL SUMS VISUALIZATION
            n = self._n_partitions
            dx = (b - a) / float(n)
            rule = self._riemann_rule

            for i in range(n):
                x_left = a + i * dx
                x_right = x_left + dx

                if rule == 1:  # Left sum
                    sample_x = x_left
                    h_val = self.eval_f(sample_x)
                    p_tl = to_canvas(x_left, h_val)
                    p_tr = to_canvas(x_right, h_val)
                    p_bl = to_canvas(x_left, 0.0)
                    p_br = to_canvas(x_right, 0.0)

                    poly = QPainterPath()
                    poly.moveTo(p_bl)
                    poly.lineTo(p_tl)
                    poly.lineTo(p_tr)
                    poly.lineTo(p_br)
                    poly.closeSubpath()

                    painter.setBrush(QBrush(QColor(56, 189, 248, 50)))
                    painter.setPen(QPen(QColor("#38bdf8"), 1.0))
                    painter.drawPath(poly)

                elif rule == 2:  # Right sum
                    sample_x = x_right
                    h_val = self.eval_f(sample_x)
                    p_tl = to_canvas(x_left, h_val)
                    p_tr = to_canvas(x_right, h_val)
                    p_bl = to_canvas(x_left, 0.0)
                    p_br = to_canvas(x_right, 0.0)

                    poly = QPainterPath()
                    poly.moveTo(p_bl)
                    poly.lineTo(p_tl)
                    poly.lineTo(p_tr)
                    poly.lineTo(p_br)
                    poly.closeSubpath()

                    painter.setBrush(QBrush(QColor(251, 191, 36, 50)))
                    painter.setPen(QPen(QColor("#fbbf24"), 1.0))
                    painter.drawPath(poly)

                elif rule == 0:  # Midpoint sum
                    sample_x = (x_left + x_right) / 2.0
                    h_val = self.eval_f(sample_x)
                    p_tl = to_canvas(x_left, h_val)
                    p_tr = to_canvas(x_right, h_val)
                    p_bl = to_canvas(x_left, 0.0)
                    p_br = to_canvas(x_right, 0.0)

                    poly = QPainterPath()
                    poly.moveTo(p_bl)
                    poly.lineTo(p_tl)
                    poly.lineTo(p_tr)
                    poly.lineTo(p_br)
                    poly.closeSubpath()

                    painter.setBrush(QBrush(QColor(16, 185, 129, 60)))
                    painter.setPen(QPen(QColor("#10b981"), 1.0))
                    painter.drawPath(poly)

                elif rule in [3, 4]:  # Trapezoid / Simpson segments
                    y_l = self.eval_f(x_left)
                    y_r = self.eval_f(x_right)
                    p_tl = to_canvas(x_left, y_l)
                    p_tr = to_canvas(x_right, y_r)
                    p_bl = to_canvas(x_left, 0.0)
                    p_br = to_canvas(x_right, 0.0)

                    poly = QPainterPath()
                    poly.moveTo(p_bl)
                    poly.lineTo(p_tl)
                    poly.lineTo(p_tr)
                    poly.lineTo(p_br)
                    poly.closeSubpath()

                    painter.setBrush(QBrush(QColor(168, 85, 247, 50)))
                    painter.setPen(QPen(QColor("#a855f7"), 1.2))
                    painter.drawPath(poly)

        else:
            # 2. TAYLOR SERIES POLYNOMIAL VISUALIZATION
            deg = self._taylor_n
            t_path = QPainterPath()
            n_samples = 120
            for i in range(n_samples + 1):
                xv = a + (i / float(n_samples)) * dom_w
                yv = self.eval_taylor(xv, deg)
                # Clamp to avoid off-screen overflow
                yv_clamped = max(y_min - 10.0, min(y_max + 10.0, yv))
                pt = to_canvas(xv, yv_clamped)
                if i == 0:
                    t_path.moveTo(pt)
                else:
                    t_path.lineTo(pt)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#fbbf24"), 2.2, Qt.PenStyle.DashLine))
            painter.drawPath(t_path)

        # 3. Plot exact curve f(x) (Cyan solid)
        f_path = QPainterPath()
        n_samples = 140
        for i in range(n_samples + 1):
            xv = a + (i / float(n_samples)) * dom_w
            yv = self.eval_f(xv)
            pt = to_canvas(xv, yv)
            if i == 0:
                f_path.moveTo(pt)
            else:
                f_path.lineTo(pt)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawPath(f_path)

        # Viewport Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        mode_hdr = f"KHAI TRIỂN TAYLOR BẬC N = {self._taylor_n}" if self._is_taylor_mode else f"TÍCH PHÂN RIEMANN: {self._preset.func_name} TRÊN [{a:.2f}, {b:.2f}]"
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 20.0), mode_hdr)

        # Legend
        leg_r = QRectF(rect.right() - 170.0, rect.top() + 8.0, 160.0, 22.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRoundedRect(leg_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(leg_r.left() + 8.0, leg_r.top() + 14.0), "━ f(x) Chuẩn")
        painter.setPen(QColor("#fbbf24"))
        legend_txt = "┅ P_N(x) Taylor" if self._is_taylor_mode else "■ Diện tích xấp xỉ"
        painter.drawText(QPointF(leg_r.left() + 75.0, leg_r.top() + 14.0), legend_txt)

        painter.restore()

    # -------------------------------------------------------------------------
    # Right Telemetry & Controls
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
        painter.drawText(QPointF(ox, y), "THÔNG SỐ GIẢI TÍCH SỐ:")

        # Badge Card
        y += 18.0
        card_r = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(card_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        val_str = f"N = {self._taylor_n}" if self._is_taylor_mode else f"n = {self._n_partitions}"
        painter.drawText(QRectF(ox + 8.0, y, 65.0, 32.0), Qt.AlignmentFlag.AlignVCenter, val_str)

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        sub_str = "Chế độ: Khai triển Chuỗi Taylor" if self._is_taylor_mode else "Chế độ: Phân vùng Riemann"
        rule_names = ["Điểm giữa (Midpoint)", "Tổng Trái (Left)", "Tổng Phải (Right)", "Hình thang (Trapezoid)", "Simpson 1/3 (Parabol)"]
        sub_desc = f"{sub_str}\n{rule_names[self._riemann_rule] if not self._is_taylor_mode else 'Xấp xỉ đa thức quanh x₀=0'}"
        painter.drawText(QRectF(ox + 75.0, y + 2.0, w - 80.0, 28.0), Qt.AlignmentFlag.AlignVCenter, sub_desc)

        # Quantitative Measurements Readout
        y += 40.0
        exact_i = self._preset.exact_integral
        num_i = self.compute_riemann_sum(self._riemann_rule)
        diff = abs(num_i - exact_i)
        err_pct = (diff / max(1e-6, abs(exact_i))) * 100.0 if exact_i != 0 else 0.0

        rows = [
            (f"• Tích phân giải tích I: {exact_i:.6f}", QColor("#38bdf8")),
            (f"• Xấp xỉ số I_num: {num_i:.6f}", QColor("#10b981")),
            (f"• Sai số tuyệt đối Δ: {diff:.6f}", QColor("#fbbf24")),
            (f"• Sai số tương đối: {err_pct:.3f}%", QColor("#cbd5e1")),
            (f"• Số phân khoảng n: {self._n_partitions}", QColor("#e2e8f0")),
            (f"• Bậc đa thức Taylor N: {self._taylor_n}", QColor("#a855f7")),
            (f"• Sai số lý thuyết: {'O(n⁻⁴) Simpson' if self._riemann_rule == 4 else 'O(n⁻²) Bậc hai'}", QColor("#94a3b8")),
        ]

        for txt, col in rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 15.0

        # Action Buttons: Switch Rule, Toggle Mode, Partition +/-
        y = rect.bottom() - 48.0
        rule_btn = QRectF(ox, y, (w - 8.0) / 2.0, 20.0)
        mode_btn = QRectF(ox + (w - 8.0) / 2.0 + 8.0, y, (w - 8.0) / 2.0, 20.0)
        self._btn_rects["cycle_rule"] = rule_btn
        self._btn_rects["toggle_mode"] = mode_btn

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(rule_btn, 3.0, 3.0)
        painter.drawRoundedRect(mode_btn, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(rule_btn, Qt.AlignmentFlag.AlignCenter, "Đổi Quy tắc Riemann")
        painter.drawText(mode_btn, Qt.AlignmentFlag.AlignCenter, "Taylor ⇋ Riemann")

        y += 24.0
        p_m = QRectF(ox, y, (w - 8.0) / 2.0, 20.0)
        p_p = QRectF(ox + (w - 8.0) / 2.0 + 8.0, y, (w - 8.0) / 2.0, 20.0)
        self._btn_rects["part_minus"] = p_m
        self._btn_rects["part_plus"] = p_p

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(p_m, 3.0, 3.0)
        painter.drawRoundedRect(p_p, 3.0, 3.0)

        step_lbl_m = "Bậc N - 2" if self._is_taylor_mode else "Phân vùng n - 4"
        step_lbl_p = "Bậc N + 2" if self._is_taylor_mode else "Phân vùng n + 4"
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(p_m, Qt.AlignmentFlag.AlignCenter, step_lbl_m)
        painter.drawText(p_p, Qt.AlignmentFlag.AlignCenter, step_lbl_p)

    # -------------------------------------------------------------------------
    # Bottom Calculus Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "NGUYÊN LÝ GIẢI TÍCH & TỔNG RIEMANN / CHUỖI TAYLOR:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Định nghĩa Tích phân Riemann: lim_{{n->∞}} Σ f(x_i*)Δx xác định diện tích đại số chính xác dưới đường cong.\n"
            f"• Công thức Simpson 1/3 đạt cấp chính xác O(n⁻⁴), vượt trội tổng Riemann hình chữ nhật O(n⁻¹).\n"
            f"• Tình huống: {self._preset.description}"
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

        # 2. Cycle Riemann rule
        if self._btn_rects.get("cycle_rule") and self._btn_rects["cycle_rule"].contains(local_pos):
            self.cycle_riemann_rule()
            return True

        # 3. Toggle Mode (Taylor vs Riemann)
        if self._btn_rects.get("toggle_mode") and self._btn_rects["toggle_mode"].contains(local_pos):
            self.toggle_mode()
            return True

        # 4. Steppers +/-
        if self._btn_rects.get("part_minus") and self._btn_rects["part_minus"].contains(local_pos):
            if self._is_taylor_mode:
                self.set_taylor_degree(self._taylor_n - 2)
            else:
                self.set_partitions(self._n_partitions - 4)
            return True

        if self._btn_rects.get("part_plus") and self._btn_rects["part_plus"].contains(local_pos):
            if self._is_taylor_mode:
                self.set_taylor_degree(self._taylor_n + 2)
            else:
                self.set_partitions(self._n_partitions + 4)
            return True

        # 5. Stamp button
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
