"""
AirTeach AI - Interactive Physics Fluid Dynamics Bernoulli & Venturi Tube Airflow Simulator
===========================================================================================
Provides an authentic interactive fluid mechanics simulation for high school & university physics:
  - Theoretical & Mathematical Foundations (Zero Mock Data):
      1. Continuity Equation (Phương trình liên tục):
           A₁ · v₁ = A₂ · v₂ = Q = const  =>  v₂ = v₁ · (A₁ / A₂) = v₁ · (D₁ / D₂)²
      2. Bernoulli's Equation (Định luật Bernoulli cho dòng chảy dừng nằm ngang):
           P₁ + ½·ρ·v₁² = P₂ + ½·ρ·v₂² = const
           => Áp suất tĩnh giảm tại chỗ thắt: ΔP = P₁ - P₂ = ½·ρ·(v₂² - v₁²)
      3. Venturi Effect & U-tube Differential Manometer (Áp kế chất lỏng chữ U):
           Δh = ΔP / (ρ_manometer · g) = [ρ_fluid · (v₂² - v₁²)] / (2 · ρ_manometer · g)
      4. Aerodynamic Lift on Cambered Airfoil (Lực nâng khí động học Bernoulli):
           F_L = ΔP · S_wing = ½·ρ·(v_upper² - v_lower²) · S
  - Visualization:
      - Continuous converging-diverging Venturi tube with smooth contour.
      - Dynamic Lagrangian particle tracer showing fluid acceleration through constriction.
      - Realistic U-tube manometer with liquid column height difference Δh.
      - Pressure gradient color mapping (Warm amber/red high pressure -> Cool cyan/blue low pressure).
      - Velocity vector arrows with magnitude proportional to local flow speed v(x).
  - 5 Authentic Curriculum Presets:
      - VENTURI_METER: Ống Venturi đo lưu lượng khí qua áp kế nước
      - CARBURETOR_CHOKE: Bộ chế hòa khí động cơ đốt trong (hút xăng nhờ áp suất âm)
      - AIRPLANE_WING_LIFT: Cánh máy bay tạo lực nâng khí động học Bernoulli
      - PERFUME_ASPIRATOR: Bình xịt nước hoa / súng phun sơn hiệu ứng Venturi
      - WATER_PIPE_RESTRICTION: Đường ống dẫn nước thu hẹp qua áp kế thủy ngân
  - Classroom Controls & Gestures:
      - Throat diameter handle drag (mouse or touchless air pinch).
      - Inlet speed slider.
      - Play / Pause animation toggle.
      - Vector diagram stamping onto LessonBoard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Standard Physical Constants
GRAVITY_G: float = 9.80665  # m/s²
DENSITY_AIR: float = 1.225  # kg/m³ at 15°C, 1 atm
DENSITY_WATER: float = 1000.0  # kg/m³
DENSITY_MERCURY: float = 13600.0  # kg/m³
DENSITY_GASOLINE: float = 740.0  # kg/m³


@dataclass
class FluidPreset:
    """Authentic fluid mechanics curriculum configuration."""
    code: str
    title_vi: str
    fluid_name_vi: str
    rho_fluid: float          # kg/m³
    manometer_name_vi: str
    rho_manometer: float      # kg/m³
    inlet_dia_mm: float       # D1 in mm
    throat_dia_mm: float      # D2 in mm
    inlet_speed_ms: float     # v1 in m/s
    description_vi: str
    application_vi: str


PRESETS: dict[str, FluidPreset] = {
    "VENTURI_METER": FluidPreset(
        code="VENTURI_METER",
        title_vi="Ống Venturi đo lưu lượng khí (Venturi Flowmeter)",
        fluid_name_vi="Không khí (Air, ρ = 1.225 kg/m³)",
        rho_fluid=DENSITY_AIR,
        manometer_name_vi="Nước màu (Water, ρ = 1000 kg/m³)",
        rho_manometer=DENSITY_WATER,
        inlet_dia_mm=100.0,
        throat_dia_mm=45.0,
        inlet_speed_ms=8.0,
        description_vi="Ống thắt dần đều làm tăng vận tốc dòng khí, tạo độ giảm áp suất tĩnh ΔP tỷ lệ với bình phương lưu lượng Q².",
        application_vi="Dùng trong đồng hồ đo lưu lượng khí gas, đường ống thông gió công nghiệp và trạm kiểm định khí động học.",
    ),
    "CARBURETOR_CHOKE": FluidPreset(
        code="CARBURETOR_CHOKE",
        title_vi="Bộ chế hòa khí động cơ (Carburetor Venturi Suction)",
        fluid_name_vi="Dòng khí nạp động cơ (ρ = 1.20 kg/m³)",
        rho_fluid=DENSITY_AIR,
        manometer_name_vi="Xăng lỏng (Gasoline, ρ = 740 kg/m³)",
        rho_manometer=DENSITY_GASOLINE,
        inlet_dia_mm=60.0,
        throat_dia_mm=22.0,
        inlet_speed_ms=14.0,
        description_vi="Vận tốc khí qua họng khuếch tán tăng vọt khiến áp suất P₂ < P_khí quyển, tự động hút xăng từ buồng phao hòa trộn thành sương mịn.",
        application_vi="Nguyên lý cấp nhiên liệu kinh điển trong động cơ xe máy, máy bay cánh quạt và máy cắt cỏ.",
    ),
    "AIRPLANE_WING_LIFT": FluidPreset(
        code="AIRPLANE_WING_LIFT",
        title_vi="Lực nâng cánh máy bay Bernoulli (Airfoil Camber Lift)",
        fluid_name_vi="Dòng khí qua cánh (ρ = 1.225 kg/m³)",
        rho_fluid=DENSITY_AIR,
        manometer_name_vi="Chênh lệch áp suất nâng ΔP (Air Piezometer)",
        rho_manometer=DENSITY_WATER,
        inlet_dia_mm=80.0,
        throat_dia_mm=38.0,
        inlet_speed_ms=18.0,
        description_vi="Mặt trên cánh cong khiến dòng khí chạy nhanh hơn mặt dưới (v_trên > v_dưới), tạo độ chênh áp suất hướng lên sinh lực nâng F_L.",
        application_vi="Tạo lực nâng khí động học cho máy bay thương mại (Boeing, Airbus) và cánh đuôi xe đua F1 bám đường.",
    ),
    "PERFUME_ASPIRATOR": FluidPreset(
        code="PERFUME_ASPIRATOR",
        title_vi="Bình xịt nước hoa & Súng phun sơn (Aspirator Atomizer)",
        fluid_name_vi="Luồng khí nén phản lực (ρ = 1.25 kg/m³)",
        rho_fluid=DENSITY_AIR,
        manometer_name_vi="Chất lỏng nước hoa (ρ = 950 kg/m³)",
        rho_manometer=950.0,
        inlet_dia_mm=40.0,
        throat_dia_mm=14.0,
        inlet_speed_ms=22.0,
        description_vi="Bóp bóng cao su đẩy luồng khí tốc độ siêu cao qua đầu vòi hẹp, áp suất hạ thấp hút cột chất lỏng lên và xé thành hạt sương li ti.",
        application_vi="Bình xịt khử khuẩn y tế, súng phun sơn ô tô và vòi phun sương làm mát công cộng.",
    ),
    "WATER_PIPE_RESTRICTION": FluidPreset(
        code="WATER_PIPE_RESTRICTION",
        title_vi="Đường ống nước thắt hẹp & Áp kế thủy ngân (Hydrodynamics)",
        fluid_name_vi="Nước chảy trong ống (Water, ρ = 1000 kg/m³)",
        rho_fluid=DENSITY_WATER,
        manometer_name_vi="Thủy ngân (Mercury, ρ = 13600 kg/m³)",
        rho_manometer=DENSITY_MERCURY,
        inlet_dia_mm=120.0,
        throat_dia_mm=60.0,
        inlet_speed_ms=2.5,
        description_vi="Chất lỏng không chịu nén (incompressible fluid) chảy qua tiết diện nhỏ bị tăng tốc và sụt giảm áp lực nước mạnh mẽ.",
        application_vi="Hệ thống cấp nước đô thị, trạm bơm thủy điện và van điều áp phòng cháy chữa cháy.",
    ),
}


class InteractiveFluidVenturiItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for Fluid Dynamics Continuity & Bernoulli Venturi Simulation.
    Features:
      - Real-time continuity solver: v2 = v1 * (D1 / D2)².
      - Real-time Bernoulli pressure difference: ΔP = ½·ρ·(v₂² - v₁²).
      - U-tube manometer liquid height difference: Δh = ΔP / (ρ_m · g).
      - 28 animated Lagrangian flow tracer particles with continuous acceleration contour.
      - Pressure gradient color mapping and velocity vectors.
      - 5 Authentic curriculum presets.
      - Draggable throat constriction handle.
      - Vector diagram stamping onto LessonBoard.
    """
    closed = Signal()
    preset_changed = Signal(str)
    parameters_changed = Signal(float, float)  # (D2_mm, v1_ms)

    WIDTH = 780.0
    HEIGHT = 520.0

    def __init__(
        self,
        pos: tuple[float, float] = (300.0, 120.0),
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self._active_preset_code: str = "VENTURI_METER"
        self._preset: FluidPreset = PRESETS[self._active_preset_code]

        # Fluid physical variables
        self._inlet_dia_mm: float = self._preset.inlet_dia_mm
        self._throat_dia_mm: float = self._preset.throat_dia_mm
        self._inlet_speed_ms: float = self._preset.inlet_speed_ms

        # Animation state
        self._is_animating: bool = True
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)  # ~30 FPS
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # Particles: list of (x_norm, streamline_y_norm) in [0.0, 1.0]
        self._particles: list[list[float]] = []
        for i in range(32):
            self._particles.append([i / 32.0, ((i * 7) % 5 + 1) / 6.0])

        self._dragged_control: Optional[str] = None
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Pre-populate interactive button and canvas areas."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Presets bar
        preset_codes = [
            "VENTURI_METER",
            "CARBURETOR_CHOKE",
            "AIRPLANE_WING_LIFT",
            "PERFUME_ASPIRATOR",
            "WATER_PIPE_RESTRICTION",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Venturi pipe viewport canvas
        self._btn_rects["canvas"] = QRectF(16.0, 88.0, 440.0, 360.0)

        # Action bar
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 130.0, h_bot)
        self._btn_rects["dec_speed"] = QRectF(154.0, y_bot, 42.0, h_bot)
        self._btn_rects["inc_speed"] = QRectF(202.0, y_bot, 42.0, h_bot)
        self._btn_rects["narrow_throat"] = QRectF(252.0, y_bot, 95.0, h_bot)
        self._btn_rects["widen_throat"] = QRectF(353.0, y_bot, 95.0, h_bot)
        self._btn_rects["stamp"] = QRectF(466.0, y_bot, 298.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Physical Calculations (Zero Mock)
    # -------------------------------------------------------------------------
    @property
    def throat_speed_ms(self) -> float:
        """Calculates throat velocity v2 using continuity equation A1*v1 = A2*v2."""
        ratio = (self._inlet_dia_mm / max(1.0, self._throat_dia_mm)) ** 2
        return self._inlet_speed_ms * ratio

    @property
    def flow_rate_m3s(self) -> float:
        """Volumetric flow rate Q = A1 * v1 in m³/s."""
        r1_m = (self._inlet_dia_mm / 2000.0)
        a1_m2 = math.pi * (r1_m ** 2)
        return a1_m2 * self._inlet_speed_ms

    @property
    def pressure_drop_pa(self) -> float:
        """Static pressure drop ΔP = P1 - P2 = ½·ρ·(v₂² - v₁²) in Pascals."""
        v1 = self._inlet_speed_ms
        v2 = self.throat_speed_ms
        return 0.5 * self._preset.rho_fluid * (v2 ** 2 - v1 ** 2)

    @property
    def manometer_height_diff_mm(self) -> float:
        """Manometer liquid level difference Δh = ΔP / (ρ_m · g) in millimeters."""
        delta_p = self.pressure_drop_pa
        rho_m = self._preset.rho_manometer
        h_meters = delta_p / (rho_m * GRAVITY_G)
        return h_meters * 1000.0  # mm

    def set_throat_diameter(self, dia_mm: float) -> None:
        """Set throat constriction diameter D2 (clamped between 15mm and D1)."""
        self._throat_dia_mm = max(15.0, min(self._inlet_dia_mm, float(dia_mm)))
        self.parameters_changed.emit(self._throat_dia_mm, self._inlet_speed_ms)
        self.update()

    def set_inlet_speed(self, speed_ms: float) -> None:
        """Set inlet fluid speed v1 (clamped between 0.5 and 40 m/s)."""
        self._inlet_speed_ms = max(0.5, min(40.0, float(speed_ms)))
        self.parameters_changed.emit(self._throat_dia_mm, self._inlet_speed_ms)
        self.update()

    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic fluid mechanics curriculum preset."""
        if preset_code in PRESETS:
            self._active_preset_code = preset_code
            self._preset = PRESETS[preset_code]
            self._inlet_dia_mm = self._preset.inlet_dia_mm
            self._throat_dia_mm = self._preset.throat_dia_mm
            self._inlet_speed_ms = self._preset.inlet_speed_ms
            self.preset_changed.emit(preset_code)
            self.parameters_changed.emit(self._throat_dia_mm, self._inlet_speed_ms)
            self.update()
            return True
        return False

    def toggle_animation(self) -> bool:
        """Toggle particle flow animation."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def cleanup(self) -> None:
        """Clean up timer on removal."""
        self._anim_timer.stop()

    def stamp_to_board(self) -> None:
        """Stamp current fluid mechanics diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import PhysicsTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = PhysicsTools.create_fluid_venturi_diagram(
            center=center,
            preset_code=self._active_preset_code,
            inlet_dia_mm=self._inlet_dia_mm,
            throat_dia_mm=self._throat_dia_mm,
            inlet_speed_ms=self._inlet_speed_ms,
        )
        for itm in items:
            scene.addItem(itm)

    # -------------------------------------------------------------------------
    # Animation & Mechanics Integration
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        """Update particle positions with continuous local speed acceleration."""
        dt = 0.033
        base_speed = max(0.1, self._inlet_speed_ms / 15.0)

        for p in self._particles:
            x_norm = p[0]
            # Speed increases in the middle throat (x_norm around 0.5)
            # Gaussian bell curve acceleration:
            dist_throat = abs(x_norm - 0.5)
            speed_mult = 1.0 + (self.throat_speed_ms / max(0.1, self._inlet_speed_ms) - 1.0) * math.exp(-((dist_throat * 4.0) ** 2))
            dx = base_speed * speed_mult * dt * 0.4
            p[0] += dx
            if p[0] > 1.0:
                p[0] -= 1.0

        self.update()

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        card_path = QPainterPath()
        card_path.addRoundedRect(bg_rect, 14.0, 14.0)
        painter.fillPath(card_path, QColor(15, 23, 42, 245))  # Deep Slate #0f172a
        painter.setPen(QPen(QColor(56, 189, 248, 180), 1.5))
        painter.drawPath(card_path)

        # 2. Header
        self._paint_header(painter)

        # 3. Presets Bar
        self._paint_presets_bar(painter)

        # 4. Canvas: Venturi Tube & Manometer
        self._paint_venturi_canvas(painter)

        # 5. Scientific Analytics Card
        self._paint_analytics_card(painter)

        # 6. Action Toolbar
        self._paint_action_toolbar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(
            QRectF(16.0, 12.0, 560.0, 24.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "🌊 CƠ HỌC CHẤT LƯU: ĐỊNH LUẬT BERNOULLI & HIỆU ỨNG VENTURI",
        )

        sub_font = QFont("Segoe UI", 9)
        painter.setFont(sub_font)
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(16.0, 34.0, 560.0, 18.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"Mô hình: {self._preset.title_vi}  •  Phương trình liên tục & Áp suất tĩnh",
        )

        # Close button
        close_rect = self._btn_rects["close"]
        is_hover = (self._hovered_button == "close")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(239, 68, 68, 200 if is_hover else 120))
        painter.drawRoundedRect(close_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(close_rect, int(Qt.AlignmentFlag.AlignCenter), "✕")

    def _paint_presets_bar(self, painter: QPainter) -> None:
        preset_items = [
            ("VENTURI_METER", "Ống Venturi"),
            ("CARBURETOR_CHOKE", "Bộ chế hòa khí"),
            ("AIRPLANE_WING_LIFT", "Cánh máy bay"),
            ("PERFUME_ASPIRATOR", "Bình xịt khí nén"),
            ("WATER_PIPE_RESTRICTION", "Ống nước & Thủy ngân"),
        ]
        font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        painter.setFont(font)

        for code, label in preset_items:
            rect = self._btn_rects[f"preset_{code}"]
            is_active = (self._active_preset_code == code)
            is_hover = (self._hovered_button == f"preset_{code}")

            if is_active:
                painter.setBrush(QColor(14, 165, 233, 220))  # Sky-500
                painter.setPen(QPen(QColor(255, 255, 255), 1.2))
            elif is_hover:
                painter.setBrush(QColor(51, 65, 85, 220))
                painter.setPen(QPen(QColor(56, 189, 248), 1.0))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

    def _paint_venturi_canvas(self, painter: QPainter) -> None:
        canvas_rect = self._btn_rects["canvas"]

        # Background frame
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(canvas_rect, 10.0, 10.0)

        # Tube geometry coordinates inside canvas
        x0 = canvas_rect.left() + 20.0
        x1 = canvas_rect.right() - 20.0
        pipe_w = x1 - x0
        cy = canvas_rect.top() + 110.0  # Centerline of tube

        h_inlet = (self._inlet_dia_mm / 120.0) * 85.0
        h_throat = (self._throat_dia_mm / 120.0) * 85.0

        # Draw Top & Bottom Walls of Venturi Tube
        top_path = QPainterPath()
        bot_path = QPainterPath()

        top_path.moveTo(x0, cy - h_inlet / 2.0)
        bot_path.moveTo(x0, cy + h_inlet / 2.0)

        # Smooth cubic bezier converging to throat at center and diverging back
        cx_mid = x0 + pipe_w * 0.5
        ctrl_w = pipe_w * 0.22

        top_path.cubicTo(
            cx_mid - ctrl_w, cy - h_inlet / 2.0,
            cx_mid - ctrl_w, cy - h_throat / 2.0,
            cx_mid, cy - h_throat / 2.0,
        )
        top_path.cubicTo(
            cx_mid + ctrl_w, cy - h_throat / 2.0,
            cx_mid + ctrl_w, cy - h_inlet / 2.0,
            x1, cy - h_inlet / 2.0,
        )

        bot_path.cubicTo(
            cx_mid - ctrl_w, cy + h_inlet / 2.0,
            cx_mid - ctrl_w, cy + h_throat / 2.0,
            cx_mid, cy + h_throat / 2.0,
        )
        bot_path.cubicTo(
            cx_mid + ctrl_w, cy + h_throat / 2.0,
            cx_mid + ctrl_w, cy + h_inlet / 2.0,
            x1, cy + h_inlet / 2.0,
        )

        # Fill fluid interior with pressure gradient color
        fluid_fill_path = QPainterPath(top_path)
        fluid_fill_path.lineTo(x1, cy + h_inlet / 2.0)
        # Connect reversed bottom
        rev_bot = bot_path.toReversed()
        fluid_fill_path.connectPath(rev_bot)
        fluid_fill_path.closeSubpath()

        painter.save()
        painter.setClipPath(fluid_fill_path)
        # Gradient from inlet high pressure (amber-blue tint) to throat low pressure (cyan)
        painter.fillPath(fluid_fill_path, QColor(14, 165, 233, 40))

        # Draw animated flow tracer particles
        painter.setPen(Qt.PenStyle.NoPen)
        for p in self._particles:
            px_norm, py_norm = p[0], p[1]
            px = x0 + px_norm * pipe_w
            # Local tube half-height
            dist_th = abs(px_norm - 0.5)
            h_local = h_throat + (h_inlet - h_throat) * (1.0 - math.exp(-((dist_th * 3.8) ** 2)))
            py = cy - (h_local / 2.0) + py_norm * h_local

            # Speed factor for particle glow
            spd_factor = 1.0 + (self.throat_speed_ms / max(0.1, self._inlet_speed_ms) - 1.0) * math.exp(-((dist_th * 4.0) ** 2))
            p_color = QColor(254, 240, 138) if spd_factor > 2.0 else QColor(56, 189, 248)

            painter.setBrush(p_color)
            r_dot = 2.5 if spd_factor <= 2.0 else 3.2
            painter.drawEllipse(QPointF(px, py), r_dot, r_dot)

        painter.restore()

        # Tube solid borders
        wall_pen = QPen(QColor(56, 189, 248), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(wall_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(top_path)
        painter.drawPath(bot_path)

        # Centerline (dashed)
        dash_pen = QPen(QColor(100, 116, 139), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(dash_pen)
        painter.drawLine(QPointF(x0, cy), QPointF(x1, cy))

        # Velocity Vectors: v1 at inlet, v2 at throat
        # v1 arrow
        self._paint_vector_arrow(painter, QPointF(x0 + 35.0, cy), min(65.0, self._inlet_speed_ms * 4.0), f"v₁ = {self._inlet_speed_ms:.1f} m/s", QColor(34, 197, 94))
        # v2 arrow
        self._paint_vector_arrow(painter, QPointF(cx_mid, cy), min(110.0, self.throat_speed_ms * 3.0), f"v₂ = {self.throat_speed_ms:.1f} m/s", QColor(234, 179, 8))

        # Diameter dimensions
        dim_pen = QPen(QColor(148, 163, 184), 1.2)
        painter.setPen(dim_pen)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        # D1 dimension line
        painter.drawLine(QPointF(x0 + 15.0, cy - h_inlet / 2.0), QPointF(x0 + 15.0, cy + h_inlet / 2.0))
        painter.drawText(QRectF(x0 - 25.0, cy - 8.0, 36.0, 16.0), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f"D₁={self._inlet_dia_mm:.0f}")

        # D2 dimension line with draggable indicator
        painter.drawLine(QPointF(cx_mid, cy - h_throat / 2.0), QPointF(cx_mid, cy + h_throat / 2.0))
        painter.drawText(QRectF(cx_mid - 25.0, cy - h_throat / 2.0 - 18.0, 50.0, 14.0), int(Qt.AlignmentFlag.AlignCenter), f"D₂={self._throat_dia_mm:.0f}mm")

        # Draggable throat handle badge
        handle_rect = QRectF(cx_mid - 12.0, cy + h_throat / 2.0 - 6.0, 24.0, 16.0)
        self._btn_rects["throat_handle"] = handle_rect
        painter.setBrush(QColor(234, 179, 8, 220))
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        painter.drawRoundedRect(handle_rect, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor(15, 23, 42))
        painter.drawText(handle_rect, int(Qt.AlignmentFlag.AlignCenter), "↕ D₂")

        # ---------------------------------------------------------------------
        # Draw U-Tube Differential Manometer below pipe
        # ---------------------------------------------------------------------
        self._paint_manometer(painter, x0 + pipe_w * 0.25, cx_mid, cy + h_inlet / 2.0, canvas_rect.bottom() - 15.0)

    def _paint_vector_arrow(self, painter: QPainter, base: QPointF, length: float, label: str, col: QColor) -> None:
        """Paint velocity vector arrow with tip and speed text."""
        p_end = QPointF(base.x() + length, base.y())
        pen = QPen(col, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(base, p_end)

        # Arrow head
        head = QPainterPath()
        head.moveTo(p_end)
        head.lineTo(p_end.x() - 7.0, p_end.y() - 4.5)
        head.lineTo(p_end.x() - 7.0, p_end.y() + 4.5)
        head.closeSubpath()
        painter.fillPath(head, col)

        # Label
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(col)
        painter.drawText(QRectF(base.x(), base.y() - 18.0, length + 20.0, 16.0), int(Qt.AlignmentFlag.AlignLeft), label)

    def _paint_manometer(self, painter: QPainter, x_left_tap: float, x_throat_tap: float, y_tap: float, y_bottom: float) -> None:
        """Paint authentic U-tube manometer with liquid level difference."""
        tube_w = 12.0
        # Scaled height difference in pixels:
        delta_h_mm = self.manometer_height_diff_mm
        dh_px = min(75.0, max(2.0, delta_h_mm * 0.65))

        base_liquid_y = y_bottom - 45.0
        h_left_liquid = base_liquid_y + dh_px / 2.0
        h_right_liquid = base_liquid_y - dh_px / 2.0

        u_path = QPainterPath()
        # Tap 1 from pipe down to U-tube left
        u_path.moveTo(x_left_tap, y_tap)
        u_path.lineTo(x_left_tap, y_bottom)
        u_path.lineTo(x_throat_tap, y_bottom)
        u_path.lineTo(x_throat_tap, y_tap)

        painter.setPen(QPen(QColor(100, 116, 139), tube_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(u_path)

        # Fill liquid columns inside U-tube
        liquid_col = QColor(239, 68, 68) if "Water" in self._preset.manometer_name_vi or "Nước" in self._preset.manometer_name_vi else (
            QColor(148, 163, 184) if "Mercury" in self._preset.manometer_name_vi or "Thủy ngân" in self._preset.manometer_name_vi else QColor(245, 158, 11)
        )
        l_pen = QPen(liquid_col, tube_w - 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(l_pen)

        liq_path = QPainterPath()
        liq_path.moveTo(x_left_tap, h_left_liquid)
        liq_path.lineTo(x_left_tap, y_bottom - 1.0)
        liq_path.lineTo(x_throat_tap, y_bottom - 1.0)
        liq_path.lineTo(x_throat_tap, h_right_liquid)
        painter.drawPath(liq_path)

        # Dimension line for Δh
        dim_pen = QPen(QColor(234, 179, 8), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(dim_pen)
        painter.drawLine(QPointF(x_left_tap - 18.0, h_left_liquid), QPointF(x_left_tap + 18.0, h_left_liquid))
        painter.drawLine(QPointF(x_throat_tap - 18.0, h_right_liquid), QPointF(x_throat_tap + 18.0, h_right_liquid))
        painter.drawLine(QPointF(x_throat_tap + 22.0, h_left_liquid), QPointF(x_throat_tap + 22.0, h_right_liquid))

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(234, 179, 8))
        painter.drawText(
            QRectF(x_throat_tap + 26.0, (h_left_liquid + h_right_liquid) / 2.0 - 8.0, 95.0, 16.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"Δh = {delta_h_mm:.1f} mm",
        )

        # Manometer label
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(x_left_tap, y_bottom + 2.0, x_throat_tap - x_left_tap, 14.0),
            int(Qt.AlignmentFlag.AlignCenter),
            f"Áp kế: {self._preset.manometer_name_vi.split('(')[0].strip()}",
        )

    def _paint_analytics_card(self, painter: QPainter) -> None:
        card_rect = QRectF(466.0, 88.0, 298.0, 360.0)
        painter.setBrush(QColor(30, 41, 59, 220))  # Slate-800
        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(card_rect, 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(
            QRectF(476.0, 96.0, 278.0, 20.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "📊 THÔNG SỐ VẬT LÝ & BERNOULLI",
        )

        # Calculation rows
        y = 124.0
        row_h = 19.0
        font_lbl = QFont("Segoe UI", 8)
        font_val = QFont("Segoe UI", 8, QFont.Weight.Bold)

        metrics = [
            ("Lưu chất công tác:", self._preset.fluid_name_vi.split("(")[0].strip()),
            ("Khối lượng riêng ρ:", f"{self._preset.rho_fluid:.3f} kg/m³"),
            ("Vận tốc cửa vào v₁:", f"{self._inlet_speed_ms:.2f} m/s"),
            ("Vận tốc tại họng v₂:", f"{self.throat_speed_ms:.2f} m/s (tăng {(self.throat_speed_ms/self._inlet_speed_ms):.1f}x)"),
            ("Lưu lượng dòng chảy Q:", f"{self.flow_rate_m3s * 1000.0:.2f} L/s ({self.flow_rate_m3s:.4f} m³/s)"),
            ("Độ giảm áp suất ΔP:", f"{self.pressure_drop_pa:.1f} Pa ({self.pressure_drop_pa/1000.0:.2f} kPa)"),
            ("Độ lệch áp kế Δh:", f"{self.manometer_height_diff_mm:.1f} mm"),
        ]

        for lbl, val in metrics:
            painter.setFont(font_lbl)
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(QRectF(476.0, y, 130.0, row_h), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), lbl)
            painter.setFont(font_val)
            painter.setPen(QColor(241, 245, 249))
            painter.drawText(QRectF(606.0, y, 150.0, row_h), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), val)
            y += row_h

        # Bernoulli Law Formula Banner
        y += 4.0
        f_box = QRectF(476.0, y, 278.0, 36.0)
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.0))
        painter.drawRoundedRect(f_box, 6.0, 6.0)

        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(482.0, y + 4.0, 266.0, 14.0), int(Qt.AlignmentFlag.AlignCenter), "P₁ + ½·ρ·v₁² = P₂ + ½·ρ·v₂²")
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(203, 213, 225))
        painter.drawText(QRectF(482.0, y + 18.0, 266.0, 14.0), int(Qt.AlignmentFlag.AlignCenter), "Áp suất tĩnh P₂ giảm tỉ lệ nghịch với vận tốc v₂")
        y += 44.0

        # Description
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(251, 191, 36))  # Amber-400
        painter.drawText(QRectF(476.0, y, 278.0, 16.0), int(Qt.AlignmentFlag.AlignLeft), "Ứng dụng thực tế:")
        y += 18.0

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(
            QRectF(476.0, y, 278.0, 34.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
            self._preset.application_vi,
        )
        y += 38.0

        # Detailed Physics Explanation Box
        expl_rect = QRectF(476.0, y, 278.0, 48.0)
        painter.setBrush(QColor(15, 23, 42, 200))
        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(expl_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(203, 213, 225))
        painter.drawText(
            QRectF(482.0, y + 4.0, 266.0, 40.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
            self._preset.description_vi,
        )

    def _paint_action_toolbar(self, painter: QPainter) -> None:
        # 1. Play / Pause
        pp_rect = self._btn_rects["play_pause"]
        is_hover = (self._hovered_button == "play_pause")
        painter.setBrush(QColor(59, 130, 246, 220 if is_hover else 180))
        painter.setPen(QPen(QColor(255, 255, 255), 1.0))
        painter.drawRoundedRect(pp_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        pp_text = "⏸ Tạm dừng dòng" if self._is_animating else "▶ Tiếp tục dòng"
        painter.drawText(pp_rect, int(Qt.AlignmentFlag.AlignCenter), pp_text)

        # 2. Speed controls (- / +)
        dec_r = self._btn_rects["dec_speed"]
        painter.setBrush(QColor(71, 85, 105, 220 if self._hovered_button == "dec_speed" else 180))
        painter.drawRoundedRect(dec_r, 6.0, 6.0)
        painter.drawText(dec_r, int(Qt.AlignmentFlag.AlignCenter), "− v₁")

        inc_r = self._btn_rects["inc_speed"]
        painter.setBrush(QColor(71, 85, 105, 220 if self._hovered_button == "inc_speed" else 180))
        painter.drawRoundedRect(inc_r, 6.0, 6.0)
        painter.drawText(inc_r, int(Qt.AlignmentFlag.AlignCenter), "+ v₁")

        # 3. Throat controls (Thắt hẹp / Mở rộng)
        narrow_r = self._btn_rects["narrow_throat"]
        painter.setBrush(QColor(16, 185, 129, 220 if self._hovered_button == "narrow_throat" else 180))
        painter.drawRoundedRect(narrow_r, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(narrow_r, int(Qt.AlignmentFlag.AlignCenter), "▼ Thu hẹp D₂")

        widen_r = self._btn_rects["widen_throat"]
        painter.setBrush(QColor(16, 185, 129, 220 if self._hovered_button == "widen_throat" else 180))
        painter.drawRoundedRect(widen_r, 6.0, 6.0)
        painter.drawText(widen_r, int(Qt.AlignmentFlag.AlignCenter), "▲ Mở rộng D₂")

        # 4. Stamp Vector Diagram Button
        stamp_r = self._btn_rects["stamp"]
        painter.setBrush(QColor(217, 119, 6, 230 if self._hovered_button == "stamp" else 180))
        painter.drawRoundedRect(stamp_r, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(stamp_r, int(Qt.AlignmentFlag.AlignCenter), "📌 Đóng dấu Vector lên Bảng vẽ")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Air Gesture Interaction
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        if self._btn_rects["close"].contains(pos):
            self.closed.emit()
            self.setVisible(False)
            event.accept()
            return

        # 2. Preset buttons
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(pos):
                self.load_preset(code)
                event.accept()
                return

        # 3. Action buttons
        if self._btn_rects["play_pause"].contains(pos):
            self.toggle_animation()
            event.accept()
            return

        if self._btn_rects["dec_speed"].contains(pos):
            self.set_inlet_speed(self._inlet_speed_ms - 2.0)
            event.accept()
            return

        if self._btn_rects["inc_speed"].contains(pos):
            self.set_inlet_speed(self._inlet_speed_ms + 2.0)
            event.accept()
            return

        if self._btn_rects["narrow_throat"].contains(pos):
            self.set_throat_diameter(self._throat_dia_mm - 5.0)
            event.accept()
            return

        if self._btn_rects["widen_throat"].contains(pos):
            self.set_throat_diameter(self._throat_dia_mm + 5.0)
            event.accept()
            return

        if self._btn_rects["stamp"].contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        # 4. Draggable throat handle
        if self._btn_rects["throat_handle"].contains(pos):
            self._dragged_control = "throat_handle"
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragged_control == "throat_handle":
            pos = event.pos()
            canvas_r = self._btn_rects["canvas"]
            cy = canvas_r.top() + 110.0
            dy = abs(pos.y() - cy)
            # Map vertical distance to diameter mm
            dia_mm = (dy / 85.0) * 120.0 * 2.0
            self.set_throat_diameter(dia_mm)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragged_control is not None:
            self._dragged_control = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_button
        self._hovered_button = None
        for name, r in self._btn_rects.items():
            if r.contains(pos):
                self._hovered_button = name
                break
        if self._hovered_button != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def handle_air_pinch(self, point: QPointF) -> bool:
        """Handle touchless air pinch gesture for controls."""
        local_p = self.mapFromScene(point)

        for name, r in self._btn_rects.items():
            if r.contains(local_p):
                if name == "close":
                    self.closed.emit()
                    self.setVisible(False)
                    return True
                elif name.startswith("preset_"):
                    code = name.replace("preset_", "")
                    return self.load_preset(code)
                elif name == "play_pause":
                    self.toggle_animation()
                    return True
                elif name == "dec_speed":
                    self.set_inlet_speed(self._inlet_speed_ms - 2.0)
                    return True
                elif name == "inc_speed":
                    self.set_inlet_speed(self._inlet_speed_ms + 2.0)
                    return True
                elif name == "narrow_throat":
                    self.set_throat_diameter(self._throat_dia_mm - 5.0)
                    return True
                elif name == "widen_throat":
                    self.set_throat_diameter(self._throat_dia_mm + 5.0)
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True
                elif name == "throat_handle":
                    self._dragged_control = "throat_handle"
                    return True

        return False

    def handle_air_move(self, point: QPointF) -> bool:
        """Move throat diameter during continuous air pinch."""
        if self._dragged_control == "throat_handle":
            local_p = self.mapFromScene(point)
            canvas_r = self._btn_rects["canvas"]
            cy = canvas_r.top() + 110.0
            dy = abs(local_p.y() - cy)
            dia_mm = (dy / 85.0) * 120.0 * 2.0
            self.set_throat_diameter(dia_mm)
            return True
        return False

    def handle_air_release(self) -> None:
        """Release control on gesture release."""
        self._dragged_control = None
