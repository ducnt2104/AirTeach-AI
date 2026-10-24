"""
AirTeach AI — Interactive Physics & Quantum Mechanics: Stern-Gerlach Experiment,
Space Quantization & Electron Spin 1/2 Simulator.
Simulates:
  - Inhomogeneous magnetic field gradient dB_z / dz and magnetic force F_z = mu_z * (dB_z / dz)
  - Intrinsic electron spin angular momentum S = 1/2, m_s = ±1/2, mu_z = ∓mu_B
  - Discrete two-beam splitting on detector screen vs classical continuous smudge
  - Sequential multi-stage Stern-Gerlach apparatus (SG_z -> SG_x -> SG_z)
  - Quantum measurement postulate, state vector projection P(up_theta) = cos^2(theta/2),
    and Pauli spin non-commutativity [S_x, S_z] != 0
"""

from __future__ import annotations
import math
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
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class SternGerlachPreset:
    """Represents a curriculum scenario for the Stern-Gerlach experiment."""
    code: str
    title_vi: str
    badge_label: str
    gradient_T_per_m: float  # Magnetic field gradient dB_z / dz in T/m
    magnet_angle_deg: float  # Angle theta of magnet axis (0 = Z-axis, 90 = X-axis)
    oven_temperature_K: float  # Thermal atomic oven temperature (Kelvin)
    is_classical_compare: bool
    is_sequential_multi_stage: bool
    description: str


STERN_GERLACH_PRESETS: dict[str, SternGerlachPreset] = {
    "SINGLE_STAGE_SILVER_ATOMS": SternGerlachPreset(
        code="SINGLE_STAGE_SILVER_ATOMS",
        title_vi="Thí nghiệm Stern-Gerlach 1922 — Chùm Nguyên tử Bạc & Tách 2 Vạch Rời rạc",
        badge_label="SILVER SG 1922",
        gradient_T_per_m=1200.0,
        magnet_angle_deg=0.0,  # Z-axis
        oven_temperature_K=1300.0,
        is_classical_compare=False,
        is_sequential_multi_stage=False,
        description="Thí nghiệm lịch sử của Otto Stern & Walther Gerlach (Giải Nobel 1943). Chùm nguyên tử Bạc bay qua từ trường không đồng nhất với gradient dHz/dz ~ 10³ T/m bị tách thành đúng 2 vạch phân biệt trên kính hứng, chứng minh spin electron bị lượng tử hóa không gian m_s = ±1/2.",
    ),
    "CLASSICAL_SMUDGE_VS_QUANTUM_SPLIT": SternGerlachPreset(
        code="CLASSICAL_SMUDGE_VS_QUANTUM_SPLIT",
        title_vi="So sánh Dự đoán Cổ điển (Vết Mờ Liên tục) vs Thực tế Lượng tử (Tách Đôi)",
        badge_label="CLASSICAL VS QUANTUM",
        gradient_T_per_m=1400.0,
        magnet_angle_deg=0.0,
        oven_temperature_K=1200.0,
        is_classical_compare=True,
        is_sequential_multi_stage=False,
        description="Vật lý cổ điển dự đoán mômen từ mu định hướng ngẫu nhiên liên tục trong không gian (cos theta ∈ [-1, +1]), tạo ra dải vệt mờ kéo dài liên tục. Thực nghiệm lượng tử chứng minh chỉ xuất hiện 2 vết tập trung rời rạc ở 2 cực.",
    ),
    "SEQUENTIAL_SGZ_SGX_SGZ_PARADOX": SternGerlachPreset(
        code="SEQUENTIAL_SGZ_SGX_SGZ_PARADOX",
        title_vi="Hệ Thống 3 Tầng Nối tiếp (SG_z → SG_x → SG_z) & Nghịch lý Phép đo Lượng tử",
        badge_label="SEQUENTIAL 3-STAGE",
        gradient_T_per_m=1500.0,
        magnet_angle_deg=0.0,
        oven_temperature_K=1300.0,
        is_classical_compare=False,
        is_sequential_multi_stage=True,
        description="Chùm spin-up |↑z⟩ sau khi lọc qua SG_z đi tiếp vào SG_x bị tách thành 50% |↑x⟩ và 50% |↓x⟩. Khi lọc lấy |↑x⟩ rồi cho qua SG_z lần nữa, trạng thái spin-down |↓z⟩ (vốn đã bị chặn ở tầng 1) bất ngờ xuất hiện lại với xác suất 50% do toán tử spin không giao hoán [Sx, Sz] ≠ 0.",
    ),
    "ARBITRARY_MAGNET_ANGLE_THETA": SternGerlachPreset(
        code="ARBITRARY_MAGNET_ANGLE_THETA",
        title_vi="Góc Nam Châm Tùy ý θ — Quy luật Xác suất Malus Lượng tử P = cos²(θ/2)",
        badge_label="ANGLE THETA SCAN",
        gradient_T_per_m=1200.0,
        magnet_angle_deg=60.0,  # 60 degrees tilt
        oven_temperature_K=1300.0,
        is_classical_compare=False,
        is_sequential_multi_stage=False,
        description="Khi xoay trục từ trường góc θ so với phương ban đầu, xác suất tìm thấy hạt ở trạng thái spin-up tuân theo quy tắc lượng tử P(↑_θ) = cos²(θ/2) và spin-down P(↓_θ) = sin²(θ/2). Tại θ = 60°: P(↑) = cos²(30°) = 75%, P(↓) = 25%.",
    ),
    "MAGNETIC_GRADIENT_DEFLECTION_SCALING": SternGerlachPreset(
        code="MAGNETIC_GRADIENT_DEFLECTION_SCALING",
        title_vi="Độ Lệch Quỹ Đạo Phụ Thuộc Gradient Từ trường dB_z/dz (100 - 2500 T/m)",
        badge_label="GRADIENT SCALING",
        gradient_T_per_m=2200.0,
        magnet_angle_deg=0.0,
        oven_temperature_K=900.0,  # Lower T gives slower atoms and larger deflection
        is_classical_compare=False,
        is_sequential_multi_stage=False,
        description="Độ lệch góc của chùm hạt tỷ lệ thuận với gradient từ trường F_z = ±mu_B * (dB_z/dz) và tỷ lệ nghịch với động năng nhiệt E_k = (3/2)k_B T. Gradient cực đại 2200 T/m tại nhiệt độ 900 K cho độ tách chùm cực đại rõ nét.",
    ),
}


class InteractiveSternGerlachItem(QGraphicsObject):
    """
    Interactive Physics & Quantum Mechanics: Stern-Gerlach Experiment Simulator.
    Simulates:
      - Force on magnetic dipole in gradient field: F_z = mu_z * (dB_z / dz)
      - Electron intrinsic spin 1/2: S_z = ±hbar / 2, mu_z = ∓mu_B
      - Space quantization into 2 distinct beam tracks
      - Sequential 3-stage apparatus (SG_z -> SG_x -> SG_z) with quantum state projections
      - Rotating magnetic gradient angle theta and P(up) = cos^2(theta/2)
    """

    preset_changed = Signal(str)
    physics_computed = Signal(float, float, float)  # deflection_mm, p_up, p_down
    stamp_requested = Signal(dict)

    CARD_W = 960.0
    CARD_H = 650.0

    # Physical Constants
    BOHR_MAGNETON_J_PER_T = 9.274010e-24  # mu_B in J/T
    BOLTZMANN_K_J_PER_K = 1.380649e-23     # k_B in J/K
    SILVER_MASS_KG = 1.791e-25             # 107.87 amu in kg

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation parameters
        self._active_preset_code: str = "SINGLE_STAGE_SILVER_ATOMS"
        self._preset: SternGerlachPreset = STERN_GERLACH_PRESETS[self._active_preset_code]

        self._gradient_T_per_m: float = self._preset.gradient_T_per_m
        self._magnet_angle_deg: float = self._preset.magnet_angle_deg
        self._oven_temp_K: float = self._preset.oven_temperature_K
        self._is_classical_mode: bool = self._preset.is_classical_compare
        self._is_sequential_mode: bool = self._preset.is_sequential_multi_stage

        # Animated beam particles: list of dicts {x, y, vy, spin_sign, progress}
        self._particles: list[dict] = []
        self._particle_spawn_tick: int = 0
        self._is_running: bool = True

        # UI & Touchless buttons
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
    # Physical & Quantum Mechanical Formulations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> SternGerlachPreset:
        return self._preset

    @property
    def most_probable_velocity_m_per_s(self) -> float:
        """Maxwell-Boltzmann most probable speed v_p = sqrt(2 * k_B * T / m)."""
        temp = max(100.0, self._oven_temp_K)
        return math.sqrt(2.0 * self.BOLTZMANN_K_J_PER_K * temp / self.SILVER_MASS_KG)

    @property
    def magnetic_deflection_force_N(self) -> float:
        """Magnetic force magnitude F_z = mu_B * (dB_z / dz)."""
        return self.BOHR_MAGNETON_J_PER_T * self._gradient_T_per_m

    @property
    def beam_deflection_mm(self) -> float:
        """
        Calculates physical deflection z on detector screen:
        z = (1/2) * (F_z / m) * (L_magnet / v)^2 + (v_z * L_drift / v)
        Assuming L_magnet = 0.05 m (5 cm) and L_drift = 0.10 m (10 cm).
        """
        f_z = self.magnetic_deflection_force_N
        m = self.SILVER_MASS_KG
        v = self.most_probable_velocity_m_per_s
        l_mag = 0.05
        l_drift = 0.10

        t_mag = l_mag / v
        t_drift = l_drift / v

        # Acceleration a_z = F_z / m
        a_z = f_z / m
        v_z = a_z * t_mag
        z_deflection = 0.5 * a_z * (t_mag ** 2) + v_z * t_drift
        return z_deflection * 1000.0  # in mm

    @property
    def quantum_prob_spin_up(self) -> float:
        """P(up_theta) = cos^2(theta / 2)."""
        th_rad = math.radians(self._magnet_angle_deg)
        return math.cos(th_rad / 2.0) ** 2.0

    @property
    def quantum_prob_spin_down(self) -> float:
        """P(down_theta) = sin^2(theta / 2)."""
        th_rad = math.radians(self._magnet_angle_deg)
        return math.sin(th_rad / 2.0) ** 2.0

    # -------------------------------------------------------------------------
    # Simulation Control & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in STERN_GERLACH_PRESETS:
            return
        self._active_preset_code = code
        self._preset = STERN_GERLACH_PRESETS[code]
        self._gradient_T_per_m = self._preset.gradient_T_per_m
        self._magnet_angle_deg = self._preset.magnet_angle_deg
        self._oven_temp_K = self._preset.oven_temperature_K
        self._is_classical_mode = self._preset.is_classical_compare
        self._is_sequential_mode = self._preset.is_sequential_multi_stage
        self._particles.clear()

        self._layout_buttons()
        self.preset_changed.emit(code)
        self.physics_computed.emit(
            self.beam_deflection_mm,
            self.quantum_prob_spin_up,
            self.quantum_prob_spin_down,
        )
        self.update()

    def set_magnet_angle(self, angle_deg: float) -> None:
        """Sets magnet gradient orientation angle theta [0.0, 180.0]."""
        self._magnet_angle_deg = max(0.0, min(180.0, angle_deg))
        self.physics_computed.emit(
            self.beam_deflection_mm,
            self.quantum_prob_spin_up,
            self.quantum_prob_spin_down,
        )
        self.update()

    def adjust_gradient(self, delta_T_per_m: float) -> None:
        """Adjusts field gradient dB_z / dz [200.0, 3000.0 T/m]."""
        self._gradient_T_per_m = max(200.0, min(3000.0, self._gradient_T_per_m + delta_T_per_m))
        self.physics_computed.emit(
            self.beam_deflection_mm,
            self.quantum_prob_spin_up,
            self.quantum_prob_spin_down,
        )
        self.update()

    def adjust_temperature(self, delta_K: float) -> None:
        """Adjusts oven thermal source temperature [300.0, 2000.0 K]."""
        self._oven_temp_K = max(300.0, min(2000.0, self._oven_temp_K + delta_K))
        self.physics_computed.emit(
            self.beam_deflection_mm,
            self.quantum_prob_spin_up,
            self.quantum_prob_spin_down,
        )
        self.update()

    def toggle_classical_comparison(self) -> None:
        """Toggles classical continuous smudge mode."""
        self._is_classical_mode = not self._is_classical_mode
        self._particles.clear()
        self.update()

    def toggle_sequential_mode(self) -> None:
        """Toggles 3-stage sequential Stern-Gerlach mode."""
        self._is_sequential_mode = not self._is_sequential_mode
        self._particles.clear()
        self.update()

    def _on_sim_tick(self) -> None:
        """Performs animation step for particle beam trajectories."""
        if not self._is_running:
            return

        self._particle_spawn_tick += 1
        if self._particle_spawn_tick % 2 == 0:
            # Spawn a new atom particle from oven slit
            import random
            if self._is_classical_mode:
                # Random classical cos(theta) in [-1, +1]
                spin_val = random.uniform(-1.0, 1.0)
            else:
                # Quantum discrete projection based on P(up) and P(down)
                p_up = self.quantum_prob_spin_up
                spin_val = +1.0 if random.random() < p_up else -1.0

            self._particles.append({
                "progress": 0.0,
                "spin_val": spin_val,
            })

        # Advance particles
        alive_particles = []
        for p in self._particles:
            p["progress"] += 0.035
            if p["progress"] < 1.05:
                alive_particles.append(p)
        self._particles = alive_particles
        self.update()

    # -------------------------------------------------------------------------
    # Layout & Touchless Hit-Testing
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Lays out interactive touchless action buttons."""
        self._buttons.clear()
        by = 595.0
        bh = 38.0

        self._buttons["btn_preset"] = QRectF(20.0, by, 160.0, bh)
        self._buttons["btn_th_down"] = QRectF(188.0, by, 68.0, bh)
        self._buttons["btn_th_up"] = QRectF(262.0, by, 68.0, bh)
        self._buttons["btn_grad_down"] = QRectF(338.0, by, 66.0, bh)
        self._buttons["btn_grad_up"] = QRectF(410.0, by, 66.0, bh)
        self._buttons["btn_temp_down"] = QRectF(484.0, by, 62.0, bh)
        self._buttons["btn_temp_up"] = QRectF(552.0, by, 62.0, bh)
        self._buttons["btn_classical"] = QRectF(622.0, by, 116.0, bh)
        self._buttons["btn_seq"] = QRectF(744.0, by, 104.0, bh)
        self._buttons["btn_stamp"] = QRectF(856.0, by, 84.0, bh)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Painting Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_apparatus_beam_panel(painter)
        self._paint_detector_screen_panel(painter)
        self._paint_sequential_or_geometry_panel(painter)
        self._paint_metrics_panel(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        """Renders main dark acrylic glass card background."""
        grad = QLinearGradient(0.0, 0.0, self.CARD_W, self.CARD_H)
        grad.setColorAt(0.0, QColor(10, 15, 29, 250))
        grad.setColorAt(1.0, QColor(4, 7, 16, 252))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(self.boundingRect(), 16.0, 16.0)

    def _paint_header(self, painter: QPainter) -> None:
        """Renders card header, badges, and curriculum title."""
        hx = 20.0
        hy = 20.0

        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(hx, hy + 18.0, "THÍ NGHIỆM STERN - GERLACH & LƯỢNG TỬ HÓA SPIN 1/2")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(hx, hy + 35.0, self._preset.title_vi)

        # Badges on right side
        bx = self.CARD_W - 20.0
        badges = [
            (f"Δz = ±{self.beam_deflection_mm:.2f} mm", QColor(168, 85, 247)),
            (f"dB/dz = {self._gradient_T_per_m:.0f} T/m", QColor(14, 165, 233)),
            (f"θ = {self._magnet_angle_deg:.0f}°", QColor(16, 185, 129)),
            (self._preset.badge_label, QColor(245, 158, 11)),
        ]
        for text, color in badges:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            text_w = painter.fontMetrics().horizontalAdvance(text) + 16.0
            bx -= text_w
            badge_rect = QRectF(bx, hy + 8.0, text_w, 22.0)
            painter.setBrush(QBrush(color.darker(240)))
            painter.setPen(QPen(color, 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QPen(color.lighter(130)))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)
            bx -= 8.0

    def _paint_apparatus_beam_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Left Panel: Inhomogeneous Magnet Poles, Collimated Beam & Deflection Trajectories.
        """
        ax = 20.0
        ay = 70.0
        aw = 500.0
        ah = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(ax, ay, aw, ah), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(ax + 12.0, ay + 18.0, "THIẾT BỊ NAM CHÂM GRADIENT & QUỸ ĐẠO CHÙM HẠT:")

        # Status badge: Quantum vs Classical
        status_txt = "CỔ ĐIỂN (VẾT MỜ LIÊN TỤC)" if self._is_classical_mode else "LƯỢNG TỬ (TÁCH 2 VẠCH RỜI RẠC)"
        status_col = QColor(245, 158, 11) if self._is_classical_mode else QColor(16, 185, 129)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        badge_w = painter.fontMetrics().horizontalAdvance(status_txt) + 14.0
        badge_rect = QRectF(ax + aw - badge_w - 12.0, ay + 6.0, badge_w, 18.0)
        painter.setBrush(QBrush(status_col.darker(250)))
        painter.setPen(QPen(status_col, 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)
        painter.setPen(QPen(status_col.lighter(130)))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, status_txt)

        # Apparatus Diagram Regions
        oven_x = ax + 30.0
        axis_y = ay + ah / 2.0 + 8.0

        # 1. Thermal Oven Source (Glowing orange chamber)
        oven_rect = QRectF(oven_x - 18.0, axis_y - 28.0, 36.0, 56.0)
        oven_grad = QLinearGradient(oven_rect.topLeft(), oven_rect.bottomRight())
        oven_grad.setColorAt(0.0, QColor("#f97316"))
        oven_grad.setColorAt(1.0, QColor("#7c2d12"))
        painter.setBrush(QBrush(oven_grad))
        painter.setPen(QPen(QColor("#fdba74"), 1.2))
        painter.drawRoundedRect(oven_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(QRectF(oven_x - 18.0, axis_y - 20.0, 36.0, 24.0), Qt.AlignmentFlag.AlignCenter, "Lò Ag")
        painter.setFont(QFont("Segoe UI", 5))
        painter.drawText(QRectF(oven_x - 18.0, axis_y + 4.0, 36.0, 16.0), Qt.AlignmentFlag.AlignCenter, f"{self._oven_temp_K:.0f}K")

        # 2. Collimator Slits
        slit1_x = oven_x + 38.0
        slit2_x = slit1_x + 30.0
        for sx in (slit1_x, slit2_x):
            painter.setPen(QPen(QColor("#64748b"), 2.5))
            painter.drawLine(QPointF(sx, axis_y - 32.0), QPointF(sx, axis_y - 6.0))
            painter.drawLine(QPointF(sx, axis_y + 6.0), QPointF(sx, axis_y + 32.0))

        # 3. Inhomogeneous Magnet Poles
        mag_x = slit2_x + 35.0
        mag_w = 120.0
        # Upper Knife-Edge Pole (North)
        north_poly = QPolygonF()
        north_poly.append(QPointF(mag_x, axis_y - 60.0))
        north_poly.append(QPointF(mag_x + mag_w, axis_y - 60.0))
        north_poly.append(QPointF(mag_x + mag_w, axis_y - 32.0))
        north_poly.append(QPointF(mag_x + mag_w * 0.5, axis_y - 18.0))  # Sharp tip
        north_poly.append(QPointF(mag_x, axis_y - 32.0))

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#ef4444"), 1.4))
        painter.drawPolygon(north_poly)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ef4444")))
        painter.drawText(mag_x + mag_w * 0.5 - 6.0, axis_y - 38.0, "N")

        # Lower Flat/Grooved Pole (South)
        south_poly = QPolygonF()
        south_poly.append(QPointF(mag_x, axis_y + 24.0))
        south_poly.append(QPointF(mag_x + mag_w * 0.5, axis_y + 18.0))  # Concave dip
        south_poly.append(QPointF(mag_x + mag_w, axis_y + 24.0))
        south_poly.append(QPointF(mag_x + mag_w, axis_y + 60.0))
        south_poly.append(QPointF(mag_x, axis_y + 60.0))

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#3b82f6"), 1.4))
        painter.drawPolygon(south_poly)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#3b82f6")))
        painter.drawText(mag_x + mag_w * 0.5 - 6.0, axis_y + 48.0, "S")

        # Inhomogeneous magnetic field gradient lines between poles (dashed cyan)
        painter.setPen(QPen(QColor(6, 182, 212, 110), 1.0, Qt.PenStyle.DotLine))
        for fx in range(int(mag_x + 15), int(mag_x + mag_w - 10), 18):
            painter.drawLine(QPointF(fx, axis_y - 16.0), QPointF(fx, axis_y + 16.0))

        # 4. Detector Glass Plate
        screen_x = ax + aw - 45.0
        painter.setPen(QPen(QColor("#e2e8f0"), 3.0))
        painter.drawLine(QPointF(screen_x, axis_y - 70.0), QPointF(screen_x, axis_y + 70.0))

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(screen_x - 12.0, axis_y - 74.0, "Kính hứng")

        # Central Reference Optical Axis
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(oven_x, axis_y), QPointF(screen_x, axis_y))

        # Trajectory Paths
        # Upper trajectory: Spin-Up |↑z⟩
        deflection_px = min(55.0, max(12.0, self.beam_deflection_mm * 22.0))

        path_up = QPainterPath()
        path_up.moveTo(slit2_x, axis_y)
        path_up.quadTo(mag_x + mag_w * 0.6, axis_y - deflection_px * 0.4, screen_x, axis_y - deflection_px)

        path_down = QPainterPath()
        path_down.moveTo(slit2_x, axis_y)
        path_down.quadTo(mag_x + mag_w * 0.6, axis_y + deflection_px * 0.4, screen_x, axis_y + deflection_px)

        if self._is_classical_mode:
            # Draw continuous envelope of classical trajectories
            painter.setPen(QPen(QColor(245, 158, 11, 70), 1.0))
            for frac in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
                p_c = QPainterPath()
                p_c.moveTo(slit2_x, axis_y)
                p_c.quadTo(mag_x + mag_w * 0.6, axis_y + deflection_px * 0.4 * frac, screen_x, axis_y + deflection_px * frac)
                painter.drawPath(p_c)

        painter.setPen(QPen(QColor("#10b981"), 1.8))
        painter.drawPath(path_up)

        painter.setPen(QPen(QColor("#f43f5e"), 1.8))
        painter.drawPath(path_down)

        # Labels for beam outputs
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(screen_x + 6.0, axis_y - deflection_px + 4.0, f"|↑z⟩ ({self.quantum_prob_spin_up * 100:.0f}%)")

        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(screen_x + 6.0, axis_y + deflection_px + 4.0, f"|↓z⟩ ({self.quantum_prob_spin_down * 100:.0f}%)")

        # Animate particles flying along the trajectories
        for p in self._particles:
            prog = p["progress"]
            spin_val = p["spin_val"]
            # X position from oven to screen
            px = oven_x + prog * (screen_x - oven_x)

            if px < slit2_x:
                py = axis_y
            elif px < mag_x:
                py = axis_y
            else:
                # Inside and past magnet: parabolic curvature
                sub_prog = (px - slit2_x) / (screen_x - slit2_x)
                py = axis_y - (deflection_px * spin_val) * (sub_prog ** 1.6)

            p_col = QColor("#10b981") if spin_val > 0 else QColor("#f43f5e")
            if self._is_classical_mode:
                p_col = QColor("#f59e0b")

            painter.setBrush(QBrush(p_col))
            painter.setPen(QPen(QColor(0, 0, 0), 0.8))
            painter.drawEllipse(QPointF(px, py), 3.0, 3.0)

    def _paint_detector_screen_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Right Panel: Front View of Glass Slide Detector & Intensity Distribution.
        """
        dx = 530.0
        dy = 70.0
        dw = 410.0
        dh = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(dx, dy, dw, dh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(dx + 12.0, dy + 18.0, "VẾT BẠC LẮNG ĐỌNG TRÊN BẢN KÍNH HỨNG:")

        # Glass Slide Target Region (Left sub-panel)
        sx = dx + 30.0
        sy = dy + 35.0
        sw = 120.0
        sh = dh - 60.0

        # Glass slide plate representation
        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(100, 116, 139), 1.6))
        painter.drawRoundedRect(QRectF(sx, sy, sw, sh), 6.0, 6.0)

        mid_y = sy + sh / 2.0
        deflection_px = min(55.0, max(12.0, self.beam_deflection_mm * 22.0))

        # Center reference crosshair
        painter.setPen(QPen(QColor(51, 65, 85, 140), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(sx + 10.0, mid_y), QPointF(sx + sw - 10.0, mid_y))
        painter.drawLine(QPointF(sx + sw / 2.0, sy + 10.0), QPointF(sx + sw / 2.0, sy + sh - 10.0))

        if self._is_classical_mode:
            # Classical continuous smudge: continuous silver gradient
            smudge_rect = QRectF(sx + sw / 2.0 - 14.0, mid_y - deflection_px, 28.0, deflection_px * 2.0)
            smudge_grad = QLinearGradient(smudge_rect.topLeft(), smudge_rect.bottomLeft())
            smudge_grad.setColorAt(0.0, QColor(245, 158, 11, 190))
            smudge_grad.setColorAt(0.5, QColor(245, 158, 11, 140))
            smudge_grad.setColorAt(1.0, QColor(245, 158, 11, 190))
            painter.setBrush(QBrush(smudge_grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(smudge_rect, 6.0, 6.0)
        else:
            # Discrete Quantum Splitting: 2 distinct silver spots
            spot_w = 34.0
            spot_h = 14.0

            # Upper spot: |↑z⟩
            up_alpha = int(240 * self.quantum_prob_spin_up)
            spot_up = QRectF(sx + sw / 2.0 - spot_w / 2.0, mid_y - deflection_px - spot_h / 2.0, spot_w, spot_h)
            painter.setBrush(QBrush(QColor(16, 185, 129, up_alpha)))
            painter.setPen(QPen(QColor(248, 250, 252, up_alpha), 1.0))
            painter.drawEllipse(spot_up)

            # Lower spot: |↓z⟩
            down_alpha = int(240 * self.quantum_prob_spin_down)
            spot_down = QRectF(sx + sw / 2.0 - spot_w / 2.0, mid_y + deflection_px - spot_h / 2.0, spot_w, spot_h)
            painter.setBrush(QBrush(QColor(244, 63, 94, down_alpha)))
            painter.setPen(QPen(QColor(248, 250, 252, down_alpha), 1.0))
            painter.drawEllipse(spot_down)

        # Right sub-panel: Vertical Intensity Profile I(z)
        px_x = dx + 180.0
        px_y = sy
        px_w = dw - 205.0
        px_h = sh

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(px_x, px_y, px_w, px_h))

        # Horizontal grid line at center z = 0
        painter.setPen(QPen(QColor(51, 65, 85, 140), 0.8, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(px_x, mid_y), QPointF(px_x + px_w, mid_y))

        # Plot I(z) along vertical axis: x is intensity, y is z position
        profile_path = QPainterPath()
        first = True
        num_pts = 80
        for i in range(num_pts + 1):
            y_cur = px_y + (i / num_pts) * px_h
            # Distance from center
            z_offset = (mid_y - y_cur)

            if self._is_classical_mode:
                # Classical broad flat-top distribution
                if abs(z_offset) <= deflection_px:
                    intens = 0.45 + 0.15 * math.cos((z_offset / deflection_px) * math.pi)
                else:
                    intens = 0.05
            else:
                # Quantum double Gaussian peaks centered at ±deflection_px
                sigma = 8.0
                g_up = self.quantum_prob_spin_up * math.exp(-((z_offset - deflection_px) ** 2) / (2.0 * sigma * sigma))
                g_down = self.quantum_prob_spin_down * math.exp(-((z_offset + deflection_px) ** 2) / (2.0 * sigma * sigma))
                intens = min(0.95, g_up + g_down + 0.02)

            x_cur = px_x + intens * (px_w - 15.0)
            if first:
                profile_path.moveTo(x_cur, y_cur)
                first = False
            else:
                profile_path.lineTo(x_cur, y_cur)

        painter.setPen(QPen(QColor(56, 189, 248), 1.8))
        painter.drawPath(profile_path)

        # Labels
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px_x + 8.0, px_y + 14.0, "Cường độ chùm I(z)")
        painter.drawText(px_x + px_w - 36.0, mid_y - 4.0, "z = 0")

    def _paint_sequential_or_geometry_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Left Panel: Sequential 3-Stage SG Experiment or Rotating Angle Geometry.
        """
        sx = 20.0
        sy = 360.0
        sw = 500.0
        sh = 220.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(sx, sy, sw, sh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))

        if self._is_sequential_mode:
            painter.drawText(sx + 12.0, sy + 18.0, "HỆ THỐNG 3 TẦNG SG NỐI TIẾP (SG_z → SG_x → SG_z):")
            self._paint_sequential_apparatus(painter, sx, sy, sw, sh)
        else:
            painter.drawText(sx + 12.0, sy + 18.0, "HÌNH HỌC KHÔNG GIAN & ĐỘ LỆCH GÓC THỦY TRỤC (θ):")
            self._paint_angle_geometry(painter, sx, sy, sw, sh)

    def _paint_sequential_apparatus(self, painter: QPainter, sx: float, sy: float, sw: float, sh: float) -> None:
        """Visualizes the 3-stage sequential quantum filtering paradox."""
        box_w = 70.0
        box_h = 45.0
        stages = [
            ("SG #1 (z)", sx + 40.0, sy + 45.0, "|↑z⟩ + |↓z⟩"),
            ("SG #2 (x)", sx + 190.0, sy + 45.0, "|↑x⟩ + |↓x⟩"),
            ("SG #3 (z)", sx + 340.0, sy + 45.0, "|↑z⟩ (50%) + |↓z⟩ (50%)"),
        ]

        for title, bx, by, desc in stages:
            # Stage box
            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.setPen(QPen(QColor(56, 189, 248), 1.2))
            painter.drawRoundedRect(QRectF(bx, by, box_w, box_h), 6.0, 6.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(248, 250, 252)))
            painter.drawText(QRectF(bx, by + 4.0, box_w, 18.0), Qt.AlignmentFlag.AlignCenter, title)

            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(QRectF(bx, by + 22.0, box_w, 18.0), Qt.AlignmentFlag.AlignCenter, desc)

        # Arrows and Blocking filters
        painter.setPen(QPen(QColor(16, 185, 129), 1.8))
        painter.drawLine(QPointF(sx + 110.0, sy + 67.0), QPointF(sx + 190.0, sy + 67.0))
        painter.drawLine(QPointF(sx + 260.0, sy + 67.0), QPointF(sx + 340.0, sy + 67.0))

        # Explanatory text box
        exp_rect = QRectF(sx + 12.0, sy + 110.0, sw - 24.0, 95.0)
        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(exp_rect, 6.0, 6.0)

        lines = [
            "• Tầng 1 (SG_z): Tách chùm thành |↑z⟩ và |↓z⟩. Chặn hoàn toàn chùm |↓z⟩.",
            "• Tầng 2 (SG_x): Chùm |↑z⟩ = (|↑x⟩ + |↓x⟩)/√2 chiếu lên trục X cho 50% |↑x⟩ & 50% |↓x⟩. Chặn |↓x⟩.",
            "• Tầng 3 (SG_z): Trạng thái |↑x⟩ = (|↑z⟩ + |↓z⟩)/√2 lại tách thành 50% |↑z⟩ và 50% |↓z⟩!",
            "• KẾT LUẬN: Đo trục X phá hủy thông tin trục Z do hai toán tử không giao hoán: [S_x, S_z] ≠ 0.",
        ]
        row_y = exp_rect.y() + 18.0
        painter.setFont(QFont("Segoe UI", 6))
        for l in lines:
            painter.setPen(QPen(QColor(245, 158, 11) if "KẾT LUẬN" in l else QColor(226, 232, 240)))
            painter.drawText(exp_rect.x() + 8.0, row_y, l)
            row_y += 20.0

    def _paint_angle_geometry(self, painter: QPainter, sx: float, sy: float, sw: float, sh: float) -> None:
        """Visualizes rotation of the magnetic gradient axis by angle theta."""
        cx = sx + 100.0
        cy = sy + sh / 2.0 + 8.0
        r = 65.0

        # Coordinate axes (Z vertical, X horizontal)
        painter.setPen(QPen(QColor(51, 65, 85, 140), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
        painter.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))

        # Rotating gradient axis (Angle theta from Z)
        th_rad = math.radians(self._magnet_angle_deg)
        ux = r * math.sin(th_rad)
        uy = -r * math.cos(th_rad)

        painter.setPen(QPen(QColor(245, 158, 11), 2.2))
        painter.drawLine(QPointF(cx - ux, cy - uy), QPointF(cx + ux, cy + uy))

        # Angle arc
        arc_r = 30.0
        painter.setPen(QPen(QColor(56, 189, 248), 1.2))
        painter.drawArc(QRectF(cx - arc_r, cy - arc_r, arc_r * 2.0, arc_r * 2.0), 90 * 16, -int(self._magnet_angle_deg * 16))

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(cx + 8.0, cy - 12.0, f"θ = {self._magnet_angle_deg:.0f}°")

        # Probability Formula Card on Right
        rx = sx + 200.0
        ry = sy + 40.0

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(rx, ry, "QUY TẮC CHIẾU XÁC SUẤT LƯỢNG TỬ (MALUS):")

        lines = [
            f"• Trạng thái lượng tử: |ψ⟩ = cos(θ/2)|↑z⟩ + sin(θ/2)|↓z⟩",
            f"• Xác suất Spin-Up: P(↑) = cos²(θ/2) = {self.quantum_prob_spin_up * 100:.1f}%",
            f"• Xác suất Spin-Down: P(↓) = sin²(θ/2) = {self.quantum_prob_spin_down * 100:.1f}%",
            f"• Tổng xác suất bảo toàn: P(↑) + P(↓) = 100.0% ≡ 1.0",
            f"• Tại θ = 0°: P(↑) = 100% | Tại θ = 90°: P(↑) = P(↓) = 50%",
            f"• Tại θ = 180°: Đảo ngược cực từ, P(↓) = 100%",
        ]

        row_y = ry + 22.0
        for l in lines:
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QPen(QColor(226, 232, 240) if "P(" in l else QColor(148, 163, 184)))
            painter.drawText(rx, row_y, l)
            row_y += 20.0

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Right Panel: Rigorous Physics Constants & Deflection Metrics.
        """
        mx = 530.0
        my = 360.0
        mw = 410.0
        mh = 220.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(mx, my, mw, mh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 18.0, "THÔNG SỐ VẬT LÝ & LỰC TỪ TRƯỜNG GRADIENT:")

        metrics = [
            ("Mômen từ Bohr (Bohr Magneton μ_B):", f"μ_B = {self.BOHR_MAGNETON_J_PER_T:.4e} J/T"),
            ("Gradient từ trường (Field Gradient dB/dz):", f"dB/dz = {self._gradient_T_per_m:.0f} T/m"),
            ("Lực từ tác dụng lên hạt (Magnetic Force F_z):", f"F_z = ±μ_B·(dB/dz) = ±{self.magnetic_deflection_force_N:.2e} N"),
            ("Vận tốc xác suất lớn nhất (v_p):", f"v_p = √(2k_B T/m) = {self.most_probable_velocity_m_per_s:.1f} m/s"),
            ("Độ lệch chùm tia trên kính (Deflection Δz):", f"Δz = ±{self.beam_deflection_mm:.2f} mm"),
            ("Hình chiếu Spin z (Spin Projection S_z):", "S_z = ±ħ/2 (m_s = ±1/2)"),
            ("Bản chất Hiện tượng (Physical Nature):", "LƯỢNG TỬ HÓA KHÔNG GIAN (Space Quantization)"),
        ]

        row_y = my + 40.0
        line_h = 24.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 12.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            val_col = QColor(16, 185, 129) if "LƯỢNG TỬ HÓA" in val or "Δz" in name else QColor(226, 232, 240)
            painter.setPen(QPen(val_col))
            painter.drawText(mx + 215.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture control buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_th_down", "θ -", QColor(51, 65, 85)),
            ("btn_th_up", "θ +", QColor(51, 65, 85)),
            ("btn_grad_down", "dB/dz -", QColor(51, 65, 85)),
            ("btn_grad_up", "dB/dz +", QColor(51, 65, 85)),
            ("btn_temp_down", "T -", QColor(51, 65, 85)),
            ("btn_temp_up", "T +", QColor(51, 65, 85)),
            ("btn_classical", "Cổ điển" if not self._is_classical_mode else "Lượng tử", QColor(245, 158, 11)),
            ("btn_seq", "3 Tầng SG" if not self._is_sequential_mode else "1 Tầng SG", QColor(168, 85, 247)),
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

    # -------------------------------------------------------------------------
    # Touch & Mouse Event Handling
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
            keys = list(STERN_GERLACH_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_th_down":
            self.set_magnet_angle(self._magnet_angle_deg - 15.0)
        elif key == "btn_th_up":
            self.set_magnet_angle(self._magnet_angle_deg + 15.0)
        elif key == "btn_grad_down":
            self.adjust_gradient(-200.0)
        elif key == "btn_grad_up":
            self.adjust_gradient(+200.0)
        elif key == "btn_temp_down":
            self.adjust_temperature(-100.0)
        elif key == "btn_temp_up":
            self.adjust_temperature(+100.0)
        elif key == "btn_classical":
            self.toggle_classical_comparison()
        elif key == "btn_seq":
            self.toggle_sequential_mode()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        data = {
            "preset_code": self._active_preset_code,
            "gradient_T_per_m": self._gradient_T_per_m,
            "magnet_angle_deg": self._magnet_angle_deg,
            "oven_temp_K": self._oven_temp_K,
            "deflection_mm": self.beam_deflection_mm,
            "p_up": self.quantum_prob_spin_up,
            "p_down": self.quantum_prob_spin_down,
            "is_classical": self._is_classical_mode,
        }
        self.stamp_requested.emit(data)
