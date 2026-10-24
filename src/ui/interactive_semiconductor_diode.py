"""
AirTeach AI — Interactive Semiconductor P-N Junction, Shockley Diode Equation & Band Gap Simulator
Authentic solid-state physics, Shockley ideal diode equation, space-charge depletion region width,
energy band diagrams (Ec, Ev, EF), carrier drift/diffusion, and touchless air gesture integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    QStyleOptionGraphicsItem,
    QWidget,
)


@dataclass(frozen=True)
class SemiconductorPreset:
    code: str
    name_vi: str
    material: str             # "Si", "Ge", "GaAs", "Zener"
    bandgap_ev: float         # Eg in eV
    v_bi: float               # Built-in potential in Volts
    is_sat_a: float           # Reverse saturation current Is in Amperes
    breakdown_v: float        # Reverse breakdown voltage (negative)
    ideality_factor: float    # n factor in Shockley equation (1.0 .. 2.0)
    description: str


PRESETS: dict[str, SemiconductorPreset] = {
    "SILICON_PN_RECTIFIER": SemiconductorPreset(
        code="SILICON_PN_RECTIFIER",
        name_vi="Điốt chỉnh lưu Silic (Si Rectifier)",
        material="Si",
        bandgap_ev=1.12,
        v_bi=0.70,
        is_sat_a=1e-12,
        breakdown_v=-50.0,
        ideality_factor=1.05,
        description="Điốt Silic thông dụng (1N4007); điện thế tiếp xúc V_bi = 0.7V, dòng rò ngược Is cực nhỏ ~1 pA.",
    ),
    "GERMANIUM_DIODE": SemiconductorPreset(
        code="GERMANIUM_DIODE",
        name_vi="Điốt Gecmani tách sóng (Ge Diode)",
        material="Ge",
        bandgap_ev=0.66,
        v_bi=0.30,
        is_sat_a=1e-6,
        breakdown_v=-25.0,
        ideality_factor=1.02,
        description="Điốt Gecmani có rào thế thấp V_bi = 0.3V dẫn điện sớm, dùng trong mạch tách sóng radio tần số cao.",
    ),
    "GALLIUM_ARSENIDE_LED": SemiconductorPreset(
        code="GALLIUM_ARSENIDE_LED",
        name_vi="Điốt phát quang LED GaAs (Optoelectronics)",
        material="GaAs",
        bandgap_ev=1.42,
        v_bi=1.20,
        is_sat_a=1e-15,
        breakdown_v=-15.0,
        ideality_factor=1.80,
        description="Bán dẫn vùng cấm thẳng GaAs (Eg = 1.42 eV); tái hợp điện tử-lỗ trống phát xạ photon ánh sáng đỏ/hồng ngoại.",
    ),
    "ZENER_BREAKDOWN_REGULATOR": SemiconductorPreset(
        code="ZENER_BREAKDOWN_REGULATOR",
        name_vi="Điốt ổn áp Zener (Tunneling Breakdown)",
        material="Si-Zener",
        bandgap_ev=1.12,
        v_bi=0.85,
        is_sat_a=5e-11,
        breakdown_v=-5.1,
        ideality_factor=1.10,
        description="P-N pha tạp cực cao tạo hiệu ứng xuyên hầm lượng tử Zener giữ điện áp ngược ổn định ở -5.1V.",
    ),
    "REVERSE_SATURATION_LEAK": SemiconductorPreset(
        code="REVERSE_SATURATION_LEAK",
        name_vi="Dòng rò nhiệt ngược (Thermal Leakage)",
        material="Si-Hot",
        bandgap_ev=1.05,
        v_bi=0.60,
        is_sat_a=5e-9,
        breakdown_v=-40.0,
        ideality_factor=1.15,
        description="Nhiệt độ cao làm bứt phá liên kết hóa trị sinh cặp electron-lỗ trống nhiệt làm tăng vọt dòng rò ngược Is.",
    ),
}


class InteractiveSemiconductorDiodeItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Semiconductor P-N Junction, Shockley Diode equation,
    depletion region width, I-V characteristic curve, and energy band diagrams.
    """

    WIDTH = 680.0
    HEIGHT = 410.0

    stamp_requested = Signal()
    voltage_changed = Signal(float)  # bias voltage (V)

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (150.0, 65.0),
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

        # Simulation State
        self._active_preset_code: str = "SILICON_PN_RECTIFIER"
        self._preset: SemiconductorPreset = PRESETS[self._active_preset_code]

        # Physical Parameters
        self._bias_voltage: float = 0.65     # Volts (Forward bias near threshold)
        self._temperature_k: float = 300.0   # Kelvin (Room temperature 27°C)

        # Carrier animation
        self._anim_phase: float = 0.0
        self._is_animating: bool = True
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)  # 30 FPS
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Sets up layout rects for toolbar buttons, junction view, and I-V curve."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0

        # 5 Preset buttons
        preset_names = [
            ("SILICON_PN_RECTIFIER", "Si 1N4007"),
            ("GERMANIUM_DIODE", "Ge Tách sóng"),
            ("GALLIUM_ARSENIDE_LED", "LED GaAs"),
            ("ZENER_BREAKDOWN_REGULATOR", "Zener 5.1V"),
            ("REVERSE_SATURATION_LEAK", "Dòng rò nhiệt"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 74.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 4.0

        # Controls
        r_v_down = QRectF(cur_x + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["v_down"] = r_v_down

        r_v_up = QRectF(r_v_down.right() + 3.0, btn_y, 28.0, btn_h)
        self._btn_rects["v_up"] = r_v_up

        r_t_down = QRectF(r_v_up.right() + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["temp_down"] = r_t_down

        r_t_up = QRectF(r_t_down.right() + 3.0, btn_y, 28.0, btn_h)
        self._btn_rects["temp_up"] = r_t_up

        r_play = QRectF(r_t_up.right() + 4.0, btn_y, 36.0, btn_h)
        self._btn_rects["toggle_anim"] = r_play

        r_stamp = QRectF(r_play.right() + 4.0, btn_y, 46.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        # Viewport areas
        self._junction_rect = QRectF(16.0, 78.0, 260.0, 205.0)
        self._iv_curve_rect = QRectF(286.0, 78.0, 195.0, 205.0)
        self._diag_rect = QRectF(491.0, 78.0, 175.0, 205.0)

    def cleanup(self) -> None:
        """Stops animation timer to prevent memory leaks."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    # -------------------------------------------------------------------------
    # Solid-State Physical Formulas (Authentic Physics)
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined semiconductor preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        if preset_code == "REVERSE_SATURATION_LEAK":
            self._temperature_k = 380.0
            self._bias_voltage = -2.0
        elif preset_code == "ZENER_BREAKDOWN_REGULATOR":
            self._bias_voltage = -5.0
        elif preset_code == "GERMANIUM_DIODE":
            self._bias_voltage = 0.28
        else:
            self._bias_voltage = 0.65
            self._temperature_k = 300.0
        self.update()

    def set_bias_voltage(self, v: float) -> None:
        """Sets external bias voltage clamped to [-10.0V, +1.5V]."""
        self._bias_voltage = max(-10.0, min(1.5, float(v)))
        self.voltage_changed.emit(self._bias_voltage)
        self.update()

    def set_temperature_k(self, t_kelvin: float) -> None:
        """Sets junction temperature clamped to [200K, 450K]."""
        self._temperature_k = max(200.0, min(450.0, float(t_kelvin)))
        self.update()

    @property
    def thermal_voltage_vt(self) -> float:
        """Calculates thermal voltage V_T = k_B * T / q (in Volts)."""
        k_b = 1.380649e-23  # J/K
        q = 1.60217663e-19  # Coulombs
        return (k_b * self._temperature_k) / q

    @property
    def current_amperes(self) -> float:
        """
        Calculates live diode current I (Amperes) using Shockley equation
        with reverse breakdown modeling:
        I = I_s * (exp(V / (n * V_T)) - 1)
        """
        v = self._bias_voltage
        v_t = self.thermal_voltage_vt
        n = self._preset.ideality_factor
        i_s = self._preset.is_sat_a * ((self._temperature_k / 300.0) ** 3) * math.exp(-self._preset.bandgap_ev * 1.602e-19 / (2 * 1.38e-23) * (1 / self._temperature_k - 1 / 300.0))

        # Reverse breakdown
        v_br = self._preset.breakdown_v
        if v < v_br:
            # Steep reverse breakdown current
            return -0.05 * (v_br - v) - 1e-4

        # Shockley forward/reverse
        exp_arg = max(-40.0, min(35.0, v / (n * v_t)))
        return float(i_s * (math.exp(exp_arg) - 1.0))

    @property
    def depletion_width_norm(self) -> float:
        """
        Normalized space-charge depletion region width W in [0.05 .. 0.95].
        W proportional to sqrt(V_bi - V).
        """
        v_diff = max(0.02, self._preset.v_bi - self._bias_voltage)
        w = math.sqrt(v_diff / self._preset.v_bi)
        return max(0.08, min(0.92, w * 0.40))

    def toggle_animation(self) -> bool:
        """Toggles carrier animation."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def _on_anim_tick(self) -> None:
        """Updates carrier flow animation phase."""
        if not self._is_animating:
            return
        # Drift speed proportional to current magnitude
        speed = 0.05 + min(0.3, abs(self.current_amperes) * 10.0)
        self._anim_phase += speed
        self.update()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def bias_voltage(self) -> float:
        return self._bias_voltage

    @property
    def temperature_k(self) -> float:
        return self._temperature_k

    @property
    def active_preset(self) -> SemiconductorPreset:
        return self._preset

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active Semiconductor Diode diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import PhysicsTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = PhysicsTools.create_semiconductor_diode_diagram(
            center=center,
            preset_code=self._active_preset_code,
            bias_v=self._bias_voltage,
            current_a=self.current_amperes,
            temp_k=self._temperature_k,
        )
        for itm in items:
            scene.addItem(itm)
        self.stamp_requested.emit()

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Voltage steppers
        r_vd = self._btn_rects.get("v_down")
        if r_vd and r_vd.contains(local_pos):
            self.set_bias_voltage(self._bias_voltage - 0.05)
            return True

        r_vu = self._btn_rects.get("v_up")
        if r_vu and r_vu.contains(local_pos):
            self.set_bias_voltage(self._bias_voltage + 0.05)
            return True

        # 3. Temp steppers
        r_td = self._btn_rects.get("temp_down")
        if r_td and r_td.contains(local_pos):
            self.set_temperature_k(self._temperature_k - 10.0)
            return True

        r_tu = self._btn_rects.get("temp_up")
        if r_tu and r_tu.contains(local_pos):
            self.set_temperature_k(self._temperature_k + 10.0)
            return True

        # 4. Toggle animation
        r_play = self._btn_rects.get("toggle_anim")
        if r_play and r_play.contains(local_pos):
            self.toggle_animation()
            return True

        # 5. Direct I-V curve voltage scrub
        if self._iv_curve_rect.contains(local_pos):
            frac = (local_pos.x() - self._iv_curve_rect.left()) / self._iv_curve_rect.width()
            # Map [0, 1] to [-2.0V, +1.0V]
            new_v = -2.0 + frac * 3.0
            self.set_bias_voltage(new_v)
            return True

        # 6. Stamp
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
            return True

        return True

    # -------------------------------------------------------------------------
    # Rendering & Painting
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

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header Banner
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 36.0), 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 26.0, self.WIDTH, 10.0))

        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(14.0, 0.0, 480.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "⚡ BÁN DẪN LỚP TIẾP XÚC P-N & PHƯƠNG TRÌNH SHOCKLEY (SOLID STATE)",
        )

        # Preset Badge
        p_badge = QRectF(self.WIDTH - 164.0, 7.0, 150.0, 22.0)
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.setPen(QPen(QColor("#818cf8"), 1.0))
        painter.drawRoundedRect(p_badge, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#c7d2fe"))
        painter.drawText(p_badge, Qt.AlignmentFlag.AlignCenter, self._preset.name_vi[:24])

        # 3. Preset Buttons & Controls
        preset_items = [
            ("SILICON_PN_RECTIFIER", "Si 1N4007"),
            ("GERMANIUM_DIODE", "Ge Tách sóng"),
            ("GALLIUM_ARSENIDE_LED", "LED GaAs"),
            ("ZENER_BREAKDOWN_REGULATOR", "Zener 5.1V"),
            ("REVERSE_SATURATION_LEAK", "Dòng rò nhiệt"),
        ]
        for p_code, p_lbl in preset_items:
            r = self._btn_rects[f"preset_{p_code}"]
            is_active = (p_code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#3b82f6") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#60a5fa") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, p_lbl)

        # Voltage steppers
        r_vd = self._btn_rects["v_down"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(r_vd, 3.0, 3.0)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_vd, Qt.AlignmentFlag.AlignCenter, "V-")

        r_vu = self._btn_rects["v_up"]
        painter.drawRoundedRect(r_vu, 3.0, 3.0)
        painter.setPen(QColor("#ef4444"))
        painter.drawText(r_vu, Qt.AlignmentFlag.AlignCenter, "V+")

        # Temp steppers
        r_td = self._btn_rects["temp_down"]
        painter.drawRoundedRect(r_td, 3.0, 3.0)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_td, Qt.AlignmentFlag.AlignCenter, "T-")

        r_tu = self._btn_rects["temp_up"]
        painter.drawRoundedRect(r_tu, 3.0, 3.0)
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(r_tu, Qt.AlignmentFlag.AlignCenter, "T+")

        # Animation toggle
        r_play = self._btn_rects["toggle_anim"]
        painter.setBrush(QBrush(QColor("#065f46") if self._is_animating else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#10b981") if self._is_animating else QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_play, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff") if self._is_animating else QColor("#94a3b8"))
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, "▶" if not self._is_animating else "⏸")

        # Stamp Button
        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#831843")))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # ---------------------------------------------------------------------
        # 4. Left Panel: Physical P-N Junction & Depletion Layer
        # ---------------------------------------------------------------------
        jx, jy, jw, jh = self._junction_rect.x(), self._junction_rect.y(), self._junction_rect.width(), self._junction_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._junction_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(jx + 8.0, jy + 16.0), "LỚP TIẾP XÚC P-N & VÙNG NGHÈO (DEPLETION)")

        # P-Type block (left) and N-Type block (right)
        box_y = jy + 26.0
        box_h = 70.0
        cx = jx + jw / 2.0

        # P-region (Rose/Red)
        painter.setBrush(QBrush(QColor("#881337")))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRect(QRectF(jx + 12.0, box_y, (jw - 24.0) / 2.0, box_h))

        # N-region (Blue)
        painter.setBrush(QBrush(QColor("#1e3a8a")))
        painter.setPen(QPen(QColor("#3b82f6"), 1.0))
        painter.drawRect(QRectF(cx, box_y, (jw - 24.0) / 2.0, box_h))

        # Depletion Layer (Space Charge Region W)
        dw_px = (jw - 24.0) * self.depletion_width_norm
        dep_rect = QRectF(cx - dw_px / 2.0, box_y, dw_px, box_h)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))  # Slate dark zone
        painter.setPen(QPen(QColor("#e2e8f0"), 1.2, Qt.PenStyle.DashLine))
        painter.drawRect(dep_rect)

        # Region Titles
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#fda4af"))
        painter.drawText(QRectF(jx + 16.0, box_y + 6.0, 60.0, 16.0), Qt.AlignmentFlag.AlignLeft, "Loại P (Lỗ trống h+)")

        painter.setPen(QColor("#93c5fd"))
        painter.drawText(QRectF(jx + jw - 80.0, box_y + 6.0, 64.0, 16.0), Qt.AlignmentFlag.AlignRight, "Loại N (Electron e-)")

        # Built-in electric field arrow E_bi in depletion zone
        painter.setPen(QPen(QColor("#fbbf24"), 1.5))
        painter.drawLine(QPointF(cx + dw_px * 0.35, box_y + box_h / 2.0), QPointF(cx - dw_px * 0.35, box_y + box_h / 2.0))
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QPointF(cx - 10.0, box_y + box_h / 2.0 - 4.0), "E_bi ➔")

        # Energy Band Diagram (Ec, Ev, EF) below junction
        band_y = box_y + box_h + 14.0
        band_h = 80.0
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRect(QRectF(jx + 12.0, band_y, jw - 24.0, band_h))

        # Band bending: Conduction band Ec and Valence band Ev
        ec_p = band_y + 18.0
        ec_n = band_y + 18.0 + (self._preset.v_bi - self._bias_voltage) * 25.0
        ec_n = max(band_y + 10.0, min(band_y + band_h - 20.0, ec_n))

        # Draw Ec path
        ec_path = QPainterPath()
        ec_path.moveTo(jx + 12.0, ec_p)
        ec_path.lineTo(cx - dw_px / 2.0, ec_p)
        ec_path.cubicTo(cx, ec_p, cx, ec_n, cx + dw_px / 2.0, ec_n)
        ec_path.lineTo(jx + jw - 12.0, ec_n)

        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(ec_path)

        # Draw Ev path (parallel, separated by bandgap)
        gap_px = self._preset.bandgap_ev * 18.0
        ev_p = ec_p + gap_px
        ev_n = ec_n + gap_px

        ev_path = QPainterPath()
        ev_path.moveTo(jx + 12.0, ev_p)
        ev_path.lineTo(cx - dw_px / 2.0, ev_p)
        ev_path.cubicTo(cx, ev_p, cx, ev_n, cx + dw_px / 2.0, ev_n)
        ev_path.lineTo(jx + jw - 12.0, ev_n)

        painter.setPen(QPen(QColor("#f43f5e"), 1.8))
        painter.drawPath(ev_path)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(jx + 16.0, ec_p - 4.0), "E_c (Vùng dẫn)")

        painter.setPen(QColor("#f43f5e"))
        painter.drawText(QPointF(jx + 16.0, ev_p + 12.0), "E_v (Vùng hóa trị)")

        # ---------------------------------------------------------------------
        # 5. Center Panel: Live I-V Characteristic Curve
        # ---------------------------------------------------------------------
        ix, iy, iw, ih = self._iv_curve_rect.x(), self._iv_curve_rect.y(), self._iv_curve_rect.width(), self._iv_curve_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._iv_curve_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ix + 8.0, iy + 16.0), "ĐẶC TUYẾN V-A (I-V SHOCKLEY)")

        # I-V Axes
        origin_iv = QPointF(ix + iw * 0.45, iy + ih * 0.70)
        painter.setPen(QPen(QColor("#475569"), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(ix + 6.0, origin_iv.y()), QPointF(ix + iw - 6.0, origin_iv.y()))  # V-axis
        painter.drawLine(QPointF(origin_iv.x(), iy + 22.0), QPointF(origin_iv.x(), iy + ih - 10.0))  # I-axis

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(ix + iw - 18.0, origin_iv.y() - 4.0), "V")
        painter.drawText(QPointF(origin_iv.x() + 4.0, iy + 30.0), "I")

        # Plot theoretical curve
        v_pts = [-2.0 + i * 0.05 for i in range(65)]
        iv_path = QPainterPath()
        n = self._preset.ideality_factor
        vt = self.thermal_voltage_vt
        isat = self._preset.is_sat_a

        for idx, v_val in enumerate(v_pts):
            i_val = isat * (math.exp(max(-20.0, min(30.0, v_val / (n * vt)))) - 1.0)
            px = origin_iv.x() + (v_val / 1.0) * 80.0
            py = origin_iv.y() - min(80.0, i_val * 400.0)
            if idx == 0:
                iv_path.moveTo(px, py)
            else:
                iv_path.lineTo(px, py)

        painter.setPen(QPen(QColor("#10b981"), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(iv_path)

        # Plot current operating point Q(V, I)
        curr_i = self.current_amperes
        q_x = origin_iv.x() + (self._bias_voltage / 1.0) * 80.0
        q_y = origin_iv.y() - min(80.0, max(-30.0, curr_i * 400.0))

        painter.setBrush(QBrush(QColor("#f43f5e")))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(QPointF(q_x, q_y), 4.5, 4.5)

        # ---------------------------------------------------------------------
        # 6. Right Panel: Analytical Diagnostic Ledger
        # ---------------------------------------------------------------------
        dx, dy, dw, dh = self._diag_rect.x(), self._diag_rect.y(), self._diag_rect.width(), self._diag_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._diag_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(dx + 8.0, dy + 16.0), "THÔNG SỐ VẬT LÝ BÁN DẪN")

        lines = [
            (f"• Điện áp phân cực V: {self._bias_voltage:+.2f} V", QColor("#38bdf8")),
            (f"• Dòng điện I: {curr_i:.4e} A", QColor("#10b981") if curr_i > 1e-4 else QColor("#ef4444")),
            (f"• Thế tiếp xúc V_bi: {self._preset.v_bi:.2f} V", QColor("#fbbf24")),
            (f"• Vùng cấm E_g: {self._preset.bandgap_ev:.2f} eV", QColor("#a855f7")),
            (f"• Điện thế nhiệt V_T: {vt*1000:.1f} mV", QColor("#38bdf8")),
            (f"• Nhiệt độ T: {self._temperature_k:.0f} K ({self._temperature_k-273.15:.0f}°C)", QColor("#f43f5e")),
            (f"• Bề rộng vùng nghèo W: {self.depletion_width_norm*100:.0f}%", QColor("#e2e8f0")),
            (f"• Trạng thái: {'Phân cực thuận (ON)' if self._bias_voltage > 0.4 else 'Phân cực ngược (OFF)'}", QColor("#10b981") if self._bias_voltage > 0.4 else QColor("#f43f5e")),
        ]

        ly = dy + 32.0
        for l_txt, l_col in lines:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(l_col)
            painter.drawText(QPointF(dx + 8.0, ly), l_txt)
            ly += 17.0

        # Operating state banner
        state_r = QRectF(dx + 8.0, dy + dh - 24.0, dw - 16.0, 16.0)
        is_fwd = (self._bias_voltage >= self._preset.v_bi * 0.9)
        painter.setBrush(QBrush(QColor("#065f46") if is_fwd else QColor("#1e1b4b")))
        painter.setPen(QPen(QColor("#10b981") if is_fwd else QColor("#6366f1"), 1.0))
        painter.drawRoundedRect(state_r, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(state_r, Qt.AlignmentFlag.AlignCenter, "DẪN THUẬN MẠNH" if is_fwd else "KHÓA NGƯỢC")

        # ---------------------------------------------------------------------
        # 7. Bottom Panel: Shockley & Solid-State Physics Principles
        # ---------------------------------------------------------------------
        bot_y = self._junction_rect.bottom() + 10.0
        bot_h = self.HEIGHT - bot_y - 10.0
        bot_rect = QRectF(14.0, bot_y, self.WIDTH - 28.0, bot_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(bot_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(22.0, bot_y + 16.0), "LÝ THUYẾT VẬT LÝ BÁN DẪN & PHƯƠNG TRÌNH ĐIỐT SHOCKLEY (1949):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        b_desc = (
            f"• Phương trình điốt lý tưởng Shockley: I = I_s · [exp(qV / n·k_B·T) - 1]. Điện thế nhiệt V_T = k_B·T/q ≈ 25.86 mV ở 300K.\n"
            f"• Bề rộng vùng nghèo W ∝ √(V_bi - V). Phân cực thuận làm hạ rào thế, hạt tải đa số khuếch tán ồ ạt qua mặt tiếp xúc.\n"
            f"• Đặc tính mẫu: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, bot_y + 22.0, bot_rect.width() - 20.0, 58.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            b_desc,
        )
