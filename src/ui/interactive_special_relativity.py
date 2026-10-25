"""
AirTeach AI - Interactive Special Relativity: Minkowski Spacetime Diagram & Lorentz Engine
==========================================================================================
Authentic relativistic physics simulation according to Einstein's Special Relativity (1905):
- Spacetime Coordinates (ct, x) in Lab Frame S and Moving Frame S'
- Lorentz Boost Transformation:
    ct' = γ (ct - β x)
    x'  = γ (x - β ct)
    where β = v / c ∈ (-0.99, 0.99), γ = 1 / sqrt(1 - β^2) >= 1.0
- Invariant Spacetime Interval:
    Δs^2 = c^2 Δt^2 - Δx^2 = c^2 (Δt')^2 - (Δx')^2  (Exact invariance)
    - Timelike (Δs^2 > 0): Causally connectable, Proper Time Δτ = sqrt(Δs^2) / c
    - Spacelike (Δs^2 < 0): Causally disconnected, Proper Length Δσ = sqrt(-Δs^2)
    - Lightlike / Null (Δs^2 = 0): Connected by light speed photon worldline
- Calibration Hyperbolas:
    c^2 t^2 - x^2 = ±1 (Unit time and unit distance calibration curves across frames)
- Relativity of Simultaneity:
    The x'-axis of simultaneity tilts by θ = arctan(β) with respect to the x-axis,
    demonstrating why events simultaneous in Frame S are not simultaneous in Frame S'.
- Time Dilation & Length Contraction:
    Δt = γ Δt_0 (Moving clocks tick slower)
    L  = L_0 / γ (Moving rods contract in the direction of motion)
- 5 Curriculum Presets:
    1. LORENTZ_GRID: Variable β boost shearing with calibrated spacetime coordinate nets.
    2. TWIN_PARADOX: Earth twin vs Rocket twin worldline path and asymmetric proper time.
    3. SIMULTANEITY_TRAIN: Einstein's train experiment with front/rear simultaneous lightning strikes.
    4. MUON_DECAY: Atmospheric cosmic muon survival via time dilation and length contraction.
    5. LIGHT_CONE_CAUSALITY: Future light cone, past light cone, and spacelike elsewhere zones.
- Touchless webcam air gestures (air-tap, pinch-slider, event dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
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

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Speed of light in natural spacetime units (c = 1.0 light-second / second)
C_LIGHT_NATURAL = 1.0


@dataclass(frozen=True)
class RelativityPreset:
    code: str
    name_vi: str
    beta: float
    e1_pos: tuple[float, float]  # (x, ct) in natural units
    e2_pos: tuple[float, float]
    description: str
    scenario_type: str


PRESETS: dict[str, RelativityPreset] = {
    "LORENTZ_GRID": RelativityPreset(
        code="LORENTZ_GRID",
        name_vi="1. Lưới Không-Thời gian & Biến đổi Lorentz",
        beta=0.60,
        e1_pos=(-2.0, 1.0),
        e2_pos=(2.0, 3.5),
        description="Quan sát hệ trục (ct', x') nghiêng góc θ = arctan(β) về phía nón ánh sáng. γ = 1.25.",
        scenario_type="grid",
    ),
    "TWIN_PARADOX": RelativityPreset(
        code="TWIN_PARADOX",
        name_vi="2. Nghịch lý Song sinh (Twin Paradox)",
        beta=0.80,
        e1_pos=(0.0, 0.0),
        e2_pos=(0.0, 5.0),
        description="Người anh ở Trái Đất (đường thế giới thẳng đứng) già đi 5 năm, người em bay quay về chỉ già đi 3 năm (γ = 1.67).",
        scenario_type="twins",
    ),
    "SIMULTANEITY_TRAIN": RelativityPreset(
        code="SIMULTANEITY_TRAIN",
        name_vi="3. Tính Tương đối của Sự đồng thời",
        beta=0.50,
        e1_pos=(-3.0, 2.0),
        e2_pos=(3.0, 2.0),
        description="Hai sự kiện xảy ra đồng thời trong hệ S (cùng ct = 2.0) nhưng xảy ra tại hai thời điểm khác nhau trong hệ S'.",
        scenario_type="simultaneity",
    ),
    "MUON_DECAY": RelativityPreset(
        code="MUON_DECAY",
        name_vi="4. Phân rã Hạt Muon trong Khí quyển",
        beta=0.95,
        e1_pos=(0.0, 0.0),
        e2_pos=(3.5, 3.68),
        description="Với β = 0.95 (γ = 3.20), thời gian sống của muon dãn dài trong hệ Trái Đất, giúp hạt đến được mặt biển.",
        scenario_type="muon",
    ),
    "LIGHT_CONE_CAUSALITY": RelativityPreset(
        code="LIGHT_CONE_CAUSALITY",
        name_vi="5. Nón Ánh sáng & Cấu trúc Nhân quả",
        beta=0.0,
        e1_pos=(0.0, 0.0),
        e2_pos=(1.5, 2.5),
        description="Phân vùng Quá khứ, Tương lai (dạng thời gian Δs² > 0) và Miền ngoài nón (dạng không gian Δs² < 0 không thể truyền tin).",
        scenario_type="lightcone",
    ),
}


class InteractiveSpecialRelativityItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Minkowski Spacetime Diagrams and Lorentz Boosts.
    """

    WIDTH = 760.0
    HEIGHT = 500.0

    beta_changed = Signal(float, float)  # (beta, gamma)
    preset_changed = Signal(str)
    interval_updated = Signal(float, str)  # (s2, interval_type)

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (120.0, 50.0),
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
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Simulation Parameters
        self._beta: float = 0.60  # Relative velocity v / c
        self._gamma: float = 1.25  # 1 / sqrt(1 - beta^2)
        self._active_preset_code: str = "LORENTZ_GRID"
        self._preset: RelativityPreset = PRESETS[self._active_preset_code]

        # Spacetime Events in S Frame: (x, ct) in natural units
        self._e1: tuple[float, float] = self._preset.e1_pos
        self._e2: tuple[float, float] = self._preset.e2_pos

        # Display Toggles
        self._show_moving_frame: bool = True
        self._show_hyperbolas: bool = True
        self._show_light_cone: bool = True
        self._show_interval_vector: bool = True

        # Interactive dragging target: None, "slider", "e1", "e2"
        self._drag_target: Optional[str] = None

        # Layout Rectangles
        self._btn_rects: dict[str, QRectF] = {}
        self._slider_rect = QRectF(24.0, 442.0, 360.0, 16.0)
        self._slider_handle_rect = QRectF(0.0, 0.0, 16.0, 22.0)

        # Coordinate transform geometry (Diagram Origin in local coordinates)
        self._diag_origin = QPointF(204.0, 260.0)
        self._scale_px: float = 38.0  # Pixels per natural unit (light-second)

        self._update_lorentz_factors()
        self._setup_layout()

    # ── Relativistic Mathematics ───────────────────────────────────────────────

    def _update_lorentz_factors(self) -> None:
        """Compute gamma and re-evaluate spacetime intervals."""
        self._beta = max(-0.99, min(0.99, float(self._beta)))
        beta_sq = self._beta * self._beta
        self._gamma = float(1.0 / math.sqrt(max(1e-6, 1.0 - beta_sq)))

    def lorentz_transform(self, x: float, ct: float) -> tuple[float, float]:
        """Apply Lorentz boost from Frame S to Frame S': (x', ct')."""
        gamma = self._gamma
        beta = self._beta
        ct_prime = gamma * (ct - beta * x)
        x_prime = gamma * (x - beta * ct)
        return float(x_prime), float(ct_prime)

    def inverse_lorentz_transform(self, x_prime: float, ct_prime: float) -> tuple[float, float]:
        """Apply inverse Lorentz boost from Frame S' to Frame S: (x, ct)."""
        gamma = self._gamma
        beta = self._beta
        ct = gamma * (ct_prime + beta * x_prime)
        x = gamma * (x_prime + beta * ct_prime)
        return float(x), float(ct)

    def compute_spacetime_interval(self) -> tuple[float, str, float]:
        """
        Compute invariant spacetime interval Δs^2 = c^2 Δt^2 - Δx^2.
        Returns:
            (delta_s2, interval_type, proper_measure)
        """
        dx = self._e2[0] - self._e1[0]
        dct = self._e2[1] - self._e1[1]

        # Invariant interval in Frame S
        s2 = (dct * dct) - (dx * dx)

        # Verify invariance against Frame S'
        x1_p, ct1_p = self.lorentz_transform(self._e1[0], self._e1[1])
        x2_p, ct2_p = self.lorentz_transform(self._e2[0], self._e2[1])
        dx_p = x2_p - x1_p
        dct_p = ct2_p - ct1_p
        s2_prime = (dct_p * dct_p) - (dx_p * dx_p)

        # Invariance check
        if abs(s2 - s2_prime) > 1e-4:
            logger.debug("Lorentz invariance numerical discrepancy: %f vs %f", s2, s2_prime)

        if s2 > 1e-5:
            inv_type = "Timelike (Dạng thời gian)"
            proper = math.sqrt(s2)  # Proper time Δτ in light-seconds
        elif s2 < -1e-5:
            inv_type = "Spacelike (Dạng không gian)"
            proper = math.sqrt(-s2)  # Proper length Δσ in light-seconds
        else:
            inv_type = "Lightlike / Null (Dạng ánh sáng)"
            proper = 0.0

        return float(s2), inv_type, float(proper)

    # ── UI Layout ─────────────────────────────────────────────────────────────

    def _setup_layout(self) -> None:
        """Setup hitboxes for toolbar buttons, mode switchers, and controls."""
        self._btn_rects.clear()

        # Row 1: Preset Buttons
        px = 14.0
        py = 40.0
        btn_h = 22.0
        for code, pr in PRESETS.items():
            btn_w = 140.0
            self._btn_rects[f"preset_{code}"] = QRectF(px, py, btn_w, btn_h)
            px += btn_w + 6.0

        # Row 2: Toggles & Quick Beta Presets
        hx = 14.0
        hy = 68.0
        btn_w_sm = 80.0
        self._btn_rects["toggle_s_prime"] = QRectF(hx, hy, 110.0, btn_h)
        hx += 116.0
        self._btn_rects["toggle_hyperbolas"] = QRectF(hx, hy, 105.0, btn_h)
        hx += 111.0
        self._btn_rects["toggle_lightcone"] = QRectF(hx, hy, 95.0, btn_h)
        hx += 101.0

        # Quick velocity values: β = 0.0, 0.5, 0.8, 0.95
        for b_val in (0.0, 0.5, 0.8, 0.95):
            self._btn_rects[f"beta_{b_val}"] = QRectF(hx, hy, 52.0, btn_h)
            hx += 56.0

    # ── Public Interaction APIs ────────────────────────────────────────────────

    def set_beta(self, beta: float) -> None:
        """Set relative velocity v / c in (-0.99, 0.99)."""
        clamped = max(-0.99, min(0.99, float(beta)))
        if abs(clamped - self._beta) > 1e-4:
            self._beta = clamped
            self._update_lorentz_factors()
            s2, itype, _ = self.compute_spacetime_interval()
            self.beta_changed.emit(self._beta, self._gamma)
            self.interval_updated.emit(s2, itype)
            self.update()

    def load_preset(self, preset_code: str) -> None:
        """Load a special relativity curriculum preset."""
        if preset_code in PRESETS:
            self._active_preset_code = preset_code
            self._preset = PRESETS[preset_code]
            self.set_beta(self._preset.beta)
            self._e1 = self._preset.e1_pos
            self._e2 = self._preset.e2_pos
            self.preset_changed.emit(preset_code)
            self.update()

    def set_event1_pos(self, x: float, ct: float) -> None:
        """Set coordinates of Spacetime Event E1 in Frame S."""
        self._e1 = (float(x), float(ct))
        s2, itype, _ = self.compute_spacetime_interval()
        self.interval_updated.emit(s2, itype)
        self.update()

    def set_event2_pos(self, x: float, ct: float) -> None:
        """Set coordinates of Spacetime Event E2 in Frame S."""
        self._e2 = (float(x), float(ct))
        s2, itype, _ = self.compute_spacetime_interval()
        self.interval_updated.emit(s2, itype)
        self.update()

    def toggle_moving_frame(self) -> bool:
        """Toggle visualization of moving Frame S' coordinate axes."""
        self._show_moving_frame = not self._show_moving_frame
        self.update()
        return self._show_moving_frame

    def toggle_hyperbolas(self) -> bool:
        """Toggle invariant calibration hyperbolas."""
        self._show_hyperbolas = not self._show_hyperbolas
        self.update()
        return self._show_hyperbolas

    def toggle_light_cone(self) -> bool:
        """Toggle future and past light cone visualization."""
        self._show_light_cone = not self._show_light_cone
        self.update()
        return self._show_light_cone

    # ── Geometry & Coordinate Mapping ──────────────────────────────────────────

    def spacetime_to_scene(self, x: float, ct: float) -> QPointF:
        """Map spacetime coordinate (x, ct) to local scene point."""
        ox = self._diag_origin.x()
        oy = self._diag_origin.y()
        px = ox + (x * self._scale_px)
        py = oy - (ct * self._scale_px)  # ct points upward
        return QPointF(px, py)

    def scene_to_spacetime(self, pt: QPointF) -> tuple[float, float]:
        """Map local scene point to spacetime coordinate (x, ct)."""
        ox = self._diag_origin.x()
        oy = self._diag_origin.y()
        x = (pt.x() - ox) / self._scale_px
        ct = (oy - pt.y()) / self._scale_px
        return float(x), float(ct)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # ── Mouse Interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Check Button Clicks
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                event.accept()
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key == "toggle_s_prime":
                    self.toggle_moving_frame()
                elif key == "toggle_hyperbolas":
                    self.toggle_hyperbolas()
                elif key == "toggle_lightcone":
                    self.toggle_light_cone()
                elif key.startswith("beta_"):
                    b_val = float(key.replace("beta_", ""))
                    self.set_beta(b_val)
                return

        # 2. Check Velocity Slider Drag
        if self._slider_rect.adjusted(-6.0, -10.0, 6.0, 10.0).contains(pos):
            event.accept()
            self._drag_target = "slider"
            frac = (pos.x() - self._slider_rect.left()) / self._slider_rect.width()
            # Map [0, 1] to [-0.95, 0.95]
            beta_val = -0.95 + (frac * 1.90)
            self.set_beta(beta_val)
            return

        # 3. Check Event Probes Drag
        pt_e1 = self.spacetime_to_scene(self._e1[0], self._e1[1])
        if math.hypot(pos.x() - pt_e1.x(), pos.y() - pt_e1.y()) <= 14.0:
            event.accept()
            self._drag_target = "e1"
            return

        pt_e2 = self.spacetime_to_scene(self._e2[0], self._e2[1])
        if math.hypot(pos.x() - pt_e2.x(), pos.y() - pt_e2.y()) <= 14.0:
            event.accept()
            self._drag_target = "e2"
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._drag_target == "slider":
            event.accept()
            frac = (pos.x() - self._slider_rect.left()) / self._slider_rect.width()
            frac = max(0.0, min(1.0, frac))
            beta_val = -0.95 + (frac * 1.90)
            self.set_beta(beta_val)
            return
        elif self._drag_target == "e1":
            event.accept()
            x, ct = self.scene_to_spacetime(pos)
            self.set_event1_pos(round(x, 2), round(ct, 2))
            return
        elif self._drag_target == "e2":
            event.accept()
            x, ct = self.scene_to_spacetime(pos)
            self.set_event2_pos(round(x, 2), round(ct, 2))
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_target is not None:
            event.accept()
            self._drag_target = None
            return
        super().mouseReleaseEvent(event)

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process touchless air tap gesture from webcam tracking."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # Check button clicks
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key == "toggle_s_prime":
                    self.toggle_moving_frame()
                elif key == "toggle_hyperbolas":
                    self.toggle_hyperbolas()
                elif key == "toggle_lightcone":
                    self.toggle_light_cone()
                elif key.startswith("beta_"):
                    b_val = float(key.replace("beta_", ""))
                    self.set_beta(b_val)
                return True

        # Check slider tap
        if self._slider_rect.adjusted(-6.0, -10.0, 6.0, 10.0).contains(pos):
            frac = (pos.x() - self._slider_rect.left()) / self._slider_rect.width()
            frac = max(0.0, min(1.0, frac))
            beta_val = -0.95 + (frac * 1.90)
            self.set_beta(beta_val)
            return True

        return True

    # ── Painting ──────────────────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self.WIDTH
        h = self.HEIGHT

        # 1. Dark Glass Card Background
        card_rect = QRectF(0.0, 0.0, w, h)
        card_path = QPainterPath()
        card_path.addRoundedRect(card_rect, 12.0, 12.0)
        painter.fillPath(card_path, QBrush(QColor(15, 23, 42, 245)))
        border_pen = QPen(
            QColor(56, 189, 248, 240) if self.isSelected() else QColor(30, 41, 59, 240),
            2.0 if self.isSelected() else 1.5,
        )
        painter.setPen(border_pen)
        painter.drawPath(card_path)

        # 2. Header Bar
        header_rect = QRectF(0.0, 0.0, w, 36.0)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, 12.0, 12.0)
        painter.save()
        painter.setClipRect(0.0, 0.0, w, 36.0)
        painter.fillPath(header_path, QBrush(QColor(30, 41, 59, 230)))
        painter.restore()

        # Header Title
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(14, 23, "🌌 MINKOWSKI SPACETIME DIAGRAM & SPECIAL RELATIVITY ENGINE")

        # Lorentz Factor Badge
        beta_str = f"β = {self._beta:+.2f}  |  γ = {self._gamma:.3f}"
        b_rect = QRectF(w - 210.0, 7.0, 196.0, 22.0)
        b_path = QPainterPath()
        b_path.addRoundedRect(b_rect, 4.0, 4.0)
        painter.fillPath(b_path, QBrush(QColor(56, 189, 248, 35)))
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.drawPath(b_path)
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, beta_str)

        # 3. Toolbar Controls
        self._paint_toolbar(painter)

        # 4. Spacetime Coordinate Grid & Diagrams
        self._paint_minkowski_diagram(painter)

        # 5. Right Relativistic Telemetry & Invariance Panel
        self._paint_telemetry_panel(painter)

        # 6. Bottom Velocity Slider
        self._paint_velocity_slider(painter)

        painter.restore()

    def _paint_toolbar(self, painter: QPainter) -> None:
        """Paint preset tabs and feature toggle buttons."""
        # Presets row
        for code, pr in PRESETS.items():
            rect = self._btn_rects.get(f"preset_{code}")
            if not rect:
                continue
            is_active = (code == self._active_preset_code)
            bg_col = QColor(56, 189, 248, 50) if is_active else QColor(30, 41, 59, 160)
            txt_col = QColor("#38bdf8") if is_active else QColor("#94a3b8")

            path = QPainterPath()
            path.addRoundedRect(rect, 4.0, 4.0)
            painter.fillPath(path, QBrush(bg_col))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.drawPath(path)
            painter.setPen(QPen(txt_col))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, pr.name_vi[:22])

        # Toggles row
        toggles = [
            ("toggle_s_prime", "Hệ quy chiếu S'", self._show_moving_frame, "#38bdf8"),
            ("toggle_hyperbolas", "Đường Hyperbol", self._show_hyperbolas, "#f59e0b"),
            ("toggle_lightcone", "Nón Ánh sáng", self._show_light_cone, "#e2e8f0"),
        ]
        for key, label, is_active, col_hex in toggles:
            rect = self._btn_rects.get(key)
            if not rect:
                continue
            c = QColor(col_hex)
            bg_col = QColor(c.red(), c.green(), c.blue(), 50 if is_active else 15)
            path = QPainterPath()
            path.addRoundedRect(rect, 4.0, 4.0)
            painter.fillPath(path, QBrush(bg_col))
            painter.setPen(QPen(c if is_active else QColor("#475569"), 1.0))
            painter.drawPath(path)
            painter.setPen(QPen(c if is_active else QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # Quick velocity values
        for b_val in (0.0, 0.5, 0.8, 0.95):
            rect = self._btn_rects.get(f"beta_{b_val}")
            if not rect:
                continue
            is_cur = abs(self._beta - b_val) < 0.04
            painter.setPen(QPen(QColor("#38bdf8") if is_cur else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(56, 189, 248, 50) if is_cur else QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#38bdf8") if is_cur else QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"β={b_val:.2f}")

    def _paint_minkowski_diagram(self, painter: QPainter) -> None:
        """Paint Minkowski coordinate system, light cone, moving axes, and event probes."""
        painter.save()
        clip_rect = QRectF(14.0, 98.0, 380.0, 335.0)
        painter.setClipRect(clip_rect)

        # Diagram Background
        bg_path = QPainterPath()
        bg_path.addRoundedRect(clip_rect, 6.0, 6.0)
        painter.fillPath(bg_path, QBrush(QColor(10, 15, 30, 240)))
        painter.setPen(QPen(QColor(30, 41, 59, 200), 1.0))
        painter.drawPath(bg_path)

        ox = self._diag_origin.x()
        oy = self._diag_origin.y()
        scale = self._scale_px

        # 1. Stationary Frame S Grid (orthogonal)
        painter.setPen(QPen(QColor(255, 255, 255, 20), 0.8, Qt.PenStyle.DashLine))
        for u in range(-5, 6):
            # Vertical lines (x = const)
            x_pos = ox + (u * scale)
            painter.drawLine(QPointF(x_pos, clip_rect.top()), QPointF(x_pos, clip_rect.bottom()))
            # Horizontal lines (ct = const)
            y_pos = oy - (u * scale)
            painter.drawLine(QPointF(clip_rect.left(), y_pos), QPointF(clip_rect.right(), y_pos))

        # 2. Light Cone (45 degree lines: x = ct and x = -ct)
        if self._show_light_cone:
            # Future Light Cone shaded area
            future_cone = QPainterPath()
            future_cone.moveTo(ox, oy)
            future_cone.lineTo(ox - (6.0 * scale), oy - (6.0 * scale))
            future_cone.lineTo(ox + (6.0 * scale), oy - (6.0 * scale))
            future_cone.closeSubpath()
            painter.fillPath(future_cone, QBrush(QColor(245, 158, 11, 20)))

            # Past Light Cone shaded area
            past_cone = QPainterPath()
            past_cone.moveTo(ox, oy)
            past_cone.lineTo(ox - (6.0 * scale), oy + (6.0 * scale))
            past_cone.lineTo(ox + (6.0 * scale), oy + (6.0 * scale))
            past_cone.closeSubpath()
            painter.fillPath(past_cone, QBrush(QColor(56, 189, 248, 15)))

            # Light cone boundary lines
            cone_pen = QPen(QColor(251, 191, 36, 180), 1.5, Qt.PenStyle.DashDotLine)
            painter.setPen(cone_pen)
            painter.drawLine(QPointF(ox - (6.0 * scale), oy + (6.0 * scale)), QPointF(ox + (6.0 * scale), oy - (6.0 * scale)))
            painter.drawLine(QPointF(ox - (6.0 * scale), oy - (6.0 * scale)), QPointF(ox + (6.0 * scale), oy + (6.0 * scale)))

            painter.setPen(QPen(QColor(251, 191, 36, 200)))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(int(ox + 45.0), int(oy - 50.0), "Tia sáng c (x = ct)")

        # 3. Invariant Calibration Hyperbolas: c^2 t^2 - x^2 = ±1
        if self._show_hyperbolas:
            hyp_pen_t = QPen(QColor(56, 189, 248, 120), 1.0, Qt.PenStyle.DotLine)
            painter.setPen(hyp_pen_t)
            # Timelike hyperbola: ct = sqrt(x^2 + 1)
            t_path = QPainterPath()
            started = False
            for step in range(-45, 46):
                x_val = step * 0.1
                ct_val = math.sqrt(x_val * x_val + 1.0)
                pt = self.spacetime_to_scene(x_val, ct_val)
                if not started:
                    t_path.moveTo(pt)
                    started = True
                else:
                    t_path.lineTo(pt)
            painter.drawPath(t_path)

            # Spacelike hyperbola: x = sqrt(ct^2 + 1)
            hyp_pen_s = QPen(QColor(245, 158, 11, 120), 1.0, Qt.PenStyle.DotLine)
            painter.setPen(hyp_pen_s)
            s_path = QPainterPath()
            started_s = False
            for step in range(-45, 46):
                ct_val = step * 0.1
                x_val = math.sqrt(ct_val * ct_val + 1.0)
                pt = self.spacetime_to_scene(x_val, ct_val)
                if not started_s:
                    s_path.moveTo(pt)
                    started_s = True
                else:
                    s_path.lineTo(pt)
            painter.drawPath(s_path)

        # 4. Moving Frame S' Coordinate Axes (Tilted by angle θ = arctan(β))
        if self._show_moving_frame and abs(self._beta) > 1e-4:
            beta = self._beta
            # ct'-axis worldline: x = β * ct -> points slope 1/β
            s_prime_pen = QPen(QColor("#ec4899"), 1.8)
            painter.setPen(s_prime_pen)
            p_ct_neg = self.spacetime_to_scene(-5.0 * beta, -5.0)
            p_ct_pos = self.spacetime_to_scene(5.0 * beta, 5.0)
            painter.drawLine(p_ct_neg, p_ct_pos)

            # x'-axis line of simultaneity: ct = β * x -> points slope β
            p_x_neg = self.spacetime_to_scene(-5.0, -5.0 * beta)
            p_x_pos = self.spacetime_to_scene(5.0, 5.0 * beta)
            painter.drawLine(p_x_neg, p_x_pos)

            painter.setPen(QPen(QColor("#f472b6")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(p_ct_pos + QPointF(6.0, 4.0), "ct'")
            painter.drawText(p_x_pos + QPointF(6.0, 12.0), "x'")

        # 5. Stationary Frame S Axes (ct and x)
        axis_pen = QPen(QColor("#38bdf8"), 2.0)
        painter.setPen(axis_pen)
        # ct-axis (vertical)
        painter.drawLine(QPointF(ox, clip_rect.top() + 8.0), QPointF(ox, clip_rect.bottom() - 8.0))
        # x-axis (horizontal)
        painter.drawLine(QPointF(clip_rect.left() + 8.0, oy), QPointF(clip_rect.right() - 8.0, oy))

        # Axis arrowheads & Labels
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(int(ox + 8.0), int(clip_rect.top() + 20.0), "Thời gian ct")
        painter.drawText(int(clip_rect.right() - 75.0), int(oy - 8.0), "Không gian x")

        # 6. Spacetime Interval Vector Δs between E1 and E2
        pt_e1 = self.spacetime_to_scene(self._e1[0], self._e1[1])
        pt_e2 = self.spacetime_to_scene(self._e2[0], self._e2[1])

        if self._show_interval_vector:
            s2, itype, proper = self.compute_spacetime_interval()
            vec_col = QColor("#34d399") if s2 > 0 else (QColor("#f87171") if s2 < 0 else QColor("#fbbf24"))
            vec_pen = QPen(vec_col, 2.0, Qt.PenStyle.SolidLine)
            painter.setPen(vec_pen)
            painter.drawLine(pt_e1, pt_e2)

        # 7. Draw Event Probes E1 and E2
        events = [
            ("E1", self._e1, pt_e1, QColor("#38bdf8")),
            ("E2", self._e2, pt_e2, QColor("#10b981")),
        ]
        for name, coords, pt, col in events:
            # Outer glow ring
            painter.setPen(QPen(col, 2.0))
            painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 60)))
            painter.drawEllipse(pt, 9.0, 9.0)
            painter.setBrush(QBrush(col))
            painter.drawEllipse(pt, 3.5, 3.5)

            # Label
            painter.setPen(QPen(QColor("#ffffff")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(int(pt.x() + 12.0), int(pt.y() - 4.0), f"{name}({coords[0]:.1f}, {coords[1]:.1f})")

        painter.restore()

    def _paint_telemetry_panel(self, painter: QPainter) -> None:
        """Paint comparative Frame S vs Frame S' coordinates and Lorentz kinematics breakdown."""
        panel_rect = QRectF(408.0, 98.0, 338.0, 386.0)

        # Glass Panel
        panel_path = QPainterPath()
        panel_path.addRoundedRect(panel_rect, 8.0, 8.0)
        painter.fillPath(panel_path, QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(30, 41, 59, 240), 1.2))
        painter.drawPath(panel_path)

        x_left = panel_rect.x() + 14.0
        y_top = panel_rect.y() + 22.0

        # Section 1: Invariant Spacetime Interval Δs²
        s2, itype, proper = self.compute_spacetime_interval()
        status_col = "#34d399" if s2 > 0 else ("#f87171" if s2 < 0 else "#fbbf24")

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(int(x_left), int(y_top), "KHOẢNG KHÔNG-THỜI GIAN BẤT BIẾN (INVARIANT Δs²):")

        # Invariant equation box
        inv_box = QRectF(x_left, y_top + 10.0, panel_rect.width() - 28.0, 52.0)
        inv_path = QPainterPath()
        inv_path.addRoundedRect(inv_box, 6.0, 6.0)
        painter.fillPath(inv_path, QBrush(QColor(30, 41, 59, 180)))
        painter.setPen(QPen(QColor(status_col), 1.2))
        painter.drawPath(inv_path)

        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(status_col)))
        painter.drawText(int(inv_box.x() + 10.0), int(inv_box.y() + 20.0), f"Δs² = c²Δt² - Δx² = {s2:+.3f}")
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(int(inv_box.x() + 10.0), int(inv_box.y() + 38.0), f"Phân loại: {itype}")

        # Section 2: Frame Comparison Table
        y_tab = y_top + 76.0
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(int(x_left), int(y_tab), "TỌA ĐỘ SỰ KIỆN QUA BIẾN ĐỔI LORENTZ:")

        # Compute coordinates in Frame S'
        x1_p, ct1_p = self.lorentz_transform(self._e1[0], self._e1[1])
        x2_p, ct2_p = self.lorentz_transform(self._e2[0], self._e2[1])
        dx = self._e2[0] - self._e1[0]
        dct = self._e2[1] - self._e1[1]
        dx_p = x2_p - x1_p
        dct_p = ct2_p - ct1_p

        # Table Header
        th_y = y_tab + 18.0
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(int(x_left), int(th_y), "Đại lượng")
        painter.drawText(int(x_left + 110.0), int(th_y), "Hệ S (Trái Đất)")
        painter.drawText(int(x_left + 220.0), int(th_y), "Hệ S' (Chuyển động)")

        rows = [
            ("Sự kiện E1:", f"({self._e1[0]:.2f}, {self._e1[1]:.2f})", f"({x1_p:.2f}, {ct1_p:.2f})"),
            ("Sự kiện E2:", f"({self._e2[0]:.2f}, {self._e2[1]:.2f})", f"({x2_p:.2f}, {ct2_p:.2f})"),
            ("Khoảng cách Δx:", f"{dx:+.2f} ls", f"{dx_p:+.2f} ls"),
            ("Thời gian cΔt:", f"{dct:+.2f} ls", f"{dct_p:+.2f} ls"),
            ("Độ dãn thời gian:", f"Δt = {abs(dct):.2f}", f"Δt' = {abs(dct_p):.2f}"),
        ]

        row_y = th_y + 16.0
        for label, val_s, val_sp in rows:
            painter.setPen(QPen(QColor("#cbd5e1")))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(int(x_left), int(row_y), label)

            painter.setPen(QPen(QColor("#38bdf8")))
            painter.setFont(QFont("Consolas", 7))
            painter.drawText(int(x_left + 110.0), int(row_y), val_s)

            painter.setPen(QPen(QColor("#f472b6")))
            painter.drawText(int(x_left + 220.0), int(row_y), val_sp)
            row_y += 18.0

        # Section 3: Relativistic Kinematics Formulas
        f_rect = QRectF(x_left, row_y + 6.0, panel_rect.width() - 28.0, 84.0)
        f_path = QPainterPath()
        f_path.addRoundedRect(f_rect, 6.0, 6.0)
        painter.fillPath(f_path, QBrush(QColor(15, 23, 42, 200)))
        painter.setPen(QPen(QColor(56, 189, 248, 120), 1.0))
        painter.drawPath(f_path)

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(int(f_rect.x() + 8.0), int(f_rect.y() + 18.0), "Lorentz Boost:")
        painter.drawText(int(f_rect.x() + 8.0), int(f_rect.y() + 34.0), "  ct' = γ·(ct - β·x)   x' = γ·(x - β·ct)")
        painter.drawText(int(f_rect.x() + 8.0), int(f_rect.y() + 52.0), f"  γ = 1 / √(1 - β²) = {self._gamma:.4f}")
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(int(f_rect.x() + 8.0), int(f_rect.y() + 70.0), f"  Đồng thời tính: θ_nghiêng = {math.degrees(math.atan(self._beta)):.1f}°")

        # Scenario Description
        desc_rect = QRectF(x_left, f_rect.bottom() + 10.0, panel_rect.width() - 28.0, 50.0)
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(desc_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), self._preset.description)

    def _paint_velocity_slider(self, painter: QPainter) -> None:
        """Paint interactive velocity slider β = v / c."""
        # Track line
        track_rect = self._slider_rect
        track_path = QPainterPath()
        track_path.addRoundedRect(track_rect, 4.0, 4.0)
        painter.fillPath(track_path, QBrush(QColor(30, 41, 59, 220)))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawPath(track_path)

        # Center line (beta = 0)
        cx = track_rect.left() + (track_rect.width() * 0.5)
        painter.setPen(QPen(QColor("#64748b"), 1.0))
        painter.drawLine(QPointF(cx, track_rect.top() - 3.0), QPointF(cx, track_rect.bottom() + 3.0))

        # Filled active bar from 0 to beta
        # Map beta in [-0.95, 0.95] to frac in [0, 1]
        frac = (self._beta - (-0.95)) / 1.90
        handle_x = track_rect.left() + (frac * track_rect.width())

        bar_rect = QRectF(min(cx, handle_x), track_rect.top() + 2.0, abs(handle_x - cx), track_rect.height() - 4.0)
        painter.fillRect(bar_rect, QBrush(QColor(56, 189, 248, 180)))

        # Handle
        handle_rect = QRectF(handle_x - 6.0, track_rect.center().y() - 9.0, 12.0, 18.0)
        h_path = QPainterPath()
        h_path.addRoundedRect(handle_rect, 4.0, 4.0)
        painter.fillPath(h_path, QBrush(QColor("#38bdf8")))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawPath(h_path)

        # Labels
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(int(track_rect.left()), int(track_rect.top() - 6.0), "Vận tốc tương đối: -0.95c")
        painter.drawText(int(cx - 8.0), int(track_rect.top() - 6.0), "0.0")
        painter.drawText(int(track_rect.right() - 30.0), int(track_rect.top() - 6.0), "+0.95c")

    # ── Canvas Stamping ────────────────────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """Stamp current relativity diagram onto LessonBoard."""
        stamped: list[QGraphicsItem] = []
        copy_item = InteractiveSpecialRelativityItem(pos=self.pos())
        copy_item.load_preset(self._active_preset_code)
        copy_item.set_beta(self._beta)
        copy_item.set_event1_pos(self._e1[0], self._e1[1])
        copy_item.set_event2_pos(self._e2[0], self._e2[1])
        board.addItem(copy_item)
        stamped.append(copy_item)
        return stamped

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for .airteach lesson packages."""
        return {
            "type": "interactive_special_relativity",
            "pos": [self.x(), self.y()],
            "preset_code": self._active_preset_code,
            "beta": round(self._beta, 4),
            "e1": list(self._e1),
            "e2": list(self._e2),
            "show_moving_frame": self._show_moving_frame,
            "show_hyperbolas": self._show_hyperbolas,
            "show_light_cone": self._show_light_cone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractiveSpecialRelativityItem:
        """Reconstruct visualizer from serialized dictionary."""
        pos = data.get("pos", [120.0, 50.0])
        item = cls(pos=(float(pos[0]), float(pos[1])))
        if data.get("preset_code"):
            item.load_preset(str(data.get("preset_code")))
        item.set_beta(float(data.get("beta", 0.60)))
        e1 = data.get("e1")
        if e1 and len(e1) == 2:
            item.set_event1_pos(float(e1[0]), float(e1[1]))
        e2 = data.get("e2")
        if e2 and len(e2) == 2:
            item.set_event2_pos(float(e2[0]), float(e2[1]))
        item._show_moving_frame = bool(data.get("show_moving_frame", True))
        item._show_hyperbolas = bool(data.get("show_hyperbolas", True))
        item._show_light_cone = bool(data.get("show_light_cone", True))
        return item
