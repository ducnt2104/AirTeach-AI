"""
AirTeach AI - Interactive Trigonometric Unit Circle Simulation Suite
====================================================================
Provides an interactive mathematical unit circle (Đường tròn lượng giác):
  - Dynamic rotating angle arm with continuous angle theta in degrees and radians.
  - Trigonometric projections:
      * cos(theta) along horizontal Ox axis (Cyan)
      * sin(theta) along vertical Oy axis (Amber)
      * tan(theta) along tangent axis x = R (Magenta)
      * cot(theta) along cotangent axis y = -R (Green)
  - Automatic cardinal angle snapping (0°, 30°, 45°, 60°, 90°, 120°, 135°, 150°, 180°, etc.).
  - Exact algebraic values table (1/2, sqrt(2)/2, sqrt(3)/2, 1, undefined).
  - Telemetry HUD badge displaying angle, quadrant, numerical and exact values.
  - Interactive mouse dragging and touchless air-gesture controls (+15°, -15°, Snap).
  - High-DPI crisp vector rendering for projectors and interactive whiteboards.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

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

logger = logging.getLogger(__name__)

# Cardinal angles dictionary with exact mathematical values
CARDINAL_EXACT_VALUES: dict[int, dict[str, str]] = {
    0: {"rad": "0", "sin": "0", "cos": "1", "tan": "0", "cot": "không xđ"},
    30: {"rad": "π/6", "sin": "1/2", "cos": "√3/2", "tan": "√3/3", "cot": "√3"},
    45: {"rad": "π/4", "sin": "√2/2", "cos": "√2/2", "tan": "1", "cot": "1"},
    60: {"rad": "π/3", "sin": "√3/2", "cos": "1/2", "tan": "√3", "cot": "√3/3"},
    90: {"rad": "π/2", "sin": "1", "cos": "0", "tan": "không xđ", "cot": "0"},
    120: {"rad": "2π/3", "sin": "√3/2", "cos": "-1/2", "tan": "-√3", "cot": "-√3/3"},
    135: {"rad": "3π/4", "sin": "√2/2", "cos": "-√2/2", "tan": "-1", "cot": "-1"},
    150: {"rad": "5π/6", "sin": "1/2", "cos": "-√3/2", "tan": "-√3/3", "cot": "-√3"},
    180: {"rad": "π", "sin": "0", "cos": "-1", "tan": "0", "cot": "không xđ"},
    210: {"rad": "7π/6", "sin": "-1/2", "cos": "-√3/2", "tan": "√3/3", "cot": "√3"},
    225: {"rad": "5π/4", "sin": "-√2/2", "cos": "-√2/2", "tan": "1", "cot": "1"},
    240: {"rad": "4π/3", "sin": "-√3/2", "cos": "-1/2", "tan": "√3", "cot": "√3/3"},
    270: {"rad": "3π/2", "sin": "-1", "cos": "0", "tan": "không xđ", "cot": "0"},
    300: {"rad": "5π/3", "sin": "-√3/2", "cos": "1/2", "tan": "-√3", "cot": "-√3/3"},
    315: {"rad": "7π/4", "sin": "-√2/2", "cos": "√2/2", "tan": "-1", "cot": "-1"},
    330: {"rad": "11π/6", "sin": "-1/2", "cos": "√3/2", "tan": "-√3/3", "cot": "-√3"},
    360: {"rad": "2π", "sin": "0", "cos": "1", "tan": "0", "cot": "không xđ"},
}


class InteractiveTrigonometricUnitCircleItem(QGraphicsObject):
    """
    Interactive Trigonometric Unit Circle vector canvas item.
    Enables interactive exploration of trigonometric functions, angles, and projections.
    """

    angle_changed = Signal(float)  # theta in degrees

    def __init__(
        self,
        center: tuple[float, float] = (500.0, 400.0),
        radius: float = 180.0,
        angle_deg: float = 45.0,
        show_tangent: bool = True,
        show_cotangent: bool = True,
        show_projections: bool = True,
        show_hud_badge: bool = True,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(float(center[0]), float(center[1]))
        self._radius = max(80.0, float(radius))
        self._angle_deg = float(angle_deg) % 360.0

        self._show_tangent = show_tangent
        self._show_cotangent = show_cotangent
        self._show_projections = show_projections
        self._show_hud_badge = show_hud_badge

        self._is_dragging_handle: bool = False

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    # ── Geometry & Properties ────────────────────────────────────────────────

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def angle_deg(self) -> float:
        return self._angle_deg

    @property
    def angle_rad(self) -> float:
        return math.radians(self._angle_deg)

    @property
    def sin_val(self) -> float:
        return math.sin(self.angle_rad)

    @property
    def cos_val(self) -> float:
        return math.cos(self.angle_rad)

    @property
    def tan_val(self) -> Optional[float]:
        cos_v = self.cos_val
        if abs(cos_v) < 1e-7:
            return None
        return math.tan(self.angle_rad)

    @property
    def cot_val(self) -> Optional[float]:
        sin_v = self.sin_val
        if abs(sin_v) < 1e-7:
            return None
        return 1.0 / math.tan(self.angle_rad)

    @property
    def quadrant_name(self) -> str:
        ang = self._angle_deg % 360.0
        if 0.0 < ang < 90.0:
            return "Góc phần tư I"
        elif 90.0 < ang < 180.0:
            return "Góc phần tư II"
        elif 180.0 < ang < 270.0:
            return "Góc phần tư III"
        elif 270.0 < ang < 360.0:
            return "Góc phần tư IV"
        else:
            return "Trục tọa độ"

    def set_angle_deg(self, deg: float) -> None:
        self.prepareGeometryChange()
        self._angle_deg = float(deg) % 360.0
        self.angle_changed.emit(self._angle_deg)
        self.update()

    def rotate_by(self, delta_deg: float) -> float:
        """Incrementally rotate angle arm by delta degrees."""
        new_ang = (self._angle_deg + delta_deg) % 360.0
        self.set_angle_deg(new_ang)
        return new_ang

    def snap_to_cardinal_angle(self, tolerance_deg: float = 4.0) -> Optional[float]:
        """Snap current angle to the closest cardinal angle if within tolerance."""
        curr = self._angle_deg % 360.0
        for card in CARDINAL_EXACT_VALUES.keys():
            diff = abs(curr - card)
            if diff <= tolerance_deg or abs(diff - 360.0) <= tolerance_deg:
                target = 0.0 if card == 360 else float(card)
                self.set_angle_deg(target)
                return target
        return None

    def get_exact_values(self) -> dict[str, str]:
        """Return exact algebraic representation if snapped, or formatted floating-point strings."""
        rounded = int(round(self._angle_deg)) % 360
        if rounded in CARDINAL_EXACT_VALUES and abs(self._angle_deg - rounded) < 0.2:
            return dict(CARDINAL_EXACT_VALUES[rounded])
        return {
            "rad": f"{self.angle_rad:.3f} rad",
            "sin": f"{self.sin_val:.4f}",
            "cos": f"{self.cos_val:.4f}",
            "tan": f"{self.tan_val:.4f}" if self.tan_val is not None else "không xđ",
            "cot": f"{self.cot_val:.4f}" if self.cot_val is not None else "không xđ",
        }

    # ── Display Toggles ──────────────────────────────────────────────────────

    def set_show_tangent(self, show: bool) -> None:
        self._show_tangent = show
        self.update()

    def set_show_cotangent(self, show: bool) -> None:
        self._show_cotangent = show
        self.update()

    def set_show_projections(self, show: bool) -> None:
        self._show_projections = show
        self.update()

    def set_show_hud_badge(self, show: bool) -> None:
        self._show_hud_badge = show
        self.update()

    # ── QGraphicsItem Overrides ──────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        r = self._radius
        margin = 160.0
        return QRectF(-r - margin, -r - margin, 2 * r + 2 * margin + 220.0, 2 * r + 2 * margin + 60.0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        r = self._radius
        ang_rad = self.angle_rad
        cos_v = self.cos_val
        sin_v = self.sin_val

        # 1. Background Panel Frame (Glassmorphism dark card)
        panel_rect = self.boundingRect().adjusted(10.0, 10.0, -10.0, -10.0)
        painter.setBrush(QBrush(QColor(20, 24, 33, 235)))
        painter.setPen(QPen(QColor(60, 75, 105, 180), 1.5))
        painter.drawRoundedRect(panel_rect, 16.0, 16.0)

        # 2. Unit Circle Base
        painter.setBrush(QBrush(QColor(30, 42, 68, 80)))
        painter.setPen(QPen(QColor(66, 133, 244, 180), 2.2))
        painter.drawEllipse(QPointF(0.0, 0.0), r, r)

        # 3. Coordinate Axes (Ox: cos axis, Oy: sin axis)
        axis_pen = QPen(QColor(160, 175, 205), 1.5)
        painter.setPen(axis_pen)
        # Ox axis
        painter.drawLine(QPointF(-r - 35.0, 0.0), QPointF(r + 45.0, 0.0))
        # Oy axis (in canvas coordinate, y points down, so top is -y)
        painter.drawLine(QPointF(0.0, r + 35.0), QPointF(0.0, -r - 45.0))

        # Axis arrowheads
        painter.setBrush(QBrush(QColor(160, 175, 205)))
        arrow_x = QPainterPath()
        arrow_x.moveTo(r + 45.0, 0.0)
        arrow_x.lineTo(r + 35.0, -4.0)
        arrow_x.lineTo(r + 35.0, 4.0)
        arrow_x.closeSubpath()
        painter.drawPath(arrow_x)

        arrow_y = QPainterPath()
        arrow_y.moveTo(0.0, -r - 45.0)
        arrow_y.lineTo(-4.0, -r - 35.0)
        arrow_y.lineTo(4.0, -r - 35.0)
        arrow_y.closeSubpath()
        painter.drawPath(arrow_y)

        # Axis labels
        font_axis = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font_axis)
        painter.setPen(QPen(QColor("#00e5ff")))
        painter.drawText(QRectF(r + 50.0, -12.0, 60.0, 24.0), Qt.AlignmentFlag.AlignLeft, "cos (x)")
        painter.setPen(QPen(QColor("#ffab00")))
        painter.drawText(QRectF(10.0, -r - 50.0, 60.0, 24.0), Qt.AlignmentFlag.AlignLeft, "sin (y)")

        # Tick marks (-1, 0, 1)
        font_tick = QFont("Segoe UI", 9)
        painter.setFont(font_tick)
        painter.setPen(QPen(QColor(180, 190, 210)))
        painter.drawText(QRectF(r - 10.0, 6.0, 20.0, 18.0), Qt.AlignmentFlag.AlignCenter, "1")
        painter.drawText(QRectF(-r - 10.0, 6.0, 20.0, 18.0), Qt.AlignmentFlag.AlignCenter, "-1")
        painter.drawText(QRectF(-20.0, -r - 12.0, 16.0, 18.0), Qt.AlignmentFlag.AlignRight, "1")
        painter.drawText(QRectF(-25.0, r - 6.0, 20.0, 18.0), Qt.AlignmentFlag.AlignRight, "-1")
        painter.drawText(QRectF(-16.0, 4.0, 14.0, 16.0), Qt.AlignmentFlag.AlignRight, "O")

        # 4. Tangent Axis x = R and Cotangent Axis y = -R
        if self._show_tangent:
            tan_pen = QPen(QColor(255, 64, 129, 140), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(tan_pen)
            painter.drawLine(QPointF(r, -r - 40.0), QPointF(r, r + 40.0))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ff4081")))
            painter.drawText(QRectF(r + 6.0, -r - 35.0, 45.0, 18.0), Qt.AlignmentFlag.AlignLeft, "tan (t)")

        if self._show_cotangent:
            cot_pen = QPen(QColor(76, 175, 80, 140), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(cot_pen)
            painter.drawLine(QPointF(-r - 40.0, -r), QPointF(r + 40.0, -r))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#4caf50")))
            painter.drawText(QRectF(-r - 40.0, -r - 22.0, 45.0, 18.0), Qt.AlignmentFlag.AlignLeft, "cot")

        # 5. Angle Arc
        arc_pen = QPen(QColor(255, 215, 0, 200), 2.0)
        painter.setPen(arc_pen)
        painter.setBrush(QBrush(QColor(255, 215, 0, 30)))
        arc_rect = QRectF(-36.0, -36.0, 72.0, 72.0)
        # QPainter drawPie expects angles in 1/16th of a degree
        painter.drawPie(arc_rect, 0, int(round(self._angle_deg * 16)))

        # 6. Point P on unit circle
        px = r * cos_v
        py = -r * sin_v  # Note: screen y is inverted

        # Projections to Axes
        if self._show_projections:
            # Cos projection (Vertical line from P to x-axis, highlighting cos on Ox)
            painter.setPen(QPen(QColor("#00e5ff"), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(px, py), QPointF(px, 0.0))

            # Sin projection (Horizontal line from P to y-axis, highlighting sin on Oy)
            painter.setPen(QPen(QColor("#ffab00"), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(px, py), QPointF(0.0, py))

            # Highlight segments on axes
            painter.setPen(QPen(QColor("#00e5ff"), 3.5))
            painter.drawLine(QPointF(0.0, 0.0), QPointF(px, 0.0))

            painter.setPen(QPen(QColor("#ffab00"), 3.5))
            painter.drawLine(QPointF(0.0, 0.0), QPointF(0.0, py))

        # 7. Tangent Ray Intersection
        if self._show_tangent and abs(cos_v) > 1e-4:
            tan_v = sin_v / cos_v
            tan_y = -r * tan_v
            # Ray from origin through P to tangent line
            painter.setPen(QPen(QColor(255, 64, 129, 100), 1.2, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(0.0, 0.0), QPointF(r, tan_y))

            # Tangent value segment on x = r
            painter.setPen(QPen(QColor("#ff4081"), 3.0))
            painter.drawLine(QPointF(r, 0.0), QPointF(r, tan_y))

            # Intersection point T
            painter.setBrush(QBrush(QColor("#ff4081")))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.drawEllipse(QPointF(r, tan_y), 4.5, 4.5)

        # 8. Main Rotating Angle Arm (Radius Vector OP)
        arm_pen = QPen(QColor("#ffffff"), 2.4)
        painter.setPen(arm_pen)
        painter.drawLine(QPointF(0.0, 0.0), QPointF(px, py))

        # Handle Point P
        painter.setBrush(QBrush(QColor("#ff5252")))
        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        painter.drawEllipse(QPointF(px, py), 7.0, 7.0)

        # Label P
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        label_offset = 14.0
        label_x = px + label_offset * (1 if px >= 0 else -1) - 6.0
        label_y = py + label_offset * (1 if py >= 0 else -1)
        painter.drawText(QPointF(label_x, label_y), "P(cos, sin)")

        # 9. Touchless Step Buttons on Bottom
        # Button [ -15° ]
        b_minus_rect = QRectF(-r - 60.0, r + 45.0, 60.0, 26.0)
        painter.setBrush(QBrush(QColor("#2d3748")))
        painter.setPen(QPen(QColor("#4a5568"), 1.2))
        painter.drawRoundedRect(b_minus_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#e2e8f0")))
        painter.drawText(b_minus_rect, Qt.AlignmentFlag.AlignCenter, "-15°")

        # Button [ +15° ]
        b_plus_rect = QRectF(-r + 10.0, r + 45.0, 60.0, 26.0)
        painter.setBrush(QBrush(QColor("#2d3748")))
        painter.setPen(QPen(QColor("#4a5568"), 1.2))
        painter.drawRoundedRect(b_plus_rect, 6.0, 6.0)
        painter.drawText(b_plus_rect, Qt.AlignmentFlag.AlignCenter, "+15°")

        # Button [ Snap ]
        b_snap_rect = QRectF(-r + 80.0, r + 45.0, 60.0, 26.0)
        painter.setBrush(QBrush(QColor("#1a73e8")))
        painter.setPen(QPen(QColor("#4285f4"), 1.2))
        painter.drawRoundedRect(b_snap_rect, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(b_snap_rect, Qt.AlignmentFlag.AlignCenter, "Snap")

        # 10. HUD Telemetry Card on Right Side
        if self._show_hud_badge:
            hud_x = r + 90.0
            hud_y = -r - 10.0
            hud_w = 200.0
            hud_h = 240.0
            hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

            painter.setBrush(QBrush(QColor(15, 20, 28, 220)))
            painter.setPen(QPen(QColor(45, 60, 85), 1.2))
            painter.drawRoundedRect(hud_rect, 10.0, 10.0)

            # Header
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#64b5f6")))
            painter.drawText(QRectF(hud_x + 12.0, hud_y + 10.0, hud_w - 24.0, 20.0), Qt.AlignmentFlag.AlignLeft, "📐 GIÁ TRỊ LƯỢNG GIÁC")

            exact = self.get_exact_values()

            painter.setFont(QFont("Segoe UI", 9))
            lines = [
                (f"Góc θ: {self._angle_deg:.1f}° ({exact['rad']})", "#ffffff"),
                (f"{self.quadrant_name}", "#b0bec5"),
                (f"sin(θ) = {self.sin_val:.4f}", "#ffab00"),
                (f"  → Đúng: {exact['sin']}", "#ffe082"),
                (f"cos(θ) = {self.cos_val:.4f}", "#00e5ff"),
                (f"  → Đúng: {exact['cos']}", "#80deea"),
                (f"tan(θ) = {exact['tan']}", "#ff4081"),
                (f"cot(θ) = {exact['cot']}", "#4caf50"),
            ]

            curr_y = hud_y + 36.0
            for text, col in lines:
                painter.setPen(QPen(QColor(col)))
                painter.drawText(QRectF(hud_x + 12.0, curr_y, hud_w - 24.0, 18.0), Qt.AlignmentFlag.AlignLeft, text)
                curr_y += 24.0

    # ── Mouse Interaction ────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        r = self._radius
        px = r * self.cos_val
        py = -r * self.sin_val

        # Handle Point P drag
        if math.hypot(pos.x() - px, pos.y() - py) <= 16.0:
            self._is_dragging_handle = True
            event.accept()
            return

        # Check buttons
        b_minus_rect = QRectF(-r - 60.0, r + 45.0, 60.0, 26.0)
        b_plus_rect = QRectF(-r + 10.0, r + 45.0, 60.0, 26.0)
        b_snap_rect = QRectF(-r + 80.0, r + 45.0, 60.0, 26.0)

        if b_minus_rect.contains(pos):
            self.rotate_by(-15.0)
            event.accept()
            return
        elif b_plus_rect.contains(pos):
            self.rotate_by(15.0)
            event.accept()
            return
        elif b_snap_rect.contains(pos):
            self.snap_to_cardinal_angle(tolerance_deg=180.0)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_handle:
            pos = event.pos()
            # Invert y since screen y is downwards
            ang_rad = math.atan2(-pos.y(), pos.x())
            ang_deg = math.degrees(ang_rad) % 360.0
            self.set_angle_deg(ang_deg)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_handle:
            self._is_dragging_handle = False
            # Auto-snap if close to cardinal angle
            self.snap_to_cardinal_angle(tolerance_deg=3.5)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── Touchless Air Tap Interaction ────────────────────────────────────────

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """
        Handle touchless air tap on interactive buttons or circle perimeter.
        Returns True if the air tap interacted with any controls.
        """
        local_pos = self.mapFromScene(QPointF(canvas_x, canvas_y))
        r = self._radius

        b_minus_rect = QRectF(-r - 60.0, r + 45.0, 60.0, 26.0)
        b_plus_rect = QRectF(-r + 10.0, r + 45.0, 60.0, 26.0)
        b_snap_rect = QRectF(-r + 80.0, r + 45.0, 60.0, 26.0)

        if b_minus_rect.contains(local_pos):
            self.rotate_by(-15.0)
            return True
        elif b_plus_rect.contains(local_pos):
            self.rotate_by(15.0)
            return True
        elif b_snap_rect.contains(local_pos):
            self.snap_to_cardinal_angle(tolerance_deg=180.0)
            return True

        # Air click directly on unit circle perimeter sets angle
        dist_from_origin = math.hypot(local_pos.x(), local_pos.y())
        if abs(dist_from_origin - r) <= 32.0:
            ang_rad = math.atan2(-local_pos.y(), local_pos.x())
            ang_deg = math.degrees(ang_rad) % 360.0
            self.set_angle_deg(ang_deg)
            self.snap_to_cardinal_angle(tolerance_deg=4.0)
            return True

        return False
