"""
AirTeach AI - Interactive Whiteboard Measurement Tools
======================================================
Provides interactive, semi-transparent touchless measurement overlays on LessonBoard:
  - VirtualRulerItem: Calibrated pixel/cm ruler with mm graduation marks and rotation handle.
  - VirtualProtractorItem: 180° acrylic protractor with dual-direction degree markings and angle pointer.
"""
from __future__ import annotations

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
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)


class VirtualRulerItem(QGraphicsObject):
    """
    Interactive Whiteboard Ruler overlay for LessonBoard.
    Displays millimeter and centimeter graduations, length readout,
    and supports moving, rotating, and drawing along the guide edge.
    """

    def __init__(
        self,
        length_cm: float = 15.0,
        pixels_per_cm: float = 40.0,
        height_px: float = 75.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.length_cm = max(5.0, length_cm)
        self.pixels_per_cm = max(10.0, pixels_per_cm)
        self.height_px = max(40.0, height_px)
        self.length_px = self.length_cm * self.pixels_per_cm

        # Styling
        self._bg_color = QColor(245, 248, 255, 205)  # Translucent acrylic
        self._border_color = QColor(26, 115, 232, 220)  # Primary blue
        self._tick_color = QColor(32, 33, 36, 210)
        self._accent_color = QColor(234, 67, 53, 230)  # Red for zero & 5/10 marks

        # Interaction flags
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Internal interaction state
        self._is_rotating: bool = False
        self._rot_start_pos: Optional[QPointF] = None
        self._start_angle: float = 0.0

    def boundingRect(self) -> QRectF:
        margin = 15.0
        return QRectF(-margin, -margin, self.length_px + 2 * margin, self.height_px + 2 * margin)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Ruler Acrylic Body
        rect = QRectF(0.0, 0.0, self.length_px, self.height_px)
        painter.setBrush(QBrush(self._bg_color))
        border_pen = QPen(self._border_color, 2.0)
        if self.isSelected():
            border_pen = QPen(QColor("#1a73e8"), 2.5, Qt.PenStyle.DashLine)
        painter.setPen(border_pen)
        painter.drawRoundedRect(rect, 8.0, 8.0)

        # 2. Measurement Edge (Top line)
        painter.setPen(QPen(self._border_color, 2.5))
        painter.drawLine(0.0, 0.0, self.length_px, 0.0)

        # 3. Graduation Ticks & Labels along Top Edge
        mm_px = self.pixels_per_cm / 10.0
        total_mm = int(self.length_cm * 10)

        font_num = QFont("Segoe UI", 9, QFont.Weight.Bold)
        font_sub = QFont("Segoe UI", 7, QFont.Weight.Normal)

        for mm in range(total_mm + 1):
            x = mm * mm_px
            if mm % 10 == 0:
                # 1 cm tick (tallest)
                cm_val = mm // 10
                tick_h = 24.0
                painter.setPen(QPen(self._accent_color if cm_val == 0 else self._tick_color, 1.8))
                painter.drawLine(x, 0.0, x, tick_h)

                # Numeral label
                painter.setFont(font_num)
                painter.setPen(QPen(self._tick_color))
                lbl = str(cm_val)
                painter.drawText(QRectF(x - 12.0, tick_h + 2.0, 24.0, 16.0), Qt.AlignmentFlag.AlignCenter, lbl)
            elif mm % 5 == 0:
                # 0.5 cm tick (medium)
                tick_h = 16.0
                painter.setPen(QPen(self._tick_color, 1.4))
                painter.drawLine(x, 0.0, x, tick_h)
            else:
                # 1 mm tick (short)
                tick_h = 9.0
                painter.setPen(QPen(self._tick_color, 0.9))
                painter.drawLine(x, 0.0, x, tick_h)

        # 4. Length & Angle Telemetry Badge on Bottom Left
        curr_angle = (self.rotation()) % 360.0
        telemetry_text = f"📏 {self.length_cm:.0f} cm ({self.length_px:.0f} px) · {curr_angle:.1f}°"
        painter.setFont(font_sub)
        painter.setPen(QPen(QColor("#5f6368")))
        painter.drawText(QRectF(14.0, self.height_px - 22.0, 200.0, 16.0), Qt.AlignmentFlag.AlignLeft, telemetry_text)

        # 5. Rotation Handle on Right End
        rot_cx = self.length_px - 20.0
        rot_cy = self.height_px / 2.0
        painter.setBrush(QBrush(QColor("#1a73e8")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(QPointF(rot_cx, rot_cy), 12.0, 12.0)

        # Rotate symbol ↻
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(QRectF(rot_cx - 10.0, rot_cy - 10.0, 20.0, 20.0), Qt.AlignmentFlag.AlignCenter, "↻")

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        rot_cx = self.length_px - 20.0
        rot_cy = self.height_px / 2.0
        dist = math.hypot(pos.x() - rot_cx, pos.y() - rot_cy)

        if dist <= 16.0:
            # Clicked rotation handle
            self._is_rotating = True
            self._rot_start_pos = event.scenePos()
            self._start_angle = self.rotation()
            event.accept()
        else:
            self._is_rotating = False
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_rotating and self._rot_start_pos is not None:
            scene_pos = event.scenePos()
            center_scene = self.mapToScene(QPointF(self.length_px / 2.0, self.height_px / 2.0))
            v_curr = scene_pos - center_scene
            angle_rad = math.atan2(v_curr.y(), v_curr.x())
            self.setRotation(math.degrees(angle_rad))
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._is_rotating = False
        self._rot_start_pos = None
        super().mouseReleaseEvent(event)

    def set_angle(self, angle_deg: float) -> None:
        """Programmatically set ruler orientation angle."""
        self.setRotation(angle_deg % 360.0)
        self.update()

    def rotate_by(self, delta_deg: float) -> float:
        """Incrementally rotate ruler by delta degrees."""
        new_angle = (self.rotation() + delta_deg) % 360.0
        self.setRotation(new_angle)
        self.update()
        return new_angle

    def snap_to_cardinal_angle(self, tolerance_deg: float = 7.5) -> float:
        """Snap current ruler angle to nearest cardinal angle (0°, 30°, 45°, 60°, 90°, etc.)."""
        cardinals = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0, 210.0, 225.0, 240.0, 270.0, 300.0, 315.0, 330.0, 360.0]
        curr = self.rotation() % 360.0
        for c in cardinals:
            if abs(curr - c) <= tolerance_deg or abs(curr - c + 360.0) <= tolerance_deg:
                target = 0.0 if c == 360.0 else c
                self.setRotation(target)
                self.update()
                return target
        return curr

    def set_length_cm(self, length_cm: float) -> None:
        """Dynamically change ruler measurement length (5cm to 50cm)."""
        self.prepareGeometryChange()
        self.length_cm = max(5.0, min(50.0, float(length_cm)))
        self.length_px = self.length_cm * self.pixels_per_cm
        self.update()

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """Touchless air tap on ruler controls (e.g. rotation handle)."""
        local_pos = self.mapFromScene(QPointF(canvas_x, canvas_y))
        rot_cx = self.length_px - 20.0
        rot_cy = self.height_px / 2.0
        dist = math.hypot(local_pos.x() - rot_cx, local_pos.y() - rot_cy)
        if dist <= 24.0:
            # Air click on rotation handle rotates by 45°
            self.rotate_by(45.0)
            return True
        return False


class VirtualProtractorItem(QGraphicsObject):
    """
    Interactive Whiteboard Protractor overlay for LessonBoard.
    Displays a 180° semi-circular transparent acrylic disk with degree
    graduations (1°, 5°, 10°), dual-direction degree numerals (0°-180°),
    origin crosshairs, and a rotatable angle measurement needle.
    """

    def __init__(
        self,
        radius: float = 220.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.radius = max(80.0, radius)
        self.needle_angle: float = 45.0  # Measured angle indicator (0° to 180°)

        # Styling
        self._bg_color = QColor(245, 248, 255, 195)
        self._border_color = QColor(26, 115, 232, 220)
        self._tick_color = QColor(32, 33, 36, 210)
        self._needle_color = QColor(234, 67, 53, 230)

        # Interaction flags
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Needle dragging state
        self._is_dragging_needle: bool = False

    def boundingRect(self) -> QRectF:
        r = self.radius + 20.0
        return QRectF(-r, -r, 2 * r, r + 40.0)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        r = self.radius

        # 1. Semicircular Acrylic Disk Body (upper half from 180° to 0°)
        path = QPainterPath()
        path.moveTo(-r, 0.0)
        path.arcTo(QRectF(-r, -r, 2 * r, 2 * r), 180.0, -180.0)
        path.lineTo(r, 0.0)
        path.lineTo(-r, 0.0)

        painter.setBrush(QBrush(self._bg_color))
        border_pen = QPen(self._border_color, 2.0)
        if self.isSelected():
            border_pen = QPen(QColor("#1a73e8"), 2.5, Qt.PenStyle.DashLine)
        painter.setPen(border_pen)
        painter.drawPath(path)

        # 2. Origin Center Dot and Crosshairs (0, 0)
        painter.setPen(QPen(self._border_color, 1.8))
        painter.drawLine(-35.0, 0.0, 35.0, 0.0)
        painter.drawLine(0.0, -35.0, 0.0, 0.0)
        painter.setBrush(QBrush(self._border_color))
        painter.drawEllipse(QPointF(0.0, 0.0), 3.5, 3.5)

        # 3. Degree Graduation Ticks & Labels
        font_major = QFont("Segoe UI", 8, QFont.Weight.Bold)
        font_sub = QFont("Segoe UI", 6, QFont.Weight.Normal)

        for deg in range(181):
            rad = math.radians(deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            if deg % 10 == 0:
                # 10 deg major tick
                t_len = 16.0
                painter.setPen(QPen(self._tick_color, 1.6))
                p_out = QPointF(r * cos_a, -r * sin_a)
                p_in = QPointF((r - t_len) * cos_a, -(r - t_len) * sin_a)
                painter.drawLine(p_out, p_in)

                # Outer label (counter-clockwise: 0° on right, 180° on left)
                painter.setFont(font_major)
                painter.setPen(QPen(self._tick_color))
                p_lbl = QPointF((r - 28.0) * cos_a, -(r - 28.0) * sin_a)
                painter.drawText(QRectF(p_lbl.x() - 12.0, p_lbl.y() - 8.0, 24.0, 16.0), Qt.AlignmentFlag.AlignCenter, str(deg))

                # Inner label (clockwise: 180° on right, 0° on left)
                p_lbl_inner = QPointF((r - 46.0) * cos_a, -(r - 46.0) * sin_a)
                painter.setFont(font_sub)
                painter.setPen(QPen(QColor("#5f6368")))
                painter.drawText(QRectF(p_lbl_inner.x() - 10.0, p_lbl_inner.y() - 6.0, 20.0, 12.0), Qt.AlignmentFlag.AlignCenter, str(180 - deg))

            elif deg % 5 == 0:
                # 5 deg medium tick
                t_len = 10.0
                painter.setPen(QPen(self._tick_color, 1.2))
                p_out = QPointF(r * cos_a, -r * sin_a)
                p_in = QPointF((r - t_len) * cos_a, -(r - t_len) * sin_a)
                painter.drawLine(p_out, p_in)
            else:
                # 1 deg minor tick
                t_len = 5.0
                painter.setPen(QPen(self._tick_color, 0.8))
                p_out = QPointF(r * cos_a, -r * sin_a)
                p_in = QPointF((r - t_len) * cos_a, -(r - t_len) * sin_a)
                painter.drawLine(p_out, p_in)

        # 4. Rotatable Angle Measurement Needle
        n_rad = math.radians(self.needle_angle)
        needle_x = (r - 2.0) * math.cos(n_rad)
        needle_y = -(r - 2.0) * math.sin(n_rad)

        painter.setPen(QPen(self._needle_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(0.0, 0.0, needle_x, needle_y)

        # Needle Tip Arrow Handle
        tip_r = 7.0
        painter.setBrush(QBrush(self._needle_color))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(QPointF(needle_x, needle_y), tip_r, tip_r)

        # 5. Angle Measurement Telemetry Badge at Bottom
        badge_text = f"🧭 Góc: {self.needle_angle:.1f}°"
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(self._needle_color))
        painter.drawText(QRectF(-60.0, 14.0, 120.0, 20.0), Qt.AlignmentFlag.AlignCenter, badge_text)

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        n_rad = math.radians(self.needle_angle)
        needle_tip = QPointF((self.radius - 2.0) * math.cos(n_rad), -(self.radius - 2.0) * math.sin(n_rad))
        dist_to_tip = math.hypot(pos.x() - needle_tip.x(), pos.y() - needle_tip.y())

        if dist_to_tip <= 20.0 or (pos.y() <= 5.0 and math.hypot(pos.x(), pos.y()) <= self.radius + 10.0):
            self._is_dragging_needle = True
            self._update_needle_from_pos(pos)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_dragging_needle:
            self._update_needle_from_pos(event.pos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._is_dragging_needle = False
        super().mouseReleaseEvent(event)

    def _update_needle_from_pos(self, pos: QPointF) -> None:
        """Calculate and clamp angle from origin (0, 0) to pos within [0°, 180°]."""
        angle_rad = math.atan2(-pos.y(), pos.x())
        angle_deg = math.degrees(angle_rad)
        self.needle_angle = max(0.0, min(180.0, angle_deg))
        self.update()

    def set_angle(self, angle_deg: float) -> None:
        """Programmatically set needle measurement angle."""
        self.needle_angle = max(0.0, min(180.0, float(angle_deg)))
        self.update()

    def adjust_needle_delta(self, delta_deg: float) -> float:
        """Adjust needle angle incrementally within [0°, 180°]."""
        self.needle_angle = max(0.0, min(180.0, self.needle_angle + delta_deg))
        self.update()
        return self.needle_angle

    def snap_needle_to_cardinal_angle(self, tolerance_deg: float = 3.0) -> float:
        """Snap needle to nearest cardinal angle (0°, 30°, 45°, 60°, 90°, 120°, 135°, 150°, 180°)."""
        cardinals = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0]
        for c in cardinals:
            if abs(self.needle_angle - c) <= tolerance_deg:
                self.needle_angle = c
                self.update()
                return c
        return self.needle_angle

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """Touchless air tap on protractor measurement disk to set needle angle directly."""
        local_pos = self.mapFromScene(QPointF(canvas_x, canvas_y))
        dist = math.hypot(local_pos.x(), local_pos.y())
        if 20.0 <= dist <= self.radius + 20.0 and local_pos.y() <= 10.0:
            self._update_needle_from_pos(local_pos)
            return True
        return False


class VirtualCompassItem(QGraphicsObject):
    """
    Interactive Whiteboard Compass (Compa) overlay for LessonBoard.
    Allows precision circle and arc construction in mathematics/geometry teaching:
      - Fixed needle point at pivot center (0, 0).
      - Adjustable radius arm (1.0 cm - 12.0 cm calibrated at 40 px/cm).
      - Rotatable pencil leg (0° - 360°) with dynamic angle readout.
      - Faint guide preview circle.
      - 'Draw Full Circle' stamp action to place crisp vector circles onto the board.
      - Mouse dragging & touchless air click support.
    """

    circle_drawn = Signal(float, float, float)  # (cx, cy, radius_px)
    radius_changed = Signal(float)              # (radius_cm)
    angle_changed = Signal(float)               # (pencil_angle_deg)

    def __init__(
        self,
        radius_cm: float = 3.5,
        pixels_per_cm: float = 40.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.pixels_per_cm: float = max(10.0, pixels_per_cm)
        self._radius_cm: float = max(1.0, min(12.0, float(radius_cm)))
        self._pencil_angle: float = 45.0  # degrees counter-clockwise from horizontal right
        self._pen_color: QColor = QColor("#1e293b")
        self._pen_width: float = 2.5

        # Colors & Styling
        self._needle_color = QColor(71, 85, 105, 230)
        self._arm_color = QColor(241, 245, 249, 215)
        self._arm_border = QColor(59, 130, 246, 220)
        self._pencil_color = QColor(239, 68, 68, 230)
        self._badge_bg = QColor(22, 24, 34, 230)

        # Interaction flags
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Interaction state
        self._is_dragging_pencil: bool = False

    # ── Properties & Mutators ──────────────────────────────────────────────────

    def radius_cm(self) -> float:
        return self._radius_cm

    def set_radius_cm(self, r_cm: float) -> None:
        clamped = max(1.0, min(12.0, float(r_cm)))
        if abs(clamped - self._radius_cm) > 1e-4:
            self.prepareGeometryChange()
            self._radius_cm = clamped
            self.radius_changed.emit(self._radius_cm)
            self.update()

    def radius_px(self) -> float:
        return self._radius_cm * self.pixels_per_cm

    def pencil_angle(self) -> float:
        return self._pencil_angle

    def set_angle(self, angle_deg: float) -> None:
        norm_angle = float(angle_deg) % 360.0
        if norm_angle < 0.0:
            norm_angle += 360.0
        if abs(norm_angle - self._pencil_angle) > 1e-4:
            self.prepareGeometryChange()
            self._pencil_angle = norm_angle
            self.angle_changed.emit(self._pencil_angle)
            self.update()

    def get_pencil_pos(self) -> QPointF:
        """Calculate local pencil position from radius and angle."""
        rad = math.radians(self._pencil_angle)
        r = self.radius_px()
        return QPointF(r * math.cos(rad), -r * math.sin(rad))

    # ── Drawing Operations ─────────────────────────────────────────────────────

    def draw_full_circle(self) -> Optional[QGraphicsEllipseItem]:
        """Stamps a crisp vector circle onto the parent scene at active compass position."""
        scene = self.scene()
        if scene is None:
            return None

        # Center in scene coordinates
        center = self.scenePos()
        cx, cy = center.x(), center.y()
        r = self.radius_px()

        circle_item = QGraphicsEllipseItem(cx - r, cy - r, 2.0 * r, 2.0 * r)
        circle_item.setPen(QPen(self._pen_color, self._pen_width))
        circle_item.setBrush(Qt.BrushStyle.NoBrush)
        circle_item.setZValue(5.0)
        scene.addItem(circle_item)

        # Trigger acoustic snap feedback
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_snap()
        except Exception:
            pass

        self.circle_drawn.emit(cx, cy, r)
        return circle_item

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        r = self.radius_px()
        margin = 35.0
        # Include top badge extending from y = -65 to -30
        return QRectF(-r - margin, -r - margin - 50.0, 2 * (r + margin), 2 * (r + margin) + 50.0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        r = self.radius_px()
        pencil_pt = self.get_pencil_pos()

        # 1. Faint dashed preview guide circle
        guide_pen = QPen(QColor(59, 130, 246, 80), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(guide_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0.0, 0.0), r, r)

        # 2. Compass Arm (from (0, 0) to pencil_pt)
        painter.save()
        rad = math.radians(self._pencil_angle)
        deg = self._pencil_angle

        # Draw connecting acrylic beam with rotation
        painter.rotate(-deg)
        arm_rect = QRectF(0.0, -8.0, r, 16.0)
        arm_path = QPainterPath()
        arm_path.addRoundedRect(arm_rect, 4.0, 4.0)
        painter.fillPath(arm_path, QBrush(self._arm_color))
        painter.strokePath(arm_path, QPen(self._arm_border, 1.5))

        # Graduation ticks along arm
        mm_px = self.pixels_per_cm / 10.0
        total_mm = int(self._radius_cm * 10)
        for mm in range(total_mm + 1):
            mx = mm * mm_px
            if mm % 10 == 0:
                painter.setPen(QPen(QColor("#1e293b"), 1.2))
                painter.drawLine(mx, -8.0, mx, -1.0)
                if mm > 0 and mm % 20 == 0:
                    painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                    painter.drawText(QRectF(mx - 8.0, -1.0, 16.0, 8.0), Qt.AlignmentFlag.AlignCenter, str(mm // 10))
            elif mm % 5 == 0:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.drawLine(mx, -8.0, mx, -3.5)
            else:
                painter.setPen(QPen(QColor("#94a3b8"), 0.7))
                painter.drawLine(mx, -8.0, mx, -5.5)

        painter.restore()

        # 3. Center Needle Point Pivot at (0, 0)
        painter.setPen(QPen(self._needle_color, 1.8))
        painter.drawLine(-12.0, 0.0, 12.0, 0.0)
        painter.drawLine(0.0, -12.0, 0.0, 12.0)

        # Chrome pivot ring
        painter.setBrush(QBrush(QColor(226, 232, 240, 240)))
        painter.setPen(QPen(QColor(71, 85, 105), 1.5))
        painter.drawEllipse(QPointF(0.0, 0.0), 6.5, 6.5)

        # Center needle dot
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0.0, 0.0), 2.5, 2.5)

        # 4. Pencil Lead Leg at pencil_pt
        # Outer drag handle ring
        painter.setBrush(QBrush(QColor(239, 68, 68, 60)))
        painter.setPen(QPen(QColor("#ef4444"), 2.0))
        painter.drawEllipse(pencil_pt, 12.0, 12.0)

        # Lead holder clamp
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(pencil_pt, 5.0, 5.0)

        # Lead point dot
        painter.setBrush(QBrush(QColor("#ef4444")))
        painter.drawEllipse(pencil_pt, 2.0, 2.0)

        # 5. Floating Control Badge / Telemetry Pill (centered above pivot)
        badge_w = 260.0
        badge_h = 32.0
        badge_x = -badge_w / 2.0
        badge_y = -r - 42.0 if r > 40.0 else -80.0

        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_x, badge_y, badge_w, badge_h, 8.0, 8.0)
        painter.fillPath(badge_path, QBrush(self._badge_bg))
        painter.strokePath(badge_path, QPen(QColor(255, 255, 255, 35), 1.0))

        # Title & Telemetry text
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        info_txt = f"📐 R={self._radius_cm:.1f}cm | {self._pencil_angle:.0f}°"
        painter.drawText(badge_x + 8.0, badge_y + 20.0, info_txt)

        # Button [🖊️ Vẽ]
        btn_draw_rect = QRectF(badge_x + 130.0, badge_y + 4.0, 42.0, 24.0)
        painter.setBrush(QBrush(QColor(16, 185, 129, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(btn_draw_rect, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(btn_draw_rect, Qt.AlignmentFlag.AlignCenter, "🖊️ Vẽ")

        # Button [−]
        btn_min_rect = QRectF(badge_x + 176.0, badge_y + 4.0, 20.0, 24.0)
        painter.setBrush(QBrush(QColor(255, 255, 255, 25)))
        painter.drawRoundedRect(btn_min_rect, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(btn_min_rect, Qt.AlignmentFlag.AlignCenter, "−")

        # Button [+]
        btn_plus_rect = QRectF(badge_x + 200.0, badge_y + 4.0, 20.0, 24.0)
        painter.setBrush(QBrush(QColor(255, 255, 255, 25)))
        painter.drawRoundedRect(btn_plus_rect, 4.0, 4.0)
        painter.drawText(btn_plus_rect, Qt.AlignmentFlag.AlignCenter, "+")

        # Close button [✕]
        btn_close_rect = QRectF(badge_x + 228.0, badge_y + 4.0, 24.0, 24.0)
        painter.setBrush(QBrush(QColor(239, 68, 68, 180)))
        painter.drawRoundedRect(btn_close_rect, 4.0, 4.0)
        painter.drawText(btn_close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

    # ── Mouse & Air Gesture Interaction ────────────────────────────────────────

    def _get_badge_y(self) -> float:
        r = self.radius_px()
        return -r - 42.0 if r > 40.0 else -80.0

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        px, py = pos.x(), pos.y()
        badge_w = 260.0
        badge_x = -badge_w / 2.0
        badge_y = self._get_badge_y()

        # 1. Close button [✕]
        if badge_x + 228.0 <= px <= badge_x + 252.0 and badge_y <= py <= badge_y + 32.0:
            self.setVisible(False)
            event.accept()
            return

        # 2. Draw Circle button [🖊️ Vẽ]
        if badge_x + 130.0 <= px <= badge_x + 172.0 and badge_y <= py <= badge_y + 32.0:
            self.draw_full_circle()
            event.accept()
            return

        # 3. Radius Minus [−]
        if badge_x + 176.0 <= px <= badge_x + 196.0 and badge_y <= py <= badge_y + 32.0:
            self.set_radius_cm(self._radius_cm - 0.5)
            event.accept()
            return

        # 4. Radius Plus [+]
        if badge_x + 200.0 <= px <= badge_x + 224.0 and badge_y <= py <= badge_y + 32.0:
            self.set_radius_cm(self._radius_cm + 0.5)
            event.accept()
            return

        # 5. Pencil Tip handle dragging (to rotate)
        pencil_pt = self.get_pencil_pos()
        dist_to_pencil = math.hypot(px - pencil_pt.x(), py - pencil_pt.y())
        if dist_to_pencil <= 22.0:
            self._is_dragging_pencil = True
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_dragging_pencil:
            pos = event.pos()
            # Calculate angle from (0, 0) to pos
            angle_rad = math.atan2(-pos.y(), pos.x())
            angle_deg = math.degrees(angle_rad)
            self.set_angle(angle_deg)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._is_dragging_pencil:
            self._is_dragging_pencil = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """Allows touchless air tapping on compass controls."""
        local_pos = self.mapFromScene(QPointF(canvas_x, canvas_y))
        px, py = local_pos.x(), local_pos.y()
        badge_w = 260.0
        badge_x = -badge_w / 2.0
        badge_y = self._get_badge_y()

        # 1. Close button
        if badge_x + 224.0 <= px <= badge_x + 256.0 and badge_y - 4.0 <= py <= badge_y + 36.0:
            self.setVisible(False)
            return True

        # 2. Draw Circle button
        if badge_x + 126.0 <= px <= badge_x + 174.0 and badge_y - 4.0 <= py <= badge_y + 36.0:
            self.draw_full_circle()
            return True

        # 3. Radius Minus
        if badge_x + 172.0 <= px <= badge_x + 198.0 and badge_y - 4.0 <= py <= badge_y + 36.0:
            self.set_radius_cm(self._radius_cm - 0.5)
            return True

        # 4. Radius Plus
        if badge_x + 198.0 <= px <= badge_x + 226.0 and badge_y - 4.0 <= py <= badge_y + 36.0:
            self.set_radius_cm(self._radius_cm + 0.5)
            return True

        # 5. Pencil handle tap
        pencil_pt = self.get_pencil_pos()
        if math.hypot(px - pencil_pt.x(), py - pencil_pt.y()) <= 25.0:
            # Advance angle by 30 degrees on tap
            self.set_angle(self._pencil_angle + 30.0)
            return True

        return False
