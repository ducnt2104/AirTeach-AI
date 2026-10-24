"""
AirTeach AI - Interactive Geometric Optics Thin Lens & Optical Bench Simulator
=================================================================================
Provides an authentic physics simulation of geometric optics and thin lens ray tracing:
  - Thin Lens & Mirror Gaussian Equations:
      1/f = 1/d + 1/d' => d' = (d * f) / (d - f)
  - Lateral Magnification:
      k = -d' / d = A'B' / AB = -f / (d - f)
  - Optical Conventions:
      - Converging Thin Lens (Thấu kính hội tụ): f > 0
      - Diverging Thin Lens (Thấu kính phân kỳ): f < 0
      - Real Object: d > 0
      - Real Image (Ảnh thật, ngược chiều): d' > 0, k < 0
      - Virtual Image (Ảnh ảo, cùng chiều): d' < 0, k > 0
      - Image at Infinity: d = f => d' -> ∞
  - 3 Authentic Principal Ray Tracings:
      1. Parallel-to-axis ray (Tia 1 - Gold): Parallel to optical axis from B to lens plane,
         refracts through primary focus F' (or appears to diverge from F').
      2. Central ray (Tia 2 - Orange): Straight through optical center O without deviation.
      3. Focal ray (Tia 3 - Cyan): Passes through (or towards) focus F, emerges parallel to axis.
      - Dashed extensions backward for virtual images.
  - Interactive Visual Elements:
      - Draggable Object Arrow AB (air pinch on tip B sets position d and height h).
      - Draggable Focal Point F' (adjusts focal length f).
      - Movable observation screen line with focus detection.
      - Dual lens modes: Converging (biconvex) & Diverging (biconcave).
      - Curriculum presets:
          1. MAGNIFYING_GLASS (Kính lúp: d < f => virtual upright magnified image)
          2. PROJECTOR_LENS (Máy chiếu: f < d < 2f => real inverted magnified image)
          3. CAMERA_OBJECTIVE (Máy ảnh: d > 2f => real inverted diminished image)
          4. DIVERGING_EYEGLASS (Kính cận: f < 0 => virtual upright diminished image)
      - Stamping to LessonBoard via PhysicsTools.create_thin_lens_diagram.
"""
from __future__ import annotations

import logging
import math
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

from src.ui.stem_tools import PhysicsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


class InteractiveThinLensBenchItem(QGraphicsObject):
    """
    Interactive Geometric Optics Thin Lens & Optical Bench Simulator for QGraphicsScene.
    """

    optics_updated = Signal(float, float, float, float)  # (d_cm, d_prime_cm, f_cm, magnification_k)
    preset_changed = Signal(str)                          # ("MAGNIFYING_GLASS", etc.)
    mode_changed = Signal(str)                            # ("CONVERGING", "DIVERGING")

    def __init__(
        self,
        pos: tuple[float, float] = (300.0, 160.0),
        preset: str = "PROJECTOR_LENS",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(38.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Dimensions
        self._width: float = 840.0
        self._height: float = 560.0

        # Physical optics parameters (in cm; 1 cm = 8.0 pixels on bench)
        self._scale: float = 8.0            # 8 px per cm
        self._focal_length_cm: float = 12.0 # Focal length f [cm] (> 0 converging, < 0 diverging)
        self._object_d_cm: float = 18.0     # Object distance d [cm]
        self._object_h_cm: float = 6.0      # Object height h [cm]
        self._lens_type: str = "CONVERGING" # "CONVERGING" or "DIVERGING"
        self._preset: str = preset

        # Visual layout
        self._canvas_rect = QRectF(15.0, 45.0, 500.0, 500.0)
        self._optic_center = QPointF(265.0, 275.0)  # Optical center O on canvas

        # Interactive toggles
        self._show_principal_rays: bool = True
        self._show_screen: bool = True
        self._screen_x_cm: float = 36.0     # Position of screen on bench

        # Drag state
        self._drag_target: Optional[str] = None  # "TITLE", "OBJECT_TIP", "FOCUS_HANDLE", "SCREEN"
        self._drag_offset: QPointF = QPointF(0.0, 0.0)
        self._hover_target: Optional[str] = None

        # Apply initial preset
        self._apply_preset(self._preset)

    # ── Physical Optics Calculations ──────────────────────────────────────────

    @property
    def image_d_prime_cm(self) -> float:
        """
        Calculates image distance d' [cm] using 1/f = 1/d + 1/d'.
        Returns large finite value if image is at infinity (d = f).
        """
        f = self._focal_length_cm
        d = self._object_d_cm
        if abs(d - f) < 0.01:
            return 9999.0 if (d - f) >= 0 else -9999.0
        return (d * f) / (d - f)

    @property
    def magnification_k(self) -> float:
        """Calculates lateral magnification k = -d' / d = -f / (d - f)."""
        f = self._focal_length_cm
        d = self._object_d_cm
        if abs(d - f) < 0.01:
            return 999.0
        return -f / (d - f)

    @property
    def image_h_prime_cm(self) -> float:
        """Calculates signed image height A'B' [cm] = k * AB."""
        return self.magnification_k * self._object_h_cm

    # ── Bounding Rect & Painting ──────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Outer Container
        self._paint_outer_container(painter)

        # 2. Left Optical Bench (Axis, Lens, Object, Image, Principal Rays)
        self._paint_optical_bench(painter)

        # 3. Right Control & Telemetry HUD
        self._paint_controls_and_hud(painter)

        painter.restore()

    def _paint_outer_container(self, painter: QPainter) -> None:
        outer = self.boundingRect()

        # Slate-900 background
        painter.setPen(QPen(QColor("#334155"), 1.8))
        painter.setBrush(QBrush(QColor(15, 23, 42, 245)))
        painter.drawRoundedRect(outer, 14.0, 14.0)

        # Title bar
        title_rect = QRectF(0.0, 0.0, self._width, 40.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59, 230)))
        painter.drawRoundedRect(title_rect, 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 26.0, self._width, 14.0))

        # Title Text
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(
            QRectF(18.0, 8.0, 520.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "🔬 QUANG HÌNH HỌC: BĂNG QUANG HỌC & THẤU KÍNH MỎNG",
        )

        # Preset badge
        preset_labels = {
            "MAGNIFYING_GLASS": "🔍 Kính Lúp (Ảnh Ảo Phóng Đại)",
            "PROJECTOR_LENS": "📽️ Máy Chiếu (Ảnh Thật Phóng Đại)",
            "CAMERA_OBJECTIVE": "📷 Máy Ảnh (Ảnh Thật Thu Nhỏ)",
            "DIVERGING_EYEGLASS": "👓 Kính Cận (Thấu Kính Phân Kỳ)",
        }
        badge = preset_labels.get(self._preset, "Thấu Kính Mỏng")
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(520.0, 8.0, 260.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            badge,
        )

        # Close button
        c_btn = self._get_close_button_rect()
        h_close = (self._hover_target == "CLOSE")
        painter.setBrush(QBrush(QColor("#dc2626") if h_close else QColor(239, 68, 68, 160)))
        painter.setPen(QPen(QColor("#fca5a5"), 1.2))
        painter.drawRoundedRect(c_btn, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(c_btn, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_optical_bench(self, painter: QPainter) -> None:
        """Paints optical axis, thin lens, focal points, object AB, image A'B', and 3 principal rays."""
        painter.save()
        painter.setClipRect(self._canvas_rect)

        # Deep cosmic viewport
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.setBrush(QBrush(QColor("#030712")))
        painter.drawRoundedRect(self._canvas_rect, 8.0, 8.0)

        ox = self._optic_center.x()
        oy = self._optic_center.y()
        scale = self._scale

        # ── 1. Principal Optical Axis ──
        painter.setPen(QPen(QColor("#64748b"), 1.5))
        painter.drawLine(QPointF(self._canvas_rect.left() + 5.0, oy), QPointF(self._canvas_rect.right() - 5.0, oy))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QPointF(self._canvas_rect.right() - 25.0, oy - 6.0), "Δ (Trục chính)")
        painter.drawText(QPointF(ox - 6.0, oy + 16.0), "O")

        # ── 2. Thin Lens Plane & Symbol ──
        lens_h = 320.0
        painter.setPen(QPen(QColor("#38bdf8"), 2.2))
        painter.drawLine(QPointF(ox, oy - lens_h / 2.0), QPointF(ox, oy + lens_h / 2.0))

        # Lens arrowheads (converging points outward, diverging points inward)
        if self._lens_type == "CONVERGING":
            # Outward arrowheads at top and bottom
            self._paint_arrowhead(painter, QPointF(ox, oy - lens_h / 2.0), -math.pi / 2.0, QColor("#38bdf8"))
            self._paint_arrowhead(painter, QPointF(ox, oy + lens_h / 2.0), math.pi / 2.0, QColor("#38bdf8"))
            lbl_type = "TK Hội Tụ (f > 0)"
        else:
            # Inward arrowheads
            self._paint_arrowhead(painter, QPointF(ox, oy - lens_h / 2.0), math.pi / 2.0, QColor("#38bdf8"))
            self._paint_arrowhead(painter, QPointF(ox, oy + lens_h / 2.0), -math.pi / 2.0, QColor("#38bdf8"))
            lbl_type = "TK Phân Kỳ (f < 0)"

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(ox - 45.0, oy - lens_h / 2.0 - 6.0), lbl_type)

        # ── 3. Focal Points F and F' ──
        f_px = abs(self._focal_length_cm) * scale
        if self._lens_type == "CONVERGING":
            # Converging: F in front (left), F' behind (right)
            fx = ox - f_px
            fpx = ox + f_px
            f_lbl, fp_lbl = "F", "F'"
        else:
            # Diverging: F' in front (left), F behind (right)
            fpx = ox - f_px
            fx = ox + f_px
            f_lbl, fp_lbl = "F'", "F"

        # Dots for F and F'
        for dot_x, dot_lbl in [(fx, f_lbl), (fpx, fp_lbl)]:
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.setBrush(QBrush(QColor("#fbbf24")))
            painter.drawEllipse(QPointF(dot_x, oy), 4.5, 4.5)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#fbbf24")))
            painter.drawText(QPointF(dot_x - 5.0, oy + 16.0), dot_lbl)

        # ── 4. Object Arrow AB ──
        obj_x = ox - self._object_d_cm * scale
        obj_y_tip = oy - self._object_h_cm * scale

        # Object arrow line in Amber
        painter.setPen(QPen(QColor("#f59e0b"), 2.6))
        painter.drawLine(QPointF(obj_x, oy), QPointF(obj_x, obj_y_tip))
        self._paint_arrowhead(painter, QPointF(obj_x, obj_y_tip), -math.pi / 2.0, QColor("#f59e0b"))

        # Object Labels A and B
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(QPointF(obj_x - 14.0, oy + 14.0), "A")
        painter.drawText(QPointF(obj_x - 14.0, obj_y_tip - 4.0), "B")

        # Draggable tip handle bead on B
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.drawEllipse(QPointF(obj_x, obj_y_tip), 6.0, 6.0)

        # ── 5. Image Arrow A'B' ──
        d_p = self.image_d_prime_cm
        is_real = (d_p > 0.0) and (abs(d_p) < 500.0)
        is_virtual = (d_p < 0.0) and (abs(d_p) < 500.0)

        if is_real or is_virtual:
            img_x = ox + d_p * scale
            img_h_px = self.image_h_prime_cm * scale
            img_y_tip = oy - img_h_px

            img_color = QColor("#10b981") if is_real else QColor("#ec4899")
            pen_img = QPen(img_color, 2.4, Qt.PenStyle.SolidLine if is_real else Qt.PenStyle.DashLine)
            painter.setPen(pen_img)
            painter.drawLine(QPointF(img_x, oy), QPointF(img_x, img_y_tip))

            # Image arrowhead
            angle_img = -math.pi / 2.0 if img_h_px > 0 else math.pi / 2.0
            self._paint_arrowhead(painter, QPointF(img_x, img_y_tip), angle_img, img_color)

            # Image Labels A' and B'
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(img_color))
            painter.drawText(QPointF(img_x + 4.0, oy + 14.0), "A'")
            painter.drawText(QPointF(img_x + 4.0, img_y_tip - 4.0), "B'")

        # ── 6. Principal Ray Tracing (3 Authentic Rays) ──
        if self._show_principal_rays and abs(self._object_d_cm - abs(self._focal_length_cm)) > 0.1:
            self._paint_principal_rays(painter, obj_x, obj_y_tip, ox, oy, f_px, scale)

        # ── 7. Movable Observation Screen ──
        if self._show_screen:
            scr_px = ox + self._screen_x_cm * scale
            if self._canvas_rect.left() + 20.0 <= scr_px <= self._canvas_rect.right() - 20.0:
                is_focused = is_real and abs(self._screen_x_cm - d_p) < 0.8
                scr_color = QColor("#22c55e") if is_focused else QColor("#94a3b8")
                pen_scr = QPen(scr_color, 2.5 if is_focused else 1.8)
                painter.setPen(pen_scr)
                painter.drawLine(QPointF(scr_px, oy - 110.0), QPointF(scr_px, oy + 110.0))

                # Screen handle bead
                painter.setBrush(QBrush(scr_color))
                painter.drawEllipse(QPointF(scr_px, oy - 110.0), 5.0, 5.0)

                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.setPen(QPen(scr_color))
                scr_text = "Màn E (ĐÚNG TIÊU CỰ!)" if is_focused else "Màn E"
                painter.drawText(QPointF(scr_px - 20.0, oy - 118.0), scr_text)

        painter.restore()

    def _paint_principal_rays(
        self,
        painter: QPainter,
        obj_x: float,
        obj_y_tip: float,
        ox: float,
        oy: float,
        f_px: float,
        scale: float,
    ) -> None:
        """Paints Ray 1 (Parallel), Ray 2 (Central), Ray 3 (Focal)."""
        right_edge = self._canvas_rect.right() - 5.0
        left_edge = self._canvas_rect.left() + 5.0

        # ── Ray 1: Parallel to Axis (Gold) ──
        # Incident: from B(obj_x, obj_y_tip) to lens plane at (ox, obj_y_tip)
        pen_r1 = QPen(QColor("#eab308"), 1.6)
        painter.setPen(pen_r1)
        painter.drawLine(QPointF(obj_x, obj_y_tip), QPointF(ox, obj_y_tip))

        if self._lens_type == "CONVERGING":
            # Converging: passes through primary focus F' at (ox + f_px, oy)
            slope1 = (oy - obj_y_tip) / f_px
            end_x1 = right_edge
            end_y1 = obj_y_tip + slope1 * (end_x1 - ox)
            painter.drawLine(QPointF(ox, obj_y_tip), QPointF(end_x1, end_y1))

            # Virtual backward extension if virtual image (d < f)
            if self.image_d_prime_cm < 0.0:
                pen_dash = QPen(QColor(234, 179, 8, 120), 1.2, Qt.PenStyle.DashLine)
                painter.setPen(pen_dash)
                back_x = left_edge
                back_y = obj_y_tip + slope1 * (back_x - ox)
                painter.drawLine(QPointF(ox, obj_y_tip), QPointF(back_x, back_y))
        else:
            # Diverging: appears to diverge from F' at (ox - f_px, oy)
            slope1 = (obj_y_tip - oy) / f_px
            end_x1 = right_edge
            end_y1 = obj_y_tip + slope1 * (end_x1 - ox)
            painter.drawLine(QPointF(ox, obj_y_tip), QPointF(end_x1, end_y1))

            # Virtual backward extension through F' (ox - f_px, oy)
            pen_dash = QPen(QColor(234, 179, 8, 120), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(pen_dash)
            painter.drawLine(QPointF(ox, obj_y_tip), QPointF(ox - f_px, oy))

        # ── Ray 2: Central Ray through O (Orange) ──
        # Straight through optical center without deviation
        pen_r2 = QPen(QColor("#f97316"), 1.6)
        painter.setPen(pen_r2)
        slope2 = (oy - obj_y_tip) / (ox - obj_x)
        end_x2 = right_edge
        end_y2 = oy + slope2 * (end_x2 - ox)
        painter.drawLine(QPointF(obj_x, obj_y_tip), QPointF(end_x2, end_y2))

        # Virtual backward extension if virtual image
        if self.image_d_prime_cm < 0.0:
            pen_dash2 = QPen(QColor(249, 115, 22, 120), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(pen_dash2)
            back_x2 = left_edge
            back_y2 = oy + slope2 * (back_x2 - ox)
            painter.drawLine(QPointF(ox, oy), QPointF(back_x2, back_y2))

    def _paint_arrowhead(self, painter: QPainter, tip: QPointF, angle_rad: float, color: QColor) -> None:
        arrow_size = 7.5
        p1 = QPointF(
            tip.x() - arrow_size * math.cos(angle_rad - math.pi / 6.0),
            tip.y() - arrow_size * math.sin(angle_rad - math.pi / 6.0),
        )
        p2 = QPointF(
            tip.x() - arrow_size * math.cos(angle_rad + math.pi / 6.0),
            tip.y() - arrow_size * math.sin(angle_rad + math.pi / 6.0),
        )
        path = QPainterPath()
        path.moveTo(tip)
        path.lineTo(p1)
        path.lineTo(p2)
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    def _paint_controls_and_hud(self, painter: QPainter) -> None:
        """Paints parameter sliders, telemetry, presets, and action buttons."""
        hud_left = 525.0
        hud_w = 300.0

        # ── 1. Parameters Sliders Card ──
        card1_rect = QRectF(hud_left, 45.0, hud_w, 150.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card1_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 50.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "THÔNG SỐ BĂNG QUANG HỌC",
        )

        # Lens Type Toggle Buttons: [Hội Tụ] and [Phân Kỳ]
        btn_cv = self._get_mode_button_rect("CONVERGING")
        btn_dv = self._get_mode_button_rect("DIVERGING")
        for btn_r, m_id, m_lbl in [(btn_cv, "CONVERGING", "Hội Tụ (f > 0)"), (btn_dv, "DIVERGING", "Phân Kỳ (f < 0)")]:
            is_active = (self._lens_type == m_id)
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(14, 165, 233, 60) if is_active else QColor(30, 41, 59, 160)))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#cbd5e1")))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, m_lbl)

        # Focal Length Slider
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QRectF(hud_left + 12.0, 98.0, 200.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"Tiêu cự |f| = {abs(self._focal_length_cm):.1f} cm")
        f_track, f_handle = self._get_f_slider_geometry()
        self._paint_mini_slider(painter, f_track, f_handle, QColor("#38bdf8"))

        # Object Distance d and Height h Sliders
        painter.drawText(QRectF(hud_left + 12.0, 126.0, 130.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"Khoảng d = {self._object_d_cm:.1f} cm")
        painter.drawText(QRectF(hud_left + 160.0, 126.0, 130.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"Chiều cao h = {self._object_h_cm:.1f} cm")

        d_track, d_handle = self._get_d_slider_geometry()
        self._paint_mini_slider(painter, d_track, d_handle, QColor("#f59e0b"))

        h_track, h_handle = self._get_h_slider_geometry()
        self._paint_mini_slider(painter, h_track, h_handle, QColor("#10b981"))

        # ── 2. Optics Telemetry & Image Classification Card ──
        card2_rect = QRectF(hud_left, 202.0, hud_w, 118.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card2_rect, 8.0, 8.0)

        # Image nature classification
        d_p = self.image_d_prime_cm
        k_val = self.magnification_k

        if abs(d_p) > 2000.0:
            nature_txt = "ẢNH Ở VÔ CỰC (d = f)"
            nature_col = QColor("#fbbf24")
        elif d_p > 0.0:
            mag_str = "Phóng đại" if abs(k_val) > 1.0 else ("Thu nhỏ" if abs(k_val) < 1.0 else "Bằng vật")
            nature_txt = f"ẢNH THẬT, NGƯỢC CHIỀU ({mag_str})"
            nature_col = QColor("#10b981")
        else:
            mag_str = "Phóng đại" if abs(k_val) > 1.0 else "Thu nhỏ"
            nature_txt = f"ẢNH ẢO, CÙNG CHIỀU ({mag_str})"
            nature_col = QColor("#ec4899")

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(nature_col))
        painter.drawText(QRectF(hud_left + 12.0, 206.0, hud_w - 24.0, 18.0), Qt.AlignmentFlag.AlignLeft, nature_txt)

        # Big Values: d' and k
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        d_p_str = f"d' = {d_p:+.1f} cm" if abs(d_p) < 2000.0 else "d' → ∞"
        painter.drawText(QRectF(hud_left + 12.0, 226.0, 140.0, 20.0), Qt.AlignmentFlag.AlignLeft, d_p_str)
        painter.setPen(QPen(QColor("#38bdf8")))
        k_str = f"k = {k_val:+.2f}" if abs(k_val) < 200.0 else "k → ∞"
        painter.drawText(QRectF(hud_left + 155.0, 226.0, 140.0, 20.0), Qt.AlignmentFlag.AlignLeft, k_str)

        # Thin-lens equation & image height
        h_p = self.image_h_prime_cm
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(
            QRectF(hud_left + 12.0, 250.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft,
            f"• Công thức thấu kính: 1/f = 1/d + 1/d' (f = {self._focal_length_cm:+.1f} cm)",
        )
        h_p_str = f"A'B' = {abs(h_p):.1f} cm" if abs(h_p) < 500.0 else "A'B' → ∞"
        painter.drawText(
            QRectF(hud_left + 12.0, 268.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft,
            f"• Kích thước: Vật AB = {self._object_h_cm:.1f} cm ➔ Ảnh {h_p_str}",
        )
        painter.drawText(
            QRectF(hud_left + 12.0, 286.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft,
            f"• Độ tụ: D = 1/f = {100.0 / self._focal_length_cm:+.2f} đi-ốp (dp)",
        )

        # ── 3. Presets (2x2 Grid) ──
        card3_rect = QRectF(hud_left, 327.0, hud_w, 98.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card3_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QRectF(hud_left + 12.0, 330.0, hud_w - 24.0, 16.0), Qt.AlignmentFlag.AlignLeft, "ỨNG DỤNG QUANG HỌC KINH ĐIỂN")

        presets = [
            ("MAGNIFYING_GLASS", "🔍 Kính Lúp (d < f)", 0, 0),
            ("PROJECTOR_LENS", "📽️ Máy Chiếu (f < d < 2f)", 0, 1),
            ("CAMERA_OBJECTIVE", "📷 Máy Ảnh (d > 2f)", 1, 0),
            ("DIVERGING_EYEGLASS", "👓 Kính Cận (TK Phân Kỳ)", 1, 1),
        ]
        for p_id, p_label, row, col in presets:
            btn_r = self._get_preset_button_rect(row, col)
            is_active = (self._preset == p_id)
            is_hover = (self._hover_target == f"PRESET_{p_id}")
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(14, 165, 233, 60) if is_active else (QColor(51, 65, 85, 200) if is_hover else QColor(30, 41, 59, 160))))
            painter.drawRoundedRect(btn_r, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#e2e8f0")))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, p_label)

        # ── 4. Visual Toggles ──
        toggles = [
            ("RAYS", "📏 3 Tia Sáng", self._show_principal_rays, 0),
            ("SCREEN", "🖥️ Màn Hứng", self._show_screen, 1),
        ]
        tog_w = (hud_w - 24.0 - 12.0) / 2.0
        for tog_key, tog_label, is_on, idx in toggles:
            tog_r = QRectF(hud_left + 12.0 + idx * (tog_w + 12.0), 432.0, tog_w, 28.0)
            is_hover = (self._hover_target == f"TOGGLE_{tog_key}")
            painter.setPen(QPen(QColor("#10b981") if is_on else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(16, 185, 129, 60) if is_on else (QColor(51, 65, 85, 200) if is_hover else QColor(30, 41, 59, 140))))
            painter.drawRoundedRect(tog_r, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#34d399") if is_on else QColor("#94a3b8")))
            painter.drawText(tog_r, Qt.AlignmentFlag.AlignCenter, tog_label)

        # ── 5. Action Buttons (Stamp & Close) ──
        stamp_btn = self._get_stamp_button_rect()
        h_stamp = (self._hover_target == "STAMP")
        painter.setPen(QPen(QColor("#0ea5e9"), 1.5))
        painter.setBrush(QBrush(QColor(14, 165, 233, 210) if h_stamp else QColor(2, 132, 199, 180)))
        painter.drawRoundedRect(stamp_btn, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(stamp_btn, Qt.AlignmentFlag.AlignCenter, "🖊️ Đóng Dấu Vector")

        close_b = self._get_close_bottom_button_rect()
        h_cb = (self._hover_target == "CLOSE_BOTTOM")
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor(71, 85, 105, 200) if h_cb else QColor(51, 65, 85, 160)))
        painter.drawRoundedRect(close_b, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(close_b, Qt.AlignmentFlag.AlignCenter, "Đóng")

    def _paint_mini_slider(self, painter: QPainter, track: QRectF, handle: QRectF, color: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(track, 3.0, 3.0)
        filled = QRectF(track.left(), track.top(), handle.center().x() - track.left(), track.height())
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(filled, 3.0, 3.0)
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(handle)

    # ── Geometry & Layout Helpers ─────────────────────────────────────────────

    def _get_close_button_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 8.0, 24.0, 24.0)

    def _get_mode_button_rect(self, mode: str) -> QRectF:
        hud_left = 525.0
        w = 134.0
        h = 22.0
        y = 70.0
        if mode == "CONVERGING":
            return QRectF(hud_left + 12.0, y, w, h)
        else:
            return QRectF(hud_left + 12.0 + w + 8.0, y, w, h)

    def _get_f_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 114.0, 276.0, 6.0)
        frac = (abs(self._focal_length_cm) - 5.0) / (25.0 - 5.0)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_d_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 144.0, 130.0, 6.0)
        frac = (self._object_d_cm - 4.0) / (40.0 - 4.0)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_h_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 160.0, 144.0, 128.0, 6.0)
        frac = (self._object_h_cm - 2.0) / (12.0 - 2.0)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_preset_button_rect(self, row: int, col: int) -> QRectF:
        hud_left = 525.0
        w = 134.0
        h = 26.0
        x = hud_left + 12.0 + col * (w + 8.0)
        y = 350.0 + row * (h + 8.0)
        return QRectF(x, y, w, h)

    def _get_stamp_button_rect(self) -> QRectF:
        return QRectF(525.0 + 12.0, 495.0, 175.0, 36.0)

    def _get_close_bottom_button_rect(self) -> QRectF:
        return QRectF(525.0 + 195.0, 495.0, 93.0, 36.0)

    # ── Presets & Physical Setters ────────────────────────────────────────────

    def set_lens_type(self, mode: str) -> None:
        """Sets lens mode ("CONVERGING" with f > 0 or "DIVERGING" with f < 0)."""
        mode_up = mode.upper()
        if mode_up == "DIVERGING":
            self._lens_type = "DIVERGING"
            if self._focal_length_cm > 0:
                self._focal_length_cm = -self._focal_length_cm
        else:
            self._lens_type = "CONVERGING"
            if self._focal_length_cm < 0:
                self._focal_length_cm = -self._focal_length_cm

        self.mode_changed.emit(self._lens_type)
        self._emit_updates()

    def set_focal_length(self, f_cm: float) -> None:
        """Sets focal length magnitude clamped to [5.0, 25.0] cm."""
        mag = max(5.0, min(25.0, abs(float(f_cm))))
        self._focal_length_cm = mag if self._lens_type == "CONVERGING" else -mag
        self._emit_updates()

    def set_object_distance(self, d_cm: float) -> None:
        """Sets object distance d clamped to [4.0, 40.0] cm."""
        self._object_d_cm = max(4.0, min(40.0, float(d_cm)))
        self._emit_updates()

    def set_object_height(self, h_cm: float) -> None:
        """Sets object height h clamped to [2.0, 12.0] cm."""
        self._object_h_cm = max(2.0, min(12.0, float(h_cm)))
        self._emit_updates()

    def set_preset(self, preset_name: str) -> None:
        """Switches through authentic optical device presets."""
        self._preset = preset_name
        self._apply_preset(preset_name)
        self.preset_changed.emit(preset_name)
        self.update()

    def _apply_preset(self, preset: str) -> None:
        p_up = preset.upper()
        if p_up == "MAGNIFYING_GLASS":
            # d < f => virtual upright magnified image
            self._lens_type = "CONVERGING"
            self._focal_length_cm = 12.0
            self._object_d_cm = 7.0
            self._object_h_cm = 5.0
            self._screen_x_cm = 25.0
        elif p_up == "PROJECTOR_LENS":
            # f < d < 2f => real inverted magnified image
            self._lens_type = "CONVERGING"
            self._focal_length_cm = 10.0
            self._object_d_cm = 15.0
            self._object_h_cm = 5.0
            self._screen_x_cm = 30.0  # d' = 30 cm
        elif p_up == "CAMERA_OBJECTIVE":
            # d > 2f => real inverted diminished image
            self._lens_type = "CONVERGING"
            self._focal_length_cm = 10.0
            self._object_d_cm = 30.0
            self._object_h_cm = 8.0
            self._screen_x_cm = 15.0  # d' = 15 cm
        elif p_up == "DIVERGING_EYEGLASS":
            # Diverging lens (kính cận)
            self._lens_type = "DIVERGING"
            self._focal_length_cm = -12.0
            self._object_d_cm = 18.0
            self._object_h_cm = 6.0
            self._screen_x_cm = 20.0

        self._emit_updates()

    def _emit_updates(self) -> None:
        self.optics_updated.emit(self._object_d_cm, self.image_d_prime_cm, self._focal_length_cm, self.magnification_k)
        self.update()

    # ── Mouse & Air Gesture Interaction ───────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Close
        if self._get_close_button_rect().contains(pos) or self._get_close_bottom_button_rect().contains(pos):
            self.close()
            event.accept()
            return

        # Stamp
        if self._get_stamp_button_rect().contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        # Lens Mode Buttons
        if self._get_mode_button_rect("CONVERGING").contains(pos):
            self.set_lens_type("CONVERGING")
            event.accept()
            return
        elif self._get_mode_button_rect("DIVERGING").contains(pos):
            self.set_lens_type("DIVERGING")
            event.accept()
            return

        # Presets
        presets = [
            ("MAGNIFYING_GLASS", 0, 0),
            ("PROJECTOR_LENS", 0, 1),
            ("CAMERA_OBJECTIVE", 1, 0),
            ("DIVERGING_EYEGLASS", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(pos):
                self.set_preset(p_id)
                event.accept()
                return

        # Visual Toggles
        hud_left = 525.0
        hud_w = 300.0
        tog_w = (hud_w - 24.0 - 12.0) / 2.0
        for idx, attr in enumerate(["_show_principal_rays", "_show_screen"]):
            tog_r = QRectF(hud_left + 12.0 + idx * (tog_w + 12.0), 432.0, tog_w, 28.0)
            if tog_r.contains(pos):
                setattr(self, attr, not getattr(self, attr))
                self.update()
                event.accept()
                return

        # Sliders
        f_track, f_handle = self._get_f_slider_geometry()
        if f_track.contains(pos) or f_handle.contains(pos):
            self._drag_target = "SLIDER_F"
            frac = (pos.x() - f_track.left()) / f_track.width()
            self.set_focal_length(5.0 + frac * (25.0 - 5.0))
            event.accept()
            return

        d_track, d_handle = self._get_d_slider_geometry()
        if d_track.contains(pos) or d_handle.contains(pos):
            self._drag_target = "SLIDER_D"
            frac = (pos.x() - d_track.left()) / d_track.width()
            self.set_object_distance(4.0 + frac * (40.0 - 4.0))
            event.accept()
            return

        h_track, h_handle = self._get_h_slider_geometry()
        if h_track.contains(pos) or h_handle.contains(pos):
            self._drag_target = "SLIDER_H"
            frac = (pos.x() - h_track.left()) / h_track.width()
            self.set_object_height(2.0 + frac * (12.0 - 2.0))
            event.accept()
            return

        # Draggable Object Tip B on Bench
        ox = self._optic_center.x()
        oy = self._optic_center.y()
        obj_x = ox - self._object_d_cm * self._scale
        obj_y_tip = oy - self._object_h_cm * self._scale
        if QRectF(obj_x - 12.0, obj_y_tip - 12.0, 24.0, 24.0).contains(pos):
            self._drag_target = "OBJECT_TIP"
            event.accept()
            return

        # Title bar dragging
        if QRectF(0.0, 0.0, self._width, 40.0).contains(pos):
            self._drag_target = "TITLE"
            self._drag_offset = event.scenePos() - self.pos()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        if self._drag_target == "TITLE":
            self.setPos(event.scenePos() - self._drag_offset)
            event.accept()
            return
        elif self._drag_target == "SLIDER_F":
            f_track, _ = self._get_f_slider_geometry()
            frac = (pos.x() - f_track.left()) / f_track.width()
            self.set_focal_length(5.0 + frac * (25.0 - 5.0))
            event.accept()
            return
        elif self._drag_target == "SLIDER_D":
            d_track, _ = self._get_d_slider_geometry()
            frac = (pos.x() - d_track.left()) / d_track.width()
            self.set_object_distance(4.0 + frac * (40.0 - 4.0))
            event.accept()
            return
        elif self._drag_target == "SLIDER_H":
            h_track, _ = self._get_h_slider_geometry()
            frac = (pos.x() - h_track.left()) / h_track.width()
            self.set_object_height(2.0 + frac * (12.0 - 2.0))
            event.accept()
            return
        elif self._drag_target == "OBJECT_TIP":
            ox = self._optic_center.x()
            oy = self._optic_center.y()
            # Calculate d and h from pos
            new_d = (ox - pos.x()) / self._scale
            new_h = (oy - pos.y()) / self._scale
            self.set_object_distance(new_d)
            self.set_object_height(new_h)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._drag_target = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old = self._hover_target

        if self._get_close_button_rect().contains(pos):
            self._hover_target = "CLOSE"
        elif self._get_close_bottom_button_rect().contains(pos):
            self._hover_target = "CLOSE_BOTTOM"
        elif self._get_stamp_button_rect().contains(pos):
            self._hover_target = "STAMP"
        elif self._get_f_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_F"
        elif self._get_d_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_D"
        elif self._get_h_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_H"
        else:
            self._hover_target = None
            presets = [
                ("MAGNIFYING_GLASS", 0, 0),
                ("PROJECTOR_LENS", 0, 1),
                ("CAMERA_OBJECTIVE", 1, 0),
                ("DIVERGING_EYEGLASS", 1, 1),
            ]
            for p_id, r, c in presets:
                if self._get_preset_button_rect(r, c).contains(pos):
                    self._hover_target = f"PRESET_{p_id}"
                    break

        if self._hover_target != old:
            self.update()

    # ── Touchless Air Gesture API ─────────────────────────────────────────────

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        loc = self.mapFromScene(scene_pt)

        # Close
        if self._get_close_button_rect().contains(loc) or self._get_close_bottom_button_rect().contains(loc):
            self.close()
            return True

        # Stamp
        if self._get_stamp_button_rect().contains(loc):
            self.stamp_to_board()
            return True

        # Presets
        presets = [
            ("MAGNIFYING_GLASS", 0, 0),
            ("PROJECTOR_LENS", 0, 1),
            ("CAMERA_OBJECTIVE", 1, 0),
            ("DIVERGING_EYEGLASS", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(loc):
                self.set_preset(p_id)
                return True

        # Mode buttons
        if self._get_mode_button_rect("CONVERGING").contains(loc):
            self.set_lens_type("CONVERGING")
            return True
        elif self._get_mode_button_rect("DIVERGING").contains(loc):
            self.set_lens_type("DIVERGING")
            return True

        # Distance d slider
        d_track, _ = self._get_d_slider_geometry()
        if d_track.contains(loc):
            frac = (loc.x() - d_track.left()) / d_track.width()
            self.set_object_distance(4.0 + frac * (40.0 - 4.0))
            return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── Board Integration & Vector Stamping ───────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of thin lens ray tracing onto LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = PhysicsTools.create_thin_lens_diagram(
            center=(cx, cy + 30.0),
            width=500.0,
            height=320.0,
            focal_length_cm=self._focal_length_cm,
            object_d_cm=self._object_d_cm,
            object_h_cm=self._object_h_cm,
            lens_type=self._lens_type,
            preset=self._preset,
        )

        for item in diagram_items:
            board.addItem(item)

        logger.info(f"Thin Lens Optical Bench diagram stamped onto LessonBoard (f={self._focal_length_cm:+.1f}cm).")

    def close(self) -> None:
        self.setVisible(False)
        self.update()

    # ── Public Properties for Tests & State Verification ─────────────────────

    @property
    def focal_length_cm(self) -> float:
        return self._focal_length_cm

    @property
    def object_d_cm(self) -> float:
        return self._object_d_cm

    @property
    def object_h_cm(self) -> float:
        return self._object_h_cm

    @property
    def lens_type(self) -> str:
        return self._lens_type

    @property
    def preset(self) -> str:
        return self._preset
