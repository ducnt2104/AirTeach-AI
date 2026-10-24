"""
AirTeach AI - Interactive Special Relativity Spacetime Minkowski Diagram & Lorentz Transformation Simulator
=============================================================================================================
Provides an authentic physics and mathematics simulation of Special Relativity on Minkowski Spacetime diagrams:
  - Minkowski Metric and Spacetime Interval Invariant (c = 1 natural units):
      ds² = c²dt² - dx² = dt² - dx² (timelike positive convention)
      Invariant under all Lorentz boosts: Δs² = Δ(ct)² - Δx² = Δ(ct')² - Δx'²
  - Lorentz Boost along x-axis with relative velocity β = v/c ∈ [0.0, 0.99]:
      γ = 1 / sqrt(1 - β²)
      ct' = γ * (ct - β * x)
      x'  = γ * (x - β * ct)
      Inverse Transformation:
      ct  = γ * (ct' + β * x')
      x   = γ * (x' + β * ct')
  - Geometric Axes in Minkowski Diagram (ct vertical, x horizontal):
      - Invariant Light Cone at 45°: ct = ±x (c is invariant in all inertial frames).
      - ct' axis (worldline of S' origin x' = 0): x = β * ct, tilted by angle θ = arctan(β) towards light cone.
      - x' axis (line of simultaneity in S' ct' = 0): ct = β * x, tilted by angle θ = arctan(β) upwards towards light cone.
      - Both axes close symmetrically like scissors towards the 45° light cone as β -> 1!
  - Invariant Calibration Hyperbolas:
      - Timelike:  (ct)² - x² = τ² (calibrates unit ticks on ct' axis)
      - Spacelike: x² - (ct)² = σ² (calibrates unit ticks on x' axis)
  - Spacetime Interval Classification:
      - Timelike (Δs² > 0): Causally connectable, proper time Δτ = sqrt(Δs²).
      - Lightlike / Null (Δs² = 0): Connected by speed of light.
      - Spacelike (Δs² < 0): Causally disconnected, proper distance Δσ = sqrt(-Δs²).
        Exhibits Relativity of Simultaneity (time order can reverse).
  - Curriculum Presets:
      1. MUON_ATMOSPHERIC_DECAY (β = 0.995, γ = 10.0, dilated lifetime t = γ * τ₀)
      2. BARN_POLE_PARADOX (β = 0.866, γ = 2.0, relativity of simultaneity & length contraction)
      3. TWIN_PARADOX_WORLDLINE (β = 0.80, broken worldline & proper time gap)
      4. LIGHT_CLOCK_GEDANKEN (β = 0.60, γ = 1.25, photon zigzag path)
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
    QRadialGradient,
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

# Speed of light in natural units
C_LIGHT: float = 1.0


class InteractiveSpacetimeRelativityItem(QGraphicsObject):
    """
    Interactive Minkowski Spacetime Diagram & Lorentz Boost Simulator item for QGraphicsScene.
    """

    relativity_updated = Signal(float, float, float)  # (beta, gamma, delta_s_sq)
    preset_changed = Signal(str)                       # ("MUON_ATMOSPHERIC_DECAY", etc.)
    events_changed = Signal(float, float, float, float)  # (x1, ct1, x2, ct2)

    def __init__(
        self,
        pos: tuple[float, float] = (300.0, 160.0),
        preset: str = "LIGHT_CLOCK_GEDANKEN",
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

        # Main bounding box
        self._width: float = 840.0
        self._height: float = 560.0

        # Physical and mathematical parameters
        self._beta: float = 0.60  # v/c in [0.0, 0.99]
        self._gamma: float = 1.0 / math.sqrt(max(0.0001, 1.0 - self._beta * self._beta))
        self._preset: str = preset

        # Feature display toggles
        self._show_hyperbolas: bool = True
        self._show_boosted_grid: bool = True
        self._show_light_cone: bool = True
        self._show_twin_worldline: bool = False

        # Minkowski Diagram Canvas Layout
        self._canvas_rect = QRectF(15.0, 45.0, 500.0, 500.0)
        self._origin_scene = QPointF(265.0, 295.0)  # Center of Minkowski canvas
        self._scale: float = 30.0  # 30 pixels per spacetime unit

        # Interactive Spacetime Events in S frame (x, ct)
        self._event1 = QPointF(0.0, 0.0)
        self._event2 = QPointF(3.0, 4.0)

        # Active interaction states
        self._drag_target: Optional[str] = None  # "TITLE", "EVENT1", "EVENT2", "SLIDER_BETA"
        self._drag_offset: QPointF = QPointF(0.0, 0.0)
        self._hover_target: Optional[str] = None

        # Worldline trajectory points for twin paradox or muon paths
        self._worldline_points: list[QPointF] = []

        # Setup initial preset configuration
        self._apply_preset(self._preset)

    # ── Bounding Rect & Painting ──────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns outer bounding rectangle including margin."""
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders comprehensive Minkowski Spacetime Diagram & Relativistic Inspector."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Outer Frosted Glass Card Container
        self._paint_outer_container(painter)

        # 2. Left Minkowski Spacetime Diagram
        self._paint_minkowski_diagram(painter)

        # 3. Right Control & Relativistic HUD
        self._paint_relativity_hud(painter)

        painter.restore()

    def _paint_outer_container(self, painter: QPainter) -> None:
        """Paints main frosted card background, border and title bar."""
        outer_rect = self.boundingRect()

        # Dark cosmic slate background
        painter.setPen(QPen(QColor("#334155"), 1.8))
        painter.setBrush(QBrush(QColor(15, 23, 42, 245)))  # Slate-900 96% opacity
        painter.drawRoundedRect(outer_rect, 14.0, 14.0)

        # Top Title Bar
        title_rect = QRectF(0.0, 0.0, self._width, 40.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59, 230)))
        painter.drawRoundedRect(title_rect, 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 26.0, self._width, 14.0))

        # Title Text with Relativistic Symbol
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(
            QRectF(18.0, 8.0, 520.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "⚡ KHÔNG-THỜI GIAN MINKOWSKI & BIẾN ĐỔI LORENTZ",
        )

        # Subtitle badge: Preset indicator
        preset_labels = {
            "MUON_ATMOSPHERIC_DECAY": "🌌 Hạt Muon Khí Quyển (Time Dilation)",
            "BARN_POLE_PARADOX": "🏚️ Nghịch Lý Cọc-Kho (Simultaneity)",
            "TWIN_PARADOX_WORLDLINE": "🚀 Nghịch Lý Sinh Đôi (Twin Paradox)",
            "LIGHT_CLOCK_GEDANKEN": "⏱️ Đồng Hồ Ánh Sáng (Light Clock)",
        }
        badge_text = preset_labels.get(self._preset, "Mô Hình Tương Đối Hẹp")
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(520.0, 8.0, 260.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            badge_text,
        )

        # Close button in top-right corner
        close_btn = self._get_close_button_rect()
        hover_close = (self._hover_target == "CLOSE")
        painter.setBrush(QBrush(QColor("#dc2626") if hover_close else QColor(239, 68, 68, 160)))
        painter.setPen(QPen(QColor("#fca5a5"), 1.2))
        painter.drawRoundedRect(close_btn, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(close_btn, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_minkowski_diagram(self, painter: QPainter) -> None:
        """Paints axes, grid, light cones, boosted frame S', hyperbolas, and events."""
        painter.save()
        painter.setClipRect(self._canvas_rect)

        # Deep cosmic viewport
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.setBrush(QBrush(QColor("#030712")))  # True deep space
        painter.drawRoundedRect(self._canvas_rect, 8.0, 8.0)

        cx = self._origin_scene.x()
        cy = self._origin_scene.y()
        scale = self._scale

        # 1. Base S Frame Coordinate Grid (Light gray)
        painter.setPen(QPen(QColor(51, 65, 85, 90), 0.8, Qt.PenStyle.DotLine))
        for u in range(-7, 8):
            if u == 0:
                continue
            # Vertical lines (x = const)
            px = cx + u * scale
            painter.drawLine(QPointF(px, self._canvas_rect.top()), QPointF(px, self._canvas_rect.bottom()))
            # Horizontal lines (ct = const)
            py = cy - u * scale
            painter.drawLine(QPointF(self._canvas_rect.left(), py), QPointF(self._canvas_rect.right(), py))

        # 2. Invariant Light Cone (Future and Past cones)
        if self._show_light_cone:
            self._paint_light_cones(painter, cx, cy)

        # 3. Invariant Hyperbolas (Timelike and Spacelike)
        if self._show_hyperbolas:
            self._paint_invariant_hyperbolas(painter, cx, cy, scale)

        # 4. Boosted Frame S' Grid Lines (Lines of Simultaneity and Stationary Worldlines)
        if self._show_boosted_grid and abs(self._beta) > 0.01:
            self._paint_boosted_grid(painter, cx, cy, scale)

        # 5. Base S Frame Axes (x horizontal, ct vertical)
        painter.setPen(QPen(QColor("#94a3b8"), 1.8))
        # Horizontal x-axis
        painter.drawLine(QPointF(self._canvas_rect.left() + 5.0, cy), QPointF(self._canvas_rect.right() - 5.0, cy))
        # Arrowhead x
        self._paint_arrowhead(painter, QPointF(self._canvas_rect.right() - 5.0, cy), 0.0, QColor("#94a3b8"))
        # Vertical ct-axis
        painter.drawLine(QPointF(cx, self._canvas_rect.bottom() - 5.0), QPointF(cx, self._canvas_rect.top() + 5.0))
        # Arrowhead ct
        self._paint_arrowhead(painter, QPointF(cx, self._canvas_rect.top() + 5.0), -math.pi / 2.0, QColor("#94a3b8"))

        # S Axis Labels
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QPointF(self._canvas_rect.right() - 24.0, cy - 8.0), "x")
        painter.drawText(QPointF(cx + 8.0, self._canvas_rect.top() + 18.0), "ct")
        painter.drawText(QPointF(cx - 16.0, cy + 18.0), "O")

        # 6. Boosted Frame S' Axes (tilted by arctan(beta))
        if abs(self._beta) > 0.005:
            self._paint_boosted_axes(painter, cx, cy)

        # 7. Worldline paths (e.g. Twin Paradox or Muon trajectory)
        if len(self._worldline_points) >= 2:
            self._paint_worldlines(painter, cx, cy, scale)

        # 8. Interactive Events E1 and E2
        self._paint_events_and_interval(painter, cx, cy, scale)

        painter.restore()

    def _paint_light_cones(self, painter: QPainter, cx: float, cy: float) -> None:
        """Paints 45° invariant light cones with glowing translucent amber/gold tint."""
        # Future Light Cone polygon
        w = 260.0
        future_path = QPainterPath()
        future_path.moveTo(cx, cy)
        future_path.lineTo(cx + w, cy - w)
        future_path.lineTo(cx - w, cy - w)
        future_path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(245, 158, 11, 28)))  # Amber gold subtle glow
        painter.drawPath(future_path)

        # Past Light Cone polygon
        past_path = QPainterPath()
        past_path.moveTo(cx, cy)
        past_path.lineTo(cx + w, cy + w)
        past_path.lineTo(cx - w, cy + w)
        past_path.closeSubpath()
        painter.setBrush(QBrush(QColor(245, 158, 11, 20)))
        painter.drawPath(past_path)

        # Boundary Light Rays ct = ±x (c = 1)
        pen_cone = QPen(QColor("#fbbf24"), 1.6, Qt.PenStyle.DashDotLine)
        painter.setPen(pen_cone)
        painter.drawLine(QPointF(cx - w, cy + w), QPointF(cx + w, cy - w))  # ct = x ray
        painter.drawLine(QPointF(cx - w, cy - w), QPointF(cx + w, cy + w))  # ct = -x ray

        # Light cone labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#fde047")))
        painter.drawText(QPointF(cx + w - 45.0, cy - w + 14.0), "ct = +x (c)")
        painter.drawText(QPointF(cx - w + 10.0, cy - w + 14.0), "ct = −x (c)")
        painter.drawText(QPointF(cx - 36.0, cy - 70.0), "TƯƠNG LAI")
        painter.drawText(QPointF(cx - 32.0, cy + 80.0), "QUÁ KHỨ")

    def _paint_invariant_hyperbolas(self, painter: QPainter, cx: float, cy: float, scale: float) -> None:
        """Paints timelike (ct)² - x² = τ² and spacelike x² - (ct)² = σ² hyperbolas."""
        # Timelike Hyperbolas in cyan (proper times τ = 1, 2, 3, 4)
        pen_time = QPen(QColor(56, 189, 248, 110), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(pen_time)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for tau in [1.0, 2.0, 3.0, 4.0]:
            # Upper branch: ct = +sqrt(tau² + x²)
            path_up = QPainterPath()
            first = True
            for i in range(-120, 121):
                x_val = (i / 120.0) * 6.5
                ct_val = math.sqrt(tau * tau + x_val * x_val)
                px = cx + x_val * scale
                py = cy - ct_val * scale
                if first:
                    path_up.moveTo(px, py)
                    first = False
                else:
                    path_up.lineTo(px, py)
            painter.drawPath(path_up)

        # Spacelike Hyperbolas in violet (proper distances σ = 2, 4)
        pen_space = QPen(QPen(QColor(168, 85, 247, 85), 1.1, Qt.PenStyle.DashLine))
        painter.setPen(pen_space)

        for sigma in [2.0, 4.0]:
            # Right branch: x = +sqrt(sigma² + ct²)
            path_right = QPainterPath()
            first = True
            for i in range(-120, 121):
                ct_val = (i / 120.0) * 6.5
                x_val = math.sqrt(sigma * sigma + ct_val * ct_val)
                px = cx + x_val * scale
                py = cy - ct_val * scale
                if first:
                    path_right.moveTo(px, py)
                    first = False
                else:
                    path_right.lineTo(px, py)
            painter.drawPath(path_right)

            # Left branch: x = -sqrt(sigma² + ct²)
            path_left = QPainterPath()
            first = True
            for i in range(-120, 121):
                ct_val = (i / 120.0) * 6.5
                x_val = -math.sqrt(sigma * sigma + ct_val * ct_val)
                px = cx + x_val * scale
                py = cy - ct_val * scale
                if first:
                    path_left.moveTo(px, py)
                    first = False
                else:
                    path_left.lineTo(px, py)
            painter.drawPath(path_left)

    def _paint_boosted_axes(self, painter: QPainter, cx: float, cy: float) -> None:
        """Paints boosted S' frame axes ct' and x' with tilt angle theta = arctan(beta)."""
        theta = math.atan(self._beta)
        L = 230.0

        pen_boost = QPen(QColor("#10b981"), 2.2)  # Emerald green for S' frame
        painter.setPen(pen_boost)

        # ct' axis: parametric (x = beta*ct)
        # Unit direction vector: (sin(theta), cos(theta)) in math, where vertical is upwards
        ct_dir = QPointF(math.sin(theta) * L, -math.cos(theta) * L)
        painter.drawLine(QPointF(cx - ct_dir.x(), cy - ct_dir.y()), QPointF(cx + ct_dir.x(), cy + ct_dir.y()))
        self._paint_arrowhead(painter, QPointF(cx + ct_dir.x(), cy + ct_dir.y()), -math.pi / 2.0 + theta, QColor("#10b981"))

        # x' axis: parametric (ct = beta*x)
        # Unit direction vector: (cos(theta), sin(theta)) where ct is upwards
        x_dir = QPointF(math.cos(theta) * L, -math.sin(theta) * L)
        painter.drawLine(QPointF(cx - x_dir.x(), cy - x_dir.y()), QPointF(cx + x_dir.x(), cy + x_dir.y()))
        self._paint_arrowhead(painter, QPointF(cx + x_dir.x(), cy + x_dir.y()), -theta, QColor("#10b981"))

        # Labels for S' axes
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#34d399")))
        painter.drawText(QPointF(cx + ct_dir.x() + 8.0, cy + ct_dir.y() + 6.0), "ct'")
        painter.drawText(QPointF(cx + x_dir.x() + 8.0, cy + x_dir.y() - 6.0), "x'")

        # Angle indicator arc showing theta = arctan(beta)
        painter.setPen(QPen(QColor(52, 211, 153, 160), 1.2, Qt.PenStyle.DotLine))
        arc_radius = 55.0
        angle_deg = math.degrees(theta)
        # Angle from vertical (90 deg in Qt standard) towards tilted ct'
        painter.drawArc(
            QRectF(cx - arc_radius, cy - arc_radius, arc_radius * 2.0, arc_radius * 2.0),
            int(90.0 * 16),
            int(-angle_deg * 16),
        )
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(QPointF(cx + 14.0, cy - arc_radius + 6.0), f"θ={angle_deg:.1f}°")

    def _paint_boosted_grid(self, painter: QPainter, cx: float, cy: float, scale: float) -> None:
        """Paints lines of simultaneity (ct'=const) and stationary worldlines (x'=const) for S'."""
        pen_sprime = QPen(QColor(16, 185, 129, 65), 0.9, Qt.PenStyle.DashLine)
        painter.setPen(pen_sprime)

        gamma = self._gamma
        beta = self._beta

        # Grid range in S'
        for n in [-4, -3, -2, -1, 1, 2, 3, 4]:
            # Line of simultaneity in S': ct' = n
            # ct = beta * x + n / gamma
            x_min = -6.0
            x_max = 6.0
            ct_1 = beta * x_min + n / gamma
            ct_2 = beta * x_max + n / gamma

            p1 = QPointF(cx + x_min * scale, cy - ct_1 * scale)
            p2 = QPointF(cx + x_max * scale, cy - ct_2 * scale)
            painter.drawLine(p1, p2)

            # Stationary worldline in S': x' = n
            # x = beta * ct + n / gamma
            ct_min = -6.0
            ct_max = 6.0
            x_1 = beta * ct_min + n / gamma
            x_2 = beta * ct_max + n / gamma

            q1 = QPointF(cx + x_1 * scale, cy - ct_min * scale)
            q2 = QPointF(cx + x_2 * scale, cy - ct_max * scale)
            painter.drawLine(q1, q2)

    def _paint_worldlines(self, painter: QPainter, cx: float, cy: float, scale: float) -> None:
        """Paints multi-segment worldlines (such as Twin Paradox outbound/return journey)."""
        pen_wl = QPen(QColor("#f43f5e"), 2.2)  # Rose red for traveler worldline
        painter.setPen(pen_wl)

        path = QPainterPath()
        p0 = self._worldline_points[0]
        path.moveTo(cx + p0.x() * scale, cy - p0.y() * scale)
        for pt in self._worldline_points[1:]:
            path.lineTo(cx + pt.x() * scale, cy - pt.y() * scale)
        painter.drawPath(path)

        # Draw segment vertex markers
        painter.setBrush(QBrush(QColor("#fb7185")))
        for pt in self._worldline_points:
            sc_pt = QPointF(cx + pt.x() * scale, cy - pt.y() * scale)
            painter.drawEllipse(sc_pt, 4.0, 4.0)

    def _paint_events_and_interval(self, painter: QPainter, cx: float, cy: float, scale: float) -> None:
        """Paints interactive events E1, E2 and their connecting spacetime interval vector."""
        p1 = QPointF(cx + self._event1.x() * scale, cy - self._event1.y() * scale)
        p2 = QPointF(cx + self._event2.x() * scale, cy - self._event2.y() * scale)

        # Calculate interval Δs² = (Δct)² - (Δx)²
        dx = self._event2.x() - self._event1.x()
        dct = self._event2.y() - self._event1.y()
        ds_sq = dct * dct - dx * dx

        # Color based on interval classification
        if ds_sq > 0.001:
            interval_color = QColor("#38bdf8")  # Timelike: Light sky blue
            interval_style = Qt.PenStyle.SolidLine
        elif ds_sq < -0.001:
            interval_color = QColor("#ec4899")  # Spacelike: Pink / Magenta
            interval_style = Qt.PenStyle.DashLine
        else:
            interval_color = QColor("#fbbf24")  # Lightlike: Amber
            interval_style = Qt.PenStyle.DashDotLine

        # Connecting interval line
        painter.setPen(QPen(interval_color, 2.0, interval_style))
        painter.drawLine(p1, p2)

        # Projections to S axes for E1
        painter.setPen(QPen(QColor(148, 163, 184, 100), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(p1, QPointF(p1.x(), cy))
        painter.drawLine(p1, QPointF(cx, p1.y()))

        # Projections to S axes for E2
        painter.drawLine(p2, QPointF(p2.x(), cy))
        painter.drawLine(p2, QPointF(cx, p2.y()))

        # Draw Event 1 (Amber Gold)
        painter.setPen(QPen(QColor("#f59e0b"), 2.0))
        painter.setBrush(QBrush(QColor("#fef3c7")))
        painter.drawEllipse(p1, 7.0, 7.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(QPointF(p1.x() + 9.0, p1.y() - 9.0), f"E₁({self._event1.x():.1f}, {self._event1.y():.1f})")

        # Draw Event 2 (Cyan / Sky)
        painter.setPen(QPen(QColor("#0284c7"), 2.0))
        painter.setBrush(QBrush(QColor("#e0f2fe")))
        painter.drawEllipse(p2, 7.0, 7.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(p2.x() + 9.0, p2.y() - 9.0), f"E₂({self._event2.x():.1f}, {self._event2.y():.1f})")

    def _paint_arrowhead(self, painter: QPainter, tip: QPointF, angle_rad: float, color: QColor) -> None:
        """Paints a solid directional arrowhead at tip pointing along angle_rad."""
        arrow_size = 9.0
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

    def _paint_relativity_hud(self, painter: QPainter) -> None:
        """Paints right-hand side metrics, sliders, presets, and action buttons."""
        hud_left = 525.0
        hud_w = 300.0

        # Calculate relativistic quantities
        dx = self._event2.x() - self._event1.x()
        dct = self._event2.y() - self._event1.y()
        ds_sq = dct * dct - dx * dx

        # Coordinates in S' frame
        gamma = self._gamma
        beta = self._beta
        x1_prime = gamma * (self._event1.x() - beta * self._event1.y())
        ct1_prime = gamma * (self._event1.y() - beta * self._event1.x())
        x2_prime = gamma * (self._event2.x() - beta * self._event2.y())
        ct2_prime = gamma * (self._event2.y() - beta * self._event2.x())
        dx_prime = x2_prime - x1_prime
        dct_prime = ct2_prime - ct1_prime
        ds_prime_sq = dct_prime * dct_prime - dx_prime * dx_prime

        # ── 1. Velocity & Lorentz Factor Control Card ──
        card1_rect = QRectF(hud_left, 45.0, hud_w, 100.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card1_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 50.0, hud_w - 24.0, 18.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "VẬN TỐC TƯƠNG ĐỐI & HỆ SỐ LORENTZ",
        )

        # Big Values: beta and gamma
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(
            QRectF(hud_left + 12.0, 70.0, 140.0, 24.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"β = v/c = {self._beta:.3f}",
        )
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(hud_left + 155.0, 70.0, 140.0, 24.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"γ = {self._gamma:.3f}",
        )

        # Velocity Slider Bar
        slider_track, slider_handle = self._get_slider_geometry()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(slider_track, 4.0, 4.0)

        # Active filled portion
        filled_track = QRectF(slider_track.left(), slider_track.top(), slider_handle.center().x() - slider_track.left(), slider_track.height())
        painter.setBrush(QBrush(QColor("#059669")))
        painter.drawRoundedRect(filled_track, 4.0, 4.0)

        # Slider Handle
        hover_handle = (self._hover_target == "SLIDER_BETA")
        painter.setPen(QPen(QColor("#a7f3d0"), 1.5))
        painter.setBrush(QBrush(QColor("#34d399") if hover_handle else QColor("#10b981")))
        painter.drawEllipse(slider_handle)

        # Increment / Decrement mini-buttons
        btn_dec = self._get_beta_button_rect(-1)
        btn_zero = self._get_beta_button_rect(0)
        btn_inc = self._get_beta_button_rect(1)

        for btn_rect, label, target_key in [
            (btn_dec, "−0.05", "BETA_DEC"),
            (btn_zero, "β = 0", "BETA_ZERO"),
            (btn_inc, "+0.05", "BETA_INC"),
        ]:
            hover_btn = (self._hover_target == target_key)
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(51, 65, 85, 230) if hover_btn else QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(btn_rect, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, label)

        # ── 2. Spacetime Interval & Invariant Analysis Card ──
        card2_rect = QRectF(hud_left, 152.0, hud_w, 148.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card2_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 156.0, hud_w - 24.0, 18.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "KHOẢNG KHÔNG-THỜI GIAN BẤT BIẾN (Δs²)",
        )

        # Interval classification
        if ds_sq > 0.001:
            class_badge = "THỜI GIAN TÍNH (Timelike)"
            class_color = QColor("#38bdf8")
            proper_text = f"Thời gian riêng Δτ = {math.sqrt(ds_sq):.3f}"
        elif ds_sq < -0.001:
            class_badge = "KHÔNG GIAN TÍNH (Spacelike)"
            class_color = QColor("#ec4899")
            proper_text = f"Khoảng cách riêng Δσ = {math.sqrt(-ds_sq):.3f}"
        else:
            class_badge = "ÁNH SÁNG TÍNH (Lightlike / Null)"
            class_color = QColor("#fbbf24")
            proper_text = "Tia photon (Δs² = 0)"

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(class_color))
        painter.drawText(
            QRectF(hud_left + 12.0, 175.0, hud_w - 24.0, 20.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Δs² = {ds_sq:.3f} = Δs'² = {ds_prime_sq:.3f}",
        )

        # Interval Badge
        badge_rect = QRectF(hud_left + 12.0, 198.0, hud_w - 24.0, 22.0)
        painter.setPen(QPen(class_color, 1.0))
        painter.setBrush(QBrush(QColor(class_color.red(), class_color.green(), class_color.blue(), 35)))
        painter.drawRoundedRect(badge_rect, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"{class_badge} • {proper_text}")

        # Frame S coordinates
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(
            QRectF(hud_left + 12.0, 224.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Hệ S:  E₁({self._event1.x():.1f}, {self._event1.y():.1f})   E₂({self._event2.x():.1f}, {self._event2.y():.1f})   Δx={dx:.2f}, Δct={dct:.2f}",
        )

        # Frame S' coordinates
        painter.setPen(QPen(QColor("#34d399")))
        painter.drawText(
            QRectF(hud_left + 12.0, 242.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Hệ S': E₁'({x1_prime:.1f}, {ct1_prime:.1f})  E₂'({x2_prime:.1f}, {ct2_prime:.1f})  Δx'={dx_prime:.2f}, Δct'={dct_prime:.2f}",
        )

        # Relativity of simultaneity notice
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Italic))
        painter.setPen(QPen(QColor("#94a3b8")))
        if abs(dct) < 0.05 and abs(dx) > 0.1:
            painter.drawText(
                QRectF(hud_left + 12.0, 262.0, hud_w - 24.0, 32.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"Đồng thời trong S nhưng trong S' chênh lệch Δct' = {dct_prime:.2f} (Tính tương đối của sự đồng thời)!",
            )
        else:
            painter.drawText(
                QRectF(hud_left + 12.0, 262.0, hud_w - 24.0, 32.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"Hệ số dãn nở thời gian: Δt = γ·Δτ = {gamma:.2f} | Co ngắn không gian: L = L₀/γ",
            )

        # ── 3. Curriculum Presets (2x2 Grid) ──
        card3_rect = QRectF(hud_left, 307.0, hud_w, 102.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card3_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 310.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "BÀI HỌC THỰC NGHIỆM KINH ĐIỂN",
        )

        presets = [
            ("MUON_ATMOSPHERIC_DECAY", "🌌 Hạt Muon Khí Quyển", 0, 0),
            ("BARN_POLE_PARADOX", "🏚️ Nghịch Lý Cọc-Kho", 0, 1),
            ("TWIN_PARADOX_WORLDLINE", "🚀 Nghịch Lý Sinh Đôi", 1, 0),
            ("LIGHT_CLOCK_GEDANKEN", "⏱️ Đồng Hồ Ánh Sáng", 1, 1),
        ]

        for p_id, p_label, row, col in presets:
            btn_rect = self._get_preset_button_rect(row, col)
            is_active = (self._preset == p_id)
            is_hover = (self._hover_target == f"PRESET_{p_id}")

            if is_active:
                painter.setPen(QPen(QColor("#38bdf8"), 1.5))
                painter.setBrush(QBrush(QColor(14, 165, 233, 70)))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(51, 65, 85, 210) if is_hover else QColor(30, 41, 59, 160)))

            painter.drawRoundedRect(btn_rect, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#e2e8f0")))
            painter.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, p_label)

        # ── 4. Visual Layer Toggles ──
        toggles = [
            ("HYPERBOLAS", "⚡ Hyperbolas", self._show_hyperbolas, 0),
            ("S_GRID", "📐 Lưới S'", self._show_boosted_grid, 1),
            ("LIGHT_CONE", "💡 Nón c", self._show_light_cone, 2),
        ]
        tog_w = (hud_w - 24.0 - 16.0) / 3.0
        for tog_key, tog_label, is_on, idx in toggles:
            tog_rect = QRectF(hud_left + 12.0 + idx * (tog_w + 8.0), 416.0, tog_w, 28.0)
            is_hover = (self._hover_target == f"TOGGLE_{tog_key}")

            if is_on:
                painter.setPen(QPen(QColor("#10b981"), 1.2))
                painter.setBrush(QBrush(QColor(16, 185, 129, 60)))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(51, 65, 85, 200) if is_hover else QColor(30, 41, 59, 140)))

            painter.drawRoundedRect(tog_rect, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#34d399") if is_on else QColor("#94a3b8")))
            painter.drawText(tog_rect, Qt.AlignmentFlag.AlignCenter, tog_label)

        # ── 5. Action Buttons (Stamp & Close) ──
        stamp_btn = self._get_stamp_button_rect()
        hover_stamp = (self._hover_target == "STAMP")
        painter.setPen(QPen(QColor("#0ea5e9"), 1.5))
        painter.setBrush(QBrush(QColor(14, 165, 233, 210) if hover_stamp else QColor(2, 132, 199, 180)))
        painter.drawRoundedRect(stamp_btn, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(stamp_btn, Qt.AlignmentFlag.AlignCenter, "🖊️ Đóng Dấu Vector")

        close_bottom_btn = self._get_close_bottom_button_rect()
        hover_close_b = (self._hover_target == "CLOSE_BOTTOM")
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor(71, 85, 105, 200) if hover_close_b else QColor(51, 65, 85, 160)))
        painter.drawRoundedRect(close_bottom_btn, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(close_bottom_btn, Qt.AlignmentFlag.AlignCenter, "Đóng")

    # ── Geometry & Layout Helpers ─────────────────────────────────────────────

    def _get_close_button_rect(self) -> QRectF:
        """Top-right corner close button."""
        return QRectF(self._width - 32.0, 8.0, 24.0, 24.0)

    def _get_slider_geometry(self) -> tuple[QRectF, QRectF]:
        """Returns (slider_track_rect, slider_handle_rect)."""
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 98.0, 180.0, 8.0)
        # Fraction beta in [0.0, 0.99]
        frac = max(0.0, min(1.0, self._beta / 0.99))
        hx = track.left() + frac * track.width()
        hy = track.center().y()
        handle = QRectF(hx - 7.0, hy - 7.0, 14.0, 14.0)
        return track, handle

    def _get_beta_button_rect(self, delta_step: int) -> QRectF:
        """Buttons next to slider: -1 -> [-0.05], 0 -> [β=0], +1 -> [+0.05]."""
        base_x = 525.0 + 200.0
        y = 93.0
        w = 30.0
        h = 18.0
        if delta_step == -1:
            return QRectF(base_x, y, w, h)
        elif delta_step == 0:
            return QRectF(base_x + 34.0, y, 28.0, h)
        else:
            return QRectF(base_x + 66.0, y, w, h)

    def _get_preset_button_rect(self, row: int, col: int) -> QRectF:
        """2x2 Grid of Curriculum Preset buttons."""
        hud_left = 525.0
        w = 134.0
        h = 28.0
        x = hud_left + 12.0 + col * (w + 8.0)
        y = 332.0 + row * (h + 8.0)
        return QRectF(x, y, w, h)

    def _get_toggle_button_rect(self, idx: int) -> QRectF:
        """Toggle buttons at index 0, 1, 2."""
        hud_left = 525.0
        hud_w = 300.0
        tog_w = (hud_w - 24.0 - 16.0) / 3.0
        return QRectF(hud_left + 12.0 + idx * (tog_w + 8.0), 416.0, tog_w, 28.0)

    def _get_stamp_button_rect(self) -> QRectF:
        """Bottom action button for vector diagram stamping."""
        return QRectF(525.0 + 12.0, 495.0, 175.0, 36.0)

    def _get_close_bottom_button_rect(self) -> QRectF:
        """Bottom action button for closing."""
        return QRectF(525.0 + 195.0, 495.0, 93.0, 36.0)

    def _event_to_scene(self, event_pt: QPointF) -> QPointF:
        """Converts Minkowski (x, ct) coordinates to local item scene coordinates."""
        return QPointF(
            self._origin_scene.x() + event_pt.x() * self._scale,
            self._origin_scene.y() - event_pt.y() * self._scale,
        )

    def _scene_to_event(self, sc_pt: QPointF) -> QPointF:
        """Converts local item scene coordinates to Minkowski (x, ct) coordinates."""
        return QPointF(
            (sc_pt.x() - self._origin_scene.x()) / self._scale,
            -(sc_pt.y() - self._origin_scene.y()) / self._scale,
        )

    # ── Presets & Physical Setters ────────────────────────────────────────────

    def set_beta(self, beta: float) -> None:
        """Sets relative velocity beta = v/c clamped to [0.0, 0.99] and updates gamma."""
        self._beta = max(0.0, min(0.99, float(beta)))
        self._gamma = 1.0 / math.sqrt(max(0.0001, 1.0 - self._beta * self._beta))
        dx = self._event2.x() - self._event1.x()
        dct = self._event2.y() - self._event1.y()
        ds_sq = dct * dct - dx * dx
        self.relativity_updated.emit(self._beta, self._gamma, ds_sq)
        self.update()

    def set_preset(self, preset_name: str) -> None:
        """Switches curriculum preset and applies preset parameters."""
        self._preset = preset_name
        self._apply_preset(preset_name)
        self.preset_changed.emit(preset_name)
        self.update()

    def _apply_preset(self, preset: str) -> None:
        """Configures events, beta, and worldlines according to authentic curriculum scenarios."""
        p_upper = preset.upper()
        if p_upper == "MUON_ATMOSPHERIC_DECAY":
            # Cosmic muon created at high altitude ct = 6.0, decays at surface ct = 0.0
            # beta = 0.995 (gamma = 10.0)
            self._beta = 0.95  # Visualizable high beta
            self._gamma = 1.0 / math.sqrt(max(0.0001, 1.0 - self._beta * self._beta))
            self._event1 = QPointF(0.0, 0.0)      # Detector at Earth surface
            self._event2 = QPointF(0.5, 5.0)      # Muon production in upper atmosphere
            self._worldline_points = [QPointF(0.0, 0.0), QPointF(0.5, 5.0)]

        elif p_upper == "BARN_POLE_PARADOX":
            # Barn of length 4.0 stationary in S. Pole moving at beta = 0.80 (gamma = 1.667)
            self._beta = 0.80
            self._gamma = 1.0 / math.sqrt(max(0.0001, 1.0 - self._beta * self._beta))
            self._event1 = QPointF(0.0, 0.0)      # Front of pole enters barn
            self._event2 = QPointF(4.0, 0.0)      # Back door closes simultaneously in barn frame S
            self._worldline_points = []

        elif p_upper == "TWIN_PARADOX_WORLDLINE":
            # Alice stays on Earth (x=0). Bob rockets to Alpha Centauri and returns.
            self._beta = 0.80
            self._gamma = 1.0 / math.sqrt(max(0.0001, 1.0 - self._beta * self._beta))
            # Outbound to (x=3.2, ct=4.0) with beta = 0.8, then inbound to (x=0, ct=8.0)
            self._event1 = QPointF(0.0, 0.0)
            self._event2 = QPointF(3.2, 4.0)      # Turnaround event
            self._worldline_points = [
                QPointF(0.0, 0.0),
                QPointF(3.2, 4.0),
                QPointF(0.0, 8.0),
            ]

        elif p_upper == "LIGHT_CLOCK_GEDANKEN":
            # Einstein's thought experiment: light clock vertical bounce in rest frame
            self._beta = 0.60
            self._gamma = 1.0 / math.sqrt(max(0.0001, 1.0 - self._beta * self._beta))
            self._event1 = QPointF(0.0, 0.0)      # Photon emitted at bottom mirror
            self._event2 = QPointF(2.4, 4.0)      # Photon hits top mirror in moving frame
            self._worldline_points = [QPointF(0.0, 0.0), QPointF(2.4, 4.0)]

        dx = self._event2.x() - self._event1.x()
        dct = self._event2.y() - self._event1.y()
        self.relativity_updated.emit(self._beta, self._gamma, dct * dct - dx * dx)
        self.events_changed.emit(self._event1.x(), self._event1.y(), self._event2.x(), self._event2.y())

    # ── Mouse & Air Gesture Interaction ───────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles mouse clicks and touchless air pinch grabs on controls."""
        pos = event.pos()

        # Check Close Button
        if self._get_close_button_rect().contains(pos) or self._get_close_bottom_button_rect().contains(pos):
            self.close()
            event.accept()
            return

        # Check Vector Stamp Button
        if self._get_stamp_button_rect().contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        # Check Preset Buttons
        presets = [
            ("MUON_ATMOSPHERIC_DECAY", 0, 0),
            ("BARN_POLE_PARADOX", 0, 1),
            ("TWIN_PARADOX_WORLDLINE", 1, 0),
            ("LIGHT_CLOCK_GEDANKEN", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(pos):
                self.set_preset(p_id)
                event.accept()
                return

        # Check Beta Increments
        if self._get_beta_button_rect(-1).contains(pos):
            self.set_beta(self._beta - 0.05)
            event.accept()
            return
        elif self._get_beta_button_rect(0).contains(pos):
            self.set_beta(0.0)
            event.accept()
            return
        elif self._get_beta_button_rect(1).contains(pos):
            self.set_beta(self._beta + 0.05)
            event.accept()
            return

        # Check Feature Toggles
        if self._get_toggle_button_rect(0).contains(pos):
            self._show_hyperbolas = not self._show_hyperbolas
            self.update()
            event.accept()
            return
        elif self._get_toggle_button_rect(1).contains(pos):
            self._show_boosted_grid = not self._show_boosted_grid
            self.update()
            event.accept()
            return
        elif self._get_toggle_button_rect(2).contains(pos):
            self._show_light_cone = not self._show_light_cone
            self.update()
            event.accept()
            return

        # Check Slider Handle / Track
        track, handle = self._get_slider_geometry()
        if handle.contains(pos) or track.contains(pos):
            self._drag_target = "SLIDER_BETA"
            frac = (pos.x() - track.left()) / track.width()
            self.set_beta(frac * 0.99)
            event.accept()
            return

        # Check Interactive Events E1 and E2
        e1_sc = self._event_to_scene(self._event1)
        if QRectF(e1_sc.x() - 14.0, e1_sc.y() - 14.0, 28.0, 28.0).contains(pos):
            self._drag_target = "EVENT1"
            event.accept()
            return

        e2_sc = self._event_to_scene(self._event2)
        if QRectF(e2_sc.x() - 14.0, e2_sc.y() - 14.0, 28.0, 28.0).contains(pos):
            self._drag_target = "EVENT2"
            event.accept()
            return

        # Top Title Bar Dragging
        if QRectF(0.0, 0.0, self._width, 40.0).contains(pos):
            self._drag_target = "TITLE"
            self._drag_offset = event.scenePos() - self.pos()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles dragging of beta slider, event points, and main card."""
        pos = event.pos()

        if self._drag_target == "TITLE":
            self.setPos(event.scenePos() - self._drag_offset)
            event.accept()
            return

        elif self._drag_target == "SLIDER_BETA":
            track, _ = self._get_slider_geometry()
            frac = (pos.x() - track.left()) / track.width()
            self.set_beta(frac * 0.99)
            event.accept()
            return

        elif self._drag_target == "EVENT1":
            new_event = self._scene_to_event(pos)
            # Clamp to visible Minkowski diagram range
            self._event1 = QPointF(
                max(-6.5, min(6.5, round(new_event.x() * 2.0) / 2.0)),
                max(-6.5, min(6.5, round(new_event.y() * 2.0) / 2.0)),
            )
            dx = self._event2.x() - self._event1.x()
            dct = self._event2.y() - self._event1.y()
            self.relativity_updated.emit(self._beta, self._gamma, dct * dct - dx * dx)
            self.events_changed.emit(self._event1.x(), self._event1.y(), self._event2.x(), self._event2.y())
            self.update()
            event.accept()
            return

        elif self._drag_target == "EVENT2":
            new_event = self._scene_to_event(pos)
            self._event2 = QPointF(
                max(-6.5, min(6.5, round(new_event.x() * 2.0) / 2.0)),
                max(-6.5, min(6.5, round(new_event.y() * 2.0) / 2.0)),
            )
            dx = self._event2.x() - self._event1.x()
            dct = self._event2.y() - self._event1.y()
            self.relativity_updated.emit(self._beta, self._gamma, dct * dct - dx * dx)
            self.events_changed.emit(self._event1.x(), self._event1.y(), self._event2.x(), self._event2.y())
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Releases active drag states."""
        self._drag_target = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        """Tracks hover highlights for buttons and interactive controls."""
        pos = event.pos()
        old_hover = self._hover_target

        if self._get_close_button_rect().contains(pos):
            self._hover_target = "CLOSE"
        elif self._get_close_bottom_button_rect().contains(pos):
            self._hover_target = "CLOSE_BOTTOM"
        elif self._get_stamp_button_rect().contains(pos):
            self._hover_target = "STAMP"
        elif self._get_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_BETA"
        elif self._get_beta_button_rect(-1).contains(pos):
            self._hover_target = "BETA_DEC"
        elif self._get_beta_button_rect(0).contains(pos):
            self._hover_target = "BETA_ZERO"
        elif self._get_beta_button_rect(1).contains(pos):
            self._hover_target = "BETA_INC"
        else:
            self._hover_target = None
            presets = [
                ("MUON_ATMOSPHERIC_DECAY", 0, 0),
                ("BARN_POLE_PARADOX", 0, 1),
                ("TWIN_PARADOX_WORLDLINE", 1, 0),
                ("LIGHT_CLOCK_GEDANKEN", 1, 1),
            ]
            for p_id, r, c in presets:
                if self._get_preset_button_rect(r, c).contains(pos):
                    self._hover_target = f"PRESET_{p_id}"
                    break

        if self._hover_target != old_hover:
            self.update()

    # ── Touchless Air Gesture API ─────────────────────────────────────────────

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        """Handles touchless air pinch gesture for dragging slider or events."""
        local_pt = self.mapFromScene(scene_pt)

        # Check close buttons
        if self._get_close_button_rect().contains(local_pt) or self._get_close_bottom_button_rect().contains(local_pt):
            self.close()
            return True

        # Check stamp button
        if self._get_stamp_button_rect().contains(local_pt):
            self.stamp_to_board()
            return True

        # Check preset buttons
        presets = [
            ("MUON_ATMOSPHERIC_DECAY", 0, 0),
            ("BARN_POLE_PARADOX", 0, 1),
            ("TWIN_PARADOX_WORLDLINE", 1, 0),
            ("LIGHT_CLOCK_GEDANKEN", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(local_pt):
                self.set_preset(p_id)
                return True

        # Check Slider
        track, handle = self._get_slider_geometry()
        if handle.contains(local_pt) or track.contains(local_pt):
            frac = (local_pt.x() - track.left()) / track.width()
            self.set_beta(frac * 0.99)
            return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        """Handles touchless air tap gesture for instant button selection."""
        return self.handle_air_pinch(scene_pt)

    # ── Board Integration & Vector Stamping ───────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps a permanent vector diagram of Minkowski spacetime onto LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = PhysicsTools.create_spacetime_relativity_diagram(
            center=(cx, cy + 30.0),
            width=500.0,
            height=320.0,
            beta=self._beta,
            gamma=self._gamma,
            preset=self._preset,
            event1=(self._event1.x(), self._event1.y()),
            event2=(self._event2.x(), self._event2.y()),
        )

        for item in diagram_items:
            board.addItem(item)

        logger.info(f"Minkowski Spacetime diagram stamped onto LessonBoard (beta={self._beta:.3f}).")

    def close(self) -> None:
        """Closes/hides the interactive relativity item."""
        self.setVisible(False)
        self.update()

    # ── Public Properties for Tests & State Verification ─────────────────────

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def preset(self) -> str:
        return self._preset

    @property
    def event1(self) -> QPointF:
        return self._event1

    @property
    def event2(self) -> QPointF:
        return self._event2

    @property
    def delta_s_squared(self) -> float:
        dx = self._event2.x() - self._event1.x()
        dct = self._event2.y() - self._event1.y()
        return dct * dct - dx * dx
