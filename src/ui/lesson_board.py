"""
AirTeach AI - Lesson Board (QGraphicsScene Canvas)
====================================================
Renders:
  - Document page images as background QGraphicsPixmapItem
  - Live ink strokes as QPainterPath (cubic Bézier, variable width)
  - Recognized text labels and LaTeX PNG overlays
  - Geometric snapped shapes (line, arrow, circle, rectangle, triangle)

Thread safety: all mutating calls must be made from the UI thread.
Use Qt Signals from worker threads to enqueue updates.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
    Signal,
    QObject,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
)

from src.core.stroke_beautifier import BezierSegment, SnappedShape as SnapResult, ShapeType, OneEuroFilter2D

logger = logging.getLogger(__name__)

# ── Color palette ─────────────────────────────────────────────────────────────
_SNAP_COLOR: dict[ShapeType, QColor] = {
    ShapeType.LINE:      QColor("#1a73e8"),
    ShapeType.ARROW:     QColor("#1a73e8"),
    ShapeType.CIRCLE:    QColor("#34a853"),
    ShapeType.ELLIPSE:   QColor("#34a853"),
    ShapeType.RECTANGLE: QColor("#fbbc04"),
    ShapeType.TRIANGLE:  QColor("#ea4335"),
    ShapeType.FREEFORM:  QColor("#202124"),
}

BOARD_WIDTH  = 1920
BOARD_HEIGHT = 1080

# ?? Board Themes & Color Schemes ??????????????????????????????????????????????

class BoardTheme:
    """Classroom whiteboard & chalkboard visual theme modes."""
    WHITE      = "white"        # Classic clean whiteboard (#f8f9fa)
    CHALKBOARD = "chalkboard"   # Classroom dark-green anti-glare board (#1e3a2f)
    OLY        = "oly"          # Vietnamese elementary notebook grid (#fcfbf7, 4-sub-cell 'V? ? ly')
    SLATE      = "slate"        # Modern dark charcoal slate (#181825)


THEME_CONFIGS: dict[str, dict] = {
    BoardTheme.WHITE: {
        "name": "B?ng tr?ng (Whiteboard)",
        "bg_color": QColor("#f8f9fa"),
        "default_ink": QColor("#202124"),
        "grid_pen": QColor("#e8eaed"),
        "dot_pen": QColor("#c5c7d0"),
        "axis_pen": QColor("#5f6368"),
        "axis_text": QColor("#3c4043"),
    },
    BoardTheme.CHALKBOARD: {
        "name": "B?ng xanh ph?n (Chalkboard)",
        "bg_color": QColor("#1e3a2f"),
        "default_ink": QColor("#ffffff"),
        "grid_pen": QColor("#284e3d"),
        "dot_pen": QColor("#2e5b47"),
        "axis_pen": QColor("#80b996"),
        "axis_text": QColor("#d1e7dd"),
    },
    BoardTheme.OLY: {
        "name": "B?ng ? ly ti?u h?c (V? 4 ? ly)",
        "bg_color": QColor("#fcfbf7"),
        "default_ink": QColor("#1a237e"),
        "grid_pen": QColor("#90caf9"),
        "subgrid_pen": QColor("#e3f2fd"),
        "dot_pen": QColor("#90caf9"),
        "axis_pen": QColor("#3f51b5"),
        "axis_text": QColor("#1a237e"),
    },
    BoardTheme.SLATE: {
        "name": "B?ng t?i hi?n ??i (Dark Slate)",
        "bg_color": QColor("#181825"),
        "default_ink": QColor("#cdd6f4"),
        "grid_pen": QColor("#313244"),
        "dot_pen": QColor("#45475a"),
        "axis_pen": QColor("#a6adc8"),
        "axis_text": QColor("#bac2de"),
    },
}



# ── Data class for a completed ink stroke ─────────────────────────────────────

@dataclass
class InkStroke:
    """A fully processed ink stroke ready for rendering."""
    stroke_id: str
    segments: list[BezierSegment]           # Cubic Bézier segments (C¹)
    widths: list[float]                     # per-segment line width
    snap: Optional[SnapResult] = None       # None = render as freehand Bézier
    color: QColor = field(default_factory=lambda: QColor("#202124"))
    timestamps_ms: list[float] = field(default_factory=list)
    hand_id: str = "primary"

    # Graphics item references (populated after addToScene)
    _path_item: Optional[QGraphicsPathItem] = field(default=None, repr=False, compare=False)
    _shape_item: Optional[QGraphicsItem]    = field(default=None, repr=False, compare=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dist_point_to_segment_2d(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Compute Euclidean distance from point (px, py) to line segment (ax, ay)-(bx, by)."""
    dx = bx - ax
    dy = by - ay
    d2 = dx * dx + dy * dy
    if d2 < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _eval_cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> tuple[float, float]:
    """Evaluate cubic Bézier curve at parameter t in [0, 1]."""
    u = 1.0 - t
    u2 = u * u
    u3 = u2 * u
    t2 = t * t
    t3 = t2 * t
    pt = u3 * p0 + 3.0 * u2 * t * p1 + 3.0 * u * t2 * p2 + t3 * p3
    return float(pt[0]), float(pt[1])

def _numpy_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Convert a BGR uint8 numpy array to QPixmap."""
    import cv2  # noqa: PLC0415
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


def _numpy_rgba_to_qpixmap(img_rgba: np.ndarray) -> QPixmap:
    """Convert an RGBA uint8 numpy array to QPixmap with transparency."""
    h, w, ch = img_rgba.shape
    fmt = QImage.Format.Format_RGBA8888
    qi = QImage(img_rgba.data, w, h, w * ch, fmt).copy()
    return QPixmap.fromImage(qi)


def _build_bezier_path(segments: list[BezierSegment]) -> QPainterPath:
    """Assemble a QPainterPath from a list of cubic Bézier segments.
    BezierSegment fields: p0 (start), p1 (control 1), p2 (control 2), p3 (end).
    """
    path = QPainterPath()
    if not segments:
        return path
    p0 = segments[0].p0
    path.moveTo(p0[0], p0[1])
    for seg in segments:
        path.cubicTo(
            seg.p1[0], seg.p1[1],   # control point 1
            seg.p2[0], seg.p2[1],   # control point 2
            seg.p3[0], seg.p3[1],   # end point
        )
    return path


# ── Main Scene ────────────────────────────────────────────────────────────────

class LessonBoard(QGraphicsScene):
    """
    QGraphicsScene that manages one lesson page:
      - Background document page (pixmap)
      - Ink strokes (QPainterPath items)
      - Snapped geometric shapes
      - OCR text / LaTeX overlays
      - Live drawing preview (current stroke being drawn)

    Signals:
        stroke_committed(str): emitted when a finalized stroke is added (stroke_id).
        selection_rect_changed(QRectF): emitted when SELECT gesture defines a region.
    """

    stroke_committed     = Signal(str)
    selection_rect_changed = Signal(QRectF)
    board_theme_changed    = Signal(str, QColor)

    # Z-layer constants
    Z_BACKGROUND  = 0
    Z_INK         = 10
    Z_SHAPE       = 11
    Z_OCR_OVERLAY = 20
    Z_PREVIEW     = 30
    Z_LASER       = 40

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.setSceneRect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        self._board_theme: str = BoardTheme.WHITE
        self.setBackgroundBrush(QBrush(THEME_CONFIGS[self._board_theme]["bg_color"]))

        # State
        self._strokes: dict[str, InkStroke] = {}
        self._bg_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_items: list[QGraphicsItem] = []
        self._select_rect_item: Optional[QGraphicsRectItem] = None

        # Multi-hand concurrent drawing previews
        # hand_id -> dict {"points": list[tuple], "item": QGraphicsPathItem, "color": QColor, "width": float}
        self._live_previews: dict[str, dict] = {}
        # Backwards-compatible aliases for primary hand
        self._live_points: list[tuple[float, float]] = []
        self._preview_item: Optional[QGraphicsPathItem] = None

        # Multi-pointer lasers & eraser reticles: hand_id -> QGraphicsEllipseItem
        self._laser_items: dict[str, QGraphicsEllipseItem] = {}
        self._laser_filters: dict[str, OneEuroFilter2D] = {}
        self._laser_trails: dict[str, Any] = {}
        self._laser_trail_enabled: bool = True
        self._laser_trail_lifetime: float = 1.5
        self._laser_color: str = "#ef4444"
        self._laser_color_idx: int = 0
        self._laser_radius: float = 12.0
        self._eraser_items: dict[str, QGraphicsEllipseItem] = {}
        self._laser_item: Optional[QGraphicsEllipseItem] = None

        # Coordinate grid / axes overlay
        self._grid_mode: str = "none"  # 'none', 'dots', 'grid', 'cartesian'
        self._grid_spacing: int = 40  # pixels
        self._desktop_overlay_mode: bool = False

        # Virtual Measurement Instruments
        self._ruler_item: Optional[QGraphicsItem] = None
        self._protractor_item: Optional[QGraphicsItem] = None
        self._compass_item: Optional[QGraphicsItem] = None

        # Presentation Spotlight Overlay
        self._spotlight_item: Optional[QGraphicsItem] = None

        # Virtual Loupe (Magnifying Glass)
        self._loupe_item: Optional[QGraphicsItem] = None

        # Presentation Countdown Timer & Stopwatch
        self._timer_item: Optional[QGraphicsItem] = None

        # Classroom Metronome & Rhythm Assistant
        self._metronome_item: Optional[QGraphicsItem] = None

        # Interactive STEM Logic Circuit Wires
        self._circuit_wires: list[Any] = []

        # Real-Time Cartesian Oxy Grid Coordinate HUD
        self._coord_hud_item: Optional[QGraphicsItem] = None
        self._coord_hud_enabled: bool = False

        # Interactive Plane Geometry Dynamic Angle Measure
        self._angle_tool_item: Optional[QGraphicsItem] = None

        # Interactive Geometric Triangle Theorem Solver
        self._triangle_tool_item: Optional[QGraphicsItem] = None

        # Interactive Function Graph Plotter & Tangent Line Inspector
        self._function_plotter_item: Optional[QGraphicsItem] = None

        # Interactive Physics Vector Force Decomposer & Inclined Plane
        self._incline_tool_item: Optional[QGraphicsItem] = None

        # Interactive Physics Optics Snell's Law & Refraction Simulator
        self._optics_refraction_item: Optional[QGraphicsItem] = None

        # Interactive Physics Wave Motion & Standing Wave Resonance Simulator
        self._standing_wave_item: Optional[QGraphicsItem] = None

        # Laser pointer dot
        self._init_laser()

    # ── Laser pointer & Multi-pointer Reticles ─────────────────────────────────

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw scene background and optional coordinate grid / axes."""
        super().drawBackground(painter, rect)

        if getattr(self, "_desktop_overlay_mode", False) or self._grid_mode == "none":
            return

        # Do not draw grid over document slides if slide is loaded and visible
        if self._bg_item is not None and self._bg_item.isVisible():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        grid_spacing = self._grid_spacing
        left = int(math.floor(rect.left() / grid_spacing) * grid_spacing)
        top = int(math.floor(rect.top() / grid_spacing) * grid_spacing)
        right = int(math.ceil(rect.right()))
        bottom = int(math.ceil(rect.bottom()))

        theme_cfg = THEME_CONFIGS.get(getattr(self, "_board_theme", BoardTheme.WHITE), THEME_CONFIGS[BoardTheme.WHITE])

        if self._grid_mode == "dots":
            dot_pen = QPen(theme_cfg["dot_pen"], 2.0)
            painter.setPen(dot_pen)
            for x in range(left, right + 1, grid_spacing):
                for y in range(top, bottom + 1, grid_spacing):
                    painter.drawPoint(x, y)

        elif self._grid_mode in ("grid", "cartesian"):
            if getattr(self, "_board_theme", BoardTheme.WHITE) == BoardTheme.OLY:
                # Authentic Vietnamese Elementary 4-sub-cell grid (V? 4 ? ly)
                sub_pen = QPen(theme_cfg.get("subgrid_pen", QColor("#e3f2fd")), 0.75)
                main_pen = QPen(theme_cfg["grid_pen"], 1.4)
                sub_step = max(5, grid_spacing // 4)

                # Faint sub-lines
                painter.setPen(sub_pen)
                for x in range(left, right + 1, sub_step):
                    if (x - left) % grid_spacing != 0:
                        painter.drawLine(x, top, x, bottom)
                for y in range(top, bottom + 1, sub_step):
                    if (y - top) % grid_spacing != 0:
                        painter.drawLine(left, y, right, y)

                # Primary grid lines
                painter.setPen(main_pen)
                for x in range(left, right + 1, grid_spacing):
                    painter.drawLine(x, top, x, bottom)
                for y in range(top, bottom + 1, grid_spacing):
                    painter.drawLine(left, y, right, y)
            else:
                grid_pen = QPen(theme_cfg["grid_pen"], 1.0)
                painter.setPen(grid_pen)
                # Vertical lines
                for x in range(left, right + 1, grid_spacing):
                    painter.drawLine(x, top, x, bottom)
                # Horizontal lines
                for y in range(top, bottom + 1, grid_spacing):
                    painter.drawLine(left, y, right, y)

            if self._grid_mode == "cartesian":
                # Primary coordinate axes Oxy
                origin_x = BOARD_WIDTH / 2.0
                origin_y = BOARD_HEIGHT / 2.0

                axis_pen = QPen(theme_cfg["axis_pen"], 2.0)
                painter.setPen(axis_pen)

                # X-axis
                painter.drawLine(0, int(origin_y), BOARD_WIDTH, int(origin_y))
                # Y-axis
                painter.drawLine(int(origin_x), 0, int(origin_x), BOARD_HEIGHT)

                # Axis arrows
                arrow_size = 10.0
                # X arrow
                painter.drawLine(
                    int(BOARD_WIDTH - arrow_size), int(origin_y - arrow_size / 2),
                    BOARD_WIDTH, int(origin_y)
                )
                painter.drawLine(
                    int(BOARD_WIDTH - arrow_size), int(origin_y + arrow_size / 2),
                    BOARD_WIDTH, int(origin_y)
                )
                # Y arrow
                painter.drawLine(
                    int(origin_x - arrow_size / 2), int(arrow_size),
                    int(origin_x), 0
                )
                painter.drawLine(
                    int(origin_x + arrow_size / 2), int(arrow_size),
                    int(origin_x), 0
                )

                # Axis labels
                painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                painter.setPen(QPen(theme_cfg["axis_text"]))
                painter.drawText(int(BOARD_WIDTH - 24), int(origin_y + 20), "x")
                painter.drawText(int(origin_x + 10), 22, "y")
                painter.drawText(int(origin_x - 18), int(origin_y + 18), "O")

        painter.restore()

    def set_desktop_overlay_mode(self, enabled: bool) -> None:
        """Switch between standard whiteboard canvas and transparent desktop overlay."""
        self._desktop_overlay_mode = bool(enabled)
        if self._bg_item is not None:
            self._bg_item.setVisible(not enabled)
        if enabled:
            from PySide6.QtGui import QBrush, QColor
            self.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
        else:
            from PySide6.QtGui import QBrush
            theme_cfg = THEME_CONFIGS.get(getattr(self, "_board_theme", BoardTheme.WHITE), THEME_CONFIGS[BoardTheme.WHITE])
            self.setBackgroundBrush(QBrush(theme_cfg["bg_color"]))
        self.update()

    def is_desktop_overlay_mode(self) -> bool:
        """Return True if board is currently rendering as a transparent desktop overlay."""
        return getattr(self, "_desktop_overlay_mode", False)

    def set_grid_mode(self, mode: str) -> None:
        """Set grid display mode: 'none', 'dots', 'grid', 'cartesian'."""
        if mode in ("none", "dots", "grid", "cartesian"):
            self._grid_mode = mode
            self.update()

    def grid_mode(self) -> str:
        """Get active grid display mode."""
        return self._grid_mode


    # ?? Classroom Board Theme Methods ??????????????????????????????????????????

    def set_board_theme(self, theme: str) -> None:
        """Set board background theme: 'white', 'chalkboard', 'oly', 'slate'."""
        if theme not in THEME_CONFIGS:
            return
        self._board_theme = theme
        if not self.is_desktop_overlay_mode():
            self.setBackgroundBrush(QBrush(THEME_CONFIGS[theme]["bg_color"]))
        self.update()
        self.board_theme_changed.emit(theme, THEME_CONFIGS[theme]["default_ink"])

    def board_theme(self) -> str:
        """Get current board background theme name ('white', 'chalkboard', 'oly', 'slate')."""
        return getattr(self, "_board_theme", BoardTheme.WHITE)

    def get_board_theme(self) -> str:
        """Alias for board_theme()."""
        return self.board_theme()

    def get_theme_config(self) -> dict:
        """Return theme dictionary configuration for active theme."""
        return THEME_CONFIGS.get(getattr(self, "_board_theme", BoardTheme.WHITE), THEME_CONFIGS[BoardTheme.WHITE])

    def cycle_board_theme(self) -> str:
        """Cycle through board themes: white -> chalkboard -> oly -> slate -> white."""
        themes = [BoardTheme.WHITE, BoardTheme.CHALKBOARD, BoardTheme.OLY, BoardTheme.SLATE]
        cur_theme = self.board_theme()
        cur_idx = themes.index(cur_theme) if cur_theme in themes else 0
        next_theme = themes[(cur_idx + 1) % len(themes)]
        self.set_board_theme(next_theme)
        return next_theme

    def toggle_grid(self) -> str:
        """Cycle through grid display modes: none -> grid -> cartesian -> dots -> none."""
        modes = ["none", "grid", "cartesian", "dots"]
        cur_idx = modes.index(self._grid_mode) if self._grid_mode in modes else 0
        next_mode = modes[(cur_idx + 1) % len(modes)]
        self.set_grid_mode(next_mode)
        return next_mode

    # ── Floating Acrylic Sticky Notes ──────────────────────────────────────────

    def add_sticky_note(
        self,
        title: str = "Ghi chú",
        text: str = "",
        theme: str = "yellow",
        x: float = 120.0,
        y: float = 120.0,
        is_collapsed: bool = False,
        note_id: Optional[str] = None,
    ) -> Any:
        """Add an acrylic floating sticky note to the scene."""
        from src.ui.sticky_note_item import StickyNoteItem
        note = StickyNoteItem(
            title=title,
            text=text,
            theme=theme,
            x=x,
            y=y,
            is_collapsed=is_collapsed,
            note_id=note_id,
        )
        self.addItem(note)
        if not hasattr(self, "_sticky_notes"):
            self._sticky_notes = []
        self._sticky_notes.append(note)
        note.note_deleted.connect(lambda nid: self._remove_sticky_note_by_id(nid))
        return note

    def get_sticky_notes(self) -> list[Any]:
        """Return all sticky notes currently attached to this scene."""
        return [n for n in getattr(self, "_sticky_notes", []) if n.scene() is self]

    def _remove_sticky_note_by_id(self, nid: str) -> None:
        """Remove sticky note record when deleted from UI."""
        if hasattr(self, "_sticky_notes"):
            self._sticky_notes = [n for n in self._sticky_notes if getattr(n, "note_id", None) != nid]

    def clear_sticky_notes(self) -> None:
        """Remove all sticky notes from the scene."""
        for note in self.get_sticky_notes():
            self.removeItem(note)
        self._sticky_notes = []

    # ── Presentation Countdown Timer & Stopwatch ───────────────────────────────

    def show_timer(self, visible: bool = True, x: float = 1640.0, y: float = 40.0, duration_sec: int = 300) -> None:
        """Show or hide the presentation countdown timer overlay on LessonBoard."""
        if self._timer_item is None:
            from src.ui.presentation_timer import PresentationTimerItem
            self._timer_item = PresentationTimerItem(duration_sec=duration_sec)
            self._timer_item.setPos(x, y)
            self.addItem(self._timer_item)
        self._timer_item.setVisible(visible)
        if visible:
            self._timer_item.setPos(x, y)

    def toggle_timer(self) -> bool:
        """Toggle timer overlay visibility; returns new visibility state."""
        if self._timer_item is None or not self._timer_item.isVisible():
            self.show_timer(True)
            return True
        else:
            self.show_timer(False)
            return False

    def is_timer_visible(self) -> bool:
        """Return True if presentation timer is currently active on the board."""
        return self._timer_item is not None and self._timer_item.isVisible()

    def start_timer(self) -> None:
        """Start running the presentation timer."""
        if self._timer_item is not None and hasattr(self._timer_item, "start"):
            self._timer_item.start()

    def pause_timer(self) -> None:
        """Pause running the presentation timer."""
        if self._timer_item is not None and hasattr(self._timer_item, "pause"):
            self._timer_item.pause()

    def reset_timer(self) -> None:
        """Reset presentation timer to default preset."""
        if self._timer_item is not None and hasattr(self._timer_item, "reset"):
            self._timer_item.reset()

    # ── Classroom Metronome & Rhythm Assistant ────────────────────────────────

    def show_metronome(
        self,
        visible: bool = True,
        x: float = 40.0,
        y: float = 140.0,
        bpm: int = 120,
        beats_per_bar: int = 4,
    ) -> None:
        """Show or hide the classroom metronome overlay on LessonBoard."""
        if self._metronome_item is None:
            from src.ui.classroom_metronome import ClassroomMetronomeItem
            self._metronome_item = ClassroomMetronomeItem(bpm=bpm, beats_per_bar=beats_per_bar)
            self._metronome_item.setPos(x, y)
            self.addItem(self._metronome_item)
        self._metronome_item.setVisible(visible)
        if visible:
            self._metronome_item.setPos(x, y)

    def toggle_metronome(self) -> bool:
        """Toggle metronome overlay visibility; returns new visibility state."""
        if self._metronome_item is None or not self._metronome_item.isVisible():
            self.show_metronome(True)
            return True
        else:
            if hasattr(self._metronome_item, "stop"):
                self._metronome_item.stop()
            self.show_metronome(False)
            return False

    def is_metronome_visible(self) -> bool:
        """Return True if metronome is currently active on the board."""
        return self._metronome_item is not None and self._metronome_item.isVisible()

    def get_metronome_item(self) -> Optional[QGraphicsItem]:
        """Return metronome item instance if created."""
        return self._metronome_item

    # ── Virtual Loupe (Magnifying Glass) ───────────────────────────────────────

    def enable_loupe(self, enabled: bool = True, zoom: float = 2.0, radius: float = 115.0) -> None:
        """Enable or disable virtual loupe magnifying glass on LessonBoard."""
        if self._loupe_item is None:
            from src.ui.virtual_loupe import VirtualLoupeItem
            self._loupe_item = VirtualLoupeItem(BOARD_WIDTH, BOARD_HEIGHT, radius=radius, zoom_factor=zoom)
            self.addItem(self._loupe_item)
        if hasattr(self._loupe_item, "set_zoom_factor"):
            self._loupe_item.set_zoom_factor(zoom)
        if hasattr(self._loupe_item, "set_radius"):
            self._loupe_item.set_radius(radius)
        self._loupe_item.setVisible(enabled)

    def move_loupe(self, x: float, y: float) -> None:
        """Update loupe center focus coordinates."""
        if self._loupe_item is not None and hasattr(self._loupe_item, "set_loupe_pos"):
            self._loupe_item.set_loupe_pos(x, y)

    def toggle_loupe(self) -> bool:
        """Toggle virtual loupe mode; returns new visibility state."""
        if self._loupe_item is None or not self._loupe_item.isVisible():
            self.enable_loupe(True)
            return True
        else:
            self.enable_loupe(False)
            return False

    def is_loupe_enabled(self) -> bool:
        """Return True if virtual loupe is currently active."""
        return self._loupe_item is not None and self._loupe_item.isVisible()

    def set_loupe_zoom(self, zoom: float) -> None:
        """Change loupe magnification factor."""
        if self._loupe_item is not None and hasattr(self._loupe_item, "set_zoom_factor"):
            self._loupe_item.set_zoom_factor(zoom)

    # ── Presentation Spotlight Overlay ─────────────────────────────────────────

    def enable_spotlight(self, enabled: bool = True, radius: float = 135.0) -> None:
        """Enable or disable presentation spotlight mode on LessonBoard."""
        if self._spotlight_item is None:
            from src.ui.spotlight_overlay import SpotlightOverlayItem
            self._spotlight_item = SpotlightOverlayItem(BOARD_WIDTH, BOARD_HEIGHT, initial_radius=radius)
            self.addItem(self._spotlight_item)
        if hasattr(self._spotlight_item, "set_radius"):
            self._spotlight_item.set_radius(radius)
        self._spotlight_item.setVisible(enabled)

    def move_spotlight(self, x: float, y: float) -> None:
        """Update spotlight focus coordinates."""
        if self._spotlight_item is not None and hasattr(self._spotlight_item, "set_spotlight_pos"):
            self._spotlight_item.set_spotlight_pos(x, y)

    def toggle_spotlight(self) -> bool:
        """Toggle spotlight mode; returns new visibility state."""
        if self._spotlight_item is None or not self._spotlight_item.isVisible():
            self.enable_spotlight(True)
            return True
        else:
            self.enable_spotlight(False)
            return False

    def is_spotlight_enabled(self) -> bool:
        """Return True if presentation spotlight is currently active."""
        return self._spotlight_item is not None and self._spotlight_item.isVisible()

    def set_spotlight_radius(self, radius: float) -> None:
        """Change spotlight aperture size."""
        if self._spotlight_item is not None and hasattr(self._spotlight_item, "set_radius"):
            self._spotlight_item.set_radius(radius)

    # ── Virtual Measurement Instruments ───────────────────────────────────────

    def show_ruler(self, visible: bool = True, x: float = 300.0, y: float = 250.0) -> None:
        """Show or hide the virtual measurement ruler on the board."""
        if self._ruler_item is None:
            from src.ui.measurement_tools import VirtualRulerItem
            self._ruler_item = VirtualRulerItem(length_cm=15.0, pixels_per_cm=40.0)
            self._ruler_item.setPos(x, y)
            self.addItem(self._ruler_item)
        self._ruler_item.setVisible(visible)
        if visible:
            self._ruler_item.setPos(x, y)

    def toggle_ruler(self) -> bool:
        """Toggle virtual ruler visibility; returns new visibility state."""
        if self._ruler_item is None or not self._ruler_item.isVisible():
            self.show_ruler(True)
            return True
        else:
            self.show_ruler(False)
            return False

    def is_ruler_visible(self) -> bool:
        """Return True if virtual ruler is currently active on the board."""
        return self._ruler_item is not None and self._ruler_item.isVisible()

    def show_protractor(self, visible: bool = True, x: float = 600.0, y: float = 400.0) -> None:
        """Show or hide the virtual angle protractor on the board."""
        if self._protractor_item is None:
            from src.ui.measurement_tools import VirtualProtractorItem
            self._protractor_item = VirtualProtractorItem(radius=220.0)
            self._protractor_item.setPos(x, y)
            self.addItem(self._protractor_item)
        self._protractor_item.setVisible(visible)
        if visible:
            self._protractor_item.setPos(x, y)

    def toggle_protractor(self) -> bool:
        """Toggle virtual protractor visibility; returns new visibility state."""
        if self._protractor_item is None or not self._protractor_item.isVisible():
            self.show_protractor(True)
            return True
        else:
            self.show_protractor(False)
            return False

    def is_protractor_visible(self) -> bool:
        """Return True if virtual protractor is currently active on the board."""
        return self._protractor_item is not None and self._protractor_item.isVisible()

    # ── Virtual Geometric Compass (Compa) ──────────────────────────────────────

    def show_compass(self, visible: bool = True, x: float = 800.0, y: float = 400.0, radius_cm: float = 3.5) -> None:
        """Show or hide the virtual geometric compass on LessonBoard."""
        if self._compass_item is None:
            from src.ui.measurement_tools import VirtualCompassItem
            self._compass_item = VirtualCompassItem(radius_cm=radius_cm)
            self._compass_item.setPos(x, y)
            self.addItem(self._compass_item)
        self._compass_item.setVisible(visible)
        if visible:
            self._compass_item.setPos(x, y)

    def toggle_compass(self) -> bool:
        """Toggle compass overlay visibility; returns new visibility state."""
        if self._compass_item is None or not self._compass_item.isVisible():
            self.show_compass(True)
            return True
        else:
            self.show_compass(False)
            return False

    def is_compass_visible(self) -> bool:
        """Return True if virtual compass is currently active on the board."""
        return self._compass_item is not None and self._compass_item.isVisible()

    def get_compass_item(self) -> Optional[QGraphicsItem]:
        """Return the virtual compass item instance if created."""
        return self._compass_item

    # ── Interactive STEM Logic Circuit Wires ───────────────────────────────────

    def connect_logic_gates(
        self,
        source_gate: Any,
        target_gate: Any,
        target_pin: str = "A",
    ) -> Any:
        """
        Create and route an orthogonal Manhattan wire connecting source_gate to target_gate.
        Propagates live boolean signals and updates geometry dynamically on gate drag.
        """
        from src.ui.interactive_circuit_wire import InteractiveCircuitWireItem

        wire = InteractiveCircuitWireItem(
            source_gate=source_gate,
            target_gate=target_gate,
            target_pin=target_pin,
        )
        self.addItem(wire)
        self._circuit_wires.append(wire)

        def _on_wire_disconnected() -> None:
            if wire in self._circuit_wires:
                self._circuit_wires.remove(wire)

        wire.disconnected.connect(_on_wire_disconnected)
        return wire

    def get_circuit_wires(self) -> list[Any]:
        """Return a copy of the list of active circuit wires on the board."""
        return list(self._circuit_wires)

    def remove_circuit_wire(self, wire: Any) -> bool:
        """Disconnect and remove a specific circuit wire from the board."""
        if wire in self._circuit_wires:
            wire.disconnect_wire()
            if wire in self._circuit_wires:
                self._circuit_wires.remove(wire)
            return True
        return False

    def clear_circuit_wires(self) -> None:
        """Disconnect and remove all active circuit wires."""
        for wire in list(self._circuit_wires):
            wire.disconnect_wire()
        self._circuit_wires.clear()

    # ── Real-Time Cartesian Oxy Grid Coordinate HUD ───────────────────────────

    def show_coordinate_hud(
        self,
        visible: bool = True,
        snap_to_grid: bool = False,
    ) -> None:
        """Show or hide the real-time Cartesian Oxy coordinate HUD."""
        if self._coord_hud_item is None:
            from src.ui.cartesian_coordinate_hud import CartesianCoordinateHudItem
            self._coord_hud_item = CartesianCoordinateHudItem(
                origin=(BOARD_WIDTH / 2.0, BOARD_HEIGHT / 2.0),
                grid_spacing=float(self._grid_spacing),
                snap_to_grid=snap_to_grid,
            )
            self.addItem(self._coord_hud_item)

        self._coord_hud_item.setVisible(visible)
        if visible and snap_to_grid is not None:
            if hasattr(self._coord_hud_item, "set_snap_to_grid"):
                self._coord_hud_item.set_snap_to_grid(snap_to_grid)
        self.update()

    def toggle_coordinate_hud(self) -> bool:
        """Toggle coordinate HUD visibility; returns new state."""
        if self._coord_hud_item is None or not self._coord_hud_item.isVisible():
            self.show_coordinate_hud(True)
            return True
        else:
            self.show_coordinate_hud(False)
            return False

    def is_coordinate_hud_visible(self) -> bool:
        """Return True if Cartesian coordinate HUD is currently active on canvas."""
        return self._coord_hud_item is not None and self._coord_hud_item.isVisible()

    def set_coordinate_hud_snap(self, enabled: bool) -> None:
        """Enable or disable magnetic grid snapping in coordinate HUD."""
        if self._coord_hud_item is None and enabled:
            self.show_coordinate_hud(True, snap_to_grid=True)
        elif self._coord_hud_item is not None and hasattr(self._coord_hud_item, "set_snap_to_grid"):
            self._coord_hud_item.set_snap_to_grid(enabled)

    def is_coordinate_hud_snap(self) -> bool:
        """Return True if coordinate HUD grid snapping is active."""
        if self._coord_hud_item is not None and hasattr(self._coord_hud_item, "snap_to_grid"):
            return self._coord_hud_item.snap_to_grid()
        return False

    def toggle_coordinate_hud_snap(self) -> bool:
        """Toggle coordinate HUD grid snapping; returns new state."""
        if self._coord_hud_item is None:
            self.show_coordinate_hud(True, snap_to_grid=True)
            return True
        elif hasattr(self._coord_hud_item, "toggle_snap_to_grid"):
            return self._coord_hud_item.toggle_snap_to_grid()
        return False

    def update_coordinate_hud(self, x: float, y: float) -> None:
        """Update floating HUD pointer position."""
        if self._coord_hud_item is not None and self._coord_hud_item.isVisible():
            if hasattr(self._coord_hud_item, "set_pointer_pos"):
                self._coord_hud_item.set_pointer_pos(x, y)

    def get_coordinate_hud_item(self) -> Optional[QGraphicsItem]:
        """Return coordinate HUD instance if created."""
        return self._coord_hud_item

    # ── Interactive Plane Geometry Dynamic Angle Measure ───────────────────────

    def show_angle_tool(
        self,
        visible: bool = True,
        apex: tuple[float, float] = (800.0, 500.0),
        arm_a: tuple[float, float] = (620.0, 360.0),
        arm_c: tuple[float, float] = (1020.0, 500.0),
    ) -> None:
        """Show or hide the interactive dynamic angle measurement tool."""
        if self._angle_tool_item is None:
            from src.ui.interactive_angle_tool import InteractiveAngleMeasureItem
            self._angle_tool_item = InteractiveAngleMeasureItem(
                apex_b=apex,
                arm_a=arm_a,
                arm_c=arm_c,
            )
            self.addItem(self._angle_tool_item)

        self._angle_tool_item.setVisible(visible)
        if visible and hasattr(self._angle_tool_item, "set_vertices"):
            self._angle_tool_item.set_vertices(arm_a, apex, arm_c)
        self.update()

    def toggle_angle_tool(self) -> bool:
        """Toggle dynamic angle measurement tool visibility; returns new state."""
        if self._angle_tool_item is None or not self._angle_tool_item.isVisible():
            self.show_angle_tool(True)
            return True
        else:
            self.show_angle_tool(False)
            return False

    def is_angle_tool_visible(self) -> bool:
        """Return True if interactive angle measure tool is active on board."""
        return self._angle_tool_item is not None and self._angle_tool_item.isVisible()

    def get_angle_tool_item(self) -> Optional[QGraphicsItem]:
        """Return interactive angle tool instance if created."""
        return self._angle_tool_item

    # ── Interactive Geometric Triangle Theorem Solver ──────────────────────────

    def show_triangle_tool(
        self,
        visible: bool = True,
        vert_a: tuple[float, float] = (750.0, 320.0),
        vert_b: tuple[float, float] = (550.0, 620.0),
        vert_c: tuple[float, float] = (1050.0, 620.0),
    ) -> None:
        """Show or hide the interactive geometric triangle theorem tool."""
        if self._triangle_tool_item is None:
            from src.ui.interactive_triangle_tool import InteractiveTriangleGeometryItem
            self._triangle_tool_item = InteractiveTriangleGeometryItem(
                vert_a=vert_a,
                vert_b=vert_b,
                vert_c=vert_c,
            )
            self.addItem(self._triangle_tool_item)

        self._triangle_tool_item.setVisible(visible)
        if visible and hasattr(self._triangle_tool_item, "set_vertices"):
            self._triangle_tool_item.set_vertices(vert_a, vert_b, vert_c)
        self.update()

    def toggle_triangle_tool(self) -> bool:
        """Toggle triangle theorem tool visibility; returns new state."""
        if self._triangle_tool_item is None or not self._triangle_tool_item.isVisible():
            self.show_triangle_tool(True)
            return True
        else:
            self.show_triangle_tool(False)
            return False

    def is_triangle_tool_visible(self) -> bool:
        """Return True if interactive triangle solver is active on board."""
        return self._triangle_tool_item is not None and self._triangle_tool_item.isVisible()

    def get_triangle_tool_item(self) -> Optional[QGraphicsItem]:
        """Return interactive triangle solver instance if created."""
        return self._triangle_tool_item

    # ── Interactive Function Graph Plotter & Tangent Line Inspector ────────────

    def show_function_plotter(
        self,
        visible: bool = True,
        expr: str = "0.25 * x**3 - x",
        probe_x: float = 1.0,
    ) -> None:
        """Show or hide the interactive function plotter item."""
        if self._function_plotter_item is None:
            from src.ui.interactive_function_plotter import InteractiveFunctionPlotterItem
            self._function_plotter_item = InteractiveFunctionPlotterItem(
                expr=expr,
                origin=(BOARD_WIDTH / 2.0, BOARD_HEIGHT / 2.0),
                grid_step=float(self._grid_spacing),
                probe_x=probe_x,
            )
            self.addItem(self._function_plotter_item)

        self._function_plotter_item.setVisible(visible)
        if visible and hasattr(self._function_plotter_item, "set_expression"):
            self._function_plotter_item.set_expression(expr)
            self._function_plotter_item.set_probe_x(probe_x)
        self.update()

    def toggle_function_plotter(self) -> bool:
        """Toggle interactive function plotter visibility; returns new state."""
        if self._function_plotter_item is None or not self._function_plotter_item.isVisible():
            self.show_function_plotter(True)
            return True
        else:
            self.show_function_plotter(False)
            return False

    def is_function_plotter_visible(self) -> bool:
        """Return True if interactive function plotter is active on board."""
        return self._function_plotter_item is not None and self._function_plotter_item.isVisible()

    def get_function_plotter_item(self) -> Optional[QGraphicsItem]:
        """Return interactive function plotter instance if created."""
        return self._function_plotter_item

    # ── Interactive Physics Vector Force Decomposer & Inclined Plane ───────────

    def show_inclined_plane(
        self,
        visible: bool = True,
        alpha: float = 30.0,
        mass: float = 2.0,
        mu: float = 0.25,
    ) -> None:
        """Show or hide the interactive physics inclined plane item."""
        if self._incline_tool_item is None:
            from src.ui.interactive_physics_incline import InteractiveInclinedPlaneItem
            self._incline_tool_item = InteractiveInclinedPlaneItem(
                base_pos=(700.0, 720.0),
                base_length=420.0,
                alpha_deg=alpha,
                mass_kg=mass,
                mu_friction=mu,
            )
            self.addItem(self._incline_tool_item)

        self._incline_tool_item.setVisible(visible)
        if visible and hasattr(self._incline_tool_item, "set_alpha"):
            self._incline_tool_item.set_alpha(alpha)
            self._incline_tool_item.set_mass(mass)
            self._incline_tool_item.set_friction_coeff(mu)
        self.update()

    def toggle_inclined_plane(self) -> bool:
        """Toggle interactive inclined plane visibility; returns new state."""
        if self._incline_tool_item is None or not self._incline_tool_item.isVisible():
            self.show_inclined_plane(True)
            return True
        else:
            self.show_inclined_plane(False)
            return False

    def is_inclined_plane_visible(self) -> bool:
        """Return True if interactive inclined plane is active on board."""
        return self._incline_tool_item is not None and self._incline_tool_item.isVisible()

    def get_inclined_plane_item(self) -> Optional[QGraphicsItem]:
        """Return interactive inclined plane instance if created."""
        return self._incline_tool_item

    # ── Interactive Physics Optics Snell's Law & Refraction Simulator ──────────

    def show_optics_refraction(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
        n1: float = 1.000,
        n2: float = 1.333,
    ) -> None:
        """Show or hide the interactive physics optics refraction simulation item."""
        if self._optics_refraction_item is None:
            from src.ui.interactive_optics_refraction import InteractiveOpticsRefractionItem
            spawn_pos = pos if pos is not None else (540.0, 320.0)
            self._optics_refraction_item = InteractiveOpticsRefractionItem(
                pos=spawn_pos,
                n1=n1,
                n2=n2,
            )
            self.addItem(self._optics_refraction_item)

        self._optics_refraction_item.setVisible(visible)
        if visible and pos is not None:
            self._optics_refraction_item.setPos(pos[0], pos[1])
        self.update()

    def toggle_optics_refraction(self) -> bool:
        """Toggle interactive optics refraction visibility; returns new state."""
        if self._optics_refraction_item is None or not self._optics_refraction_item.isVisible():
            self.show_optics_refraction(True)
            return True
        else:
            self.show_optics_refraction(False)
            return False

    def is_optics_refraction_visible(self) -> bool:
        """Return True if interactive optics refraction simulator is active on board."""
        return self._optics_refraction_item is not None and self._optics_refraction_item.isVisible()

    def get_optics_refraction_item(self) -> Optional[QGraphicsItem]:
        """Return interactive optics refraction instance if created."""
        return self._optics_refraction_item

    # ── Interactive Physics Wave Motion & Standing Wave Resonance Simulator ───

    def show_standing_wave(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
        harmonic_k: int = 3,
        amplitude: float = 35.0,
    ) -> None:
        """Show or hide the interactive physics standing wave simulation item."""
        if self._standing_wave_item is None:
            from src.ui.interactive_standing_wave import InteractiveStandingWaveItem
            spawn_pos = pos if pos is not None else (520.0, 340.0)
            self._standing_wave_item = InteractiveStandingWaveItem(
                pos=spawn_pos,
                harmonic_k=harmonic_k,
                amplitude=amplitude,
            )
            self.addItem(self._standing_wave_item)

        self._standing_wave_item.setVisible(visible)
        if visible and pos is not None:
            self._standing_wave_item.setPos(pos[0], pos[1])
        self.update()

    def toggle_standing_wave(self) -> bool:
        """Toggle interactive standing wave simulator visibility; returns new state."""
        if self._standing_wave_item is None or not self._standing_wave_item.isVisible():
            self.show_standing_wave(True)
            return True
        else:
            self.show_standing_wave(False)
            return False

    def is_standing_wave_visible(self) -> bool:
        """Return True if interactive standing wave simulator is active on board."""
        return self._standing_wave_item is not None and self._standing_wave_item.isVisible()

    def get_standing_wave_item(self) -> Optional[QGraphicsItem]:
        """Return interactive standing wave instance if created."""
        return self._standing_wave_item

    def set_grid_spacing(self, spacing: int) -> None:
        """Set grid spacing in pixels (clamped between 15 and 200)."""
        self._grid_spacing = max(15, min(200, int(spacing)))
        if self._coord_hud_item is not None and hasattr(self._coord_hud_item, "set_grid_spacing"):
            self._coord_hud_item.set_grid_spacing(float(self._grid_spacing))
        if self._function_plotter_item is not None and hasattr(self._function_plotter_item, "set_grid_step"):
            self._function_plotter_item.set_grid_step(float(self._grid_spacing))
        self.update()

    def grid_spacing(self) -> int:
        """Get active grid spacing in pixels."""
        return self._grid_spacing

    def _init_laser(self) -> None:
        r = getattr(self, "_laser_radius", 12.0)
        c_str = getattr(self, "_laser_color", "#ef4444")
        self._laser_item = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        c = QColor(c_str)
        pen = QPen(c, 2)
        self._laser_item.setPen(pen)
        self._laser_item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 80)))
        self._laser_item.setZValue(self.Z_LASER)
        self._laser_item.setVisible(False)
        self.addItem(self._laser_item)
        self._laser_items["primary"] = self._laser_item

    def laser_color(self) -> str:
        """Return active laser pointer hex color."""
        return getattr(self, "_laser_color", "#ef4444")

    def set_laser_color(self, color: str) -> None:
        """Set laser pointer hex color and update all active laser dots and trails."""
        self._laser_color = str(color)
        c = QColor(self._laser_color)
        for item in self._laser_items.values():
            if item is not None:
                item.setPen(QPen(c, 2))
                item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 80)))
        for trail in self._laser_trails.values():
            if trail is not None and hasattr(trail, "set_color"):
                trail.set_color(self._laser_color)

    def laser_radius(self) -> float:
        """Return active laser pointer radius in pixels."""
        return getattr(self, "_laser_radius", 12.0)

    def set_laser_radius(self, radius: float) -> None:
        """Set laser pointer radius in pixels and update active items."""
        r = max(4.0, min(36.0, float(radius)))
        self._laser_radius = r
        for item in self._laser_items.values():
            if item is not None:
                item.setRect(-r, -r, 2 * r, 2 * r)

    def cycle_laser_color(self) -> tuple[str, str, str]:
        """Cycle through laser color presets: Ruby -> Emerald -> Cyan -> Amber -> Violet."""
        presets = [
            ("#ef4444", "Đỏ Ruby", "🔴"),
            ("#10b981", "Xanh Lục Bảo", "🟢"),
            ("#06b6d4", "Xanh Cyan Dạ Quang", "🔵"),
            ("#f59e0b", "Vàng Hổ Phách", "🟡"),
            ("#8b5cf6", "Tím Dạ Quang", "🟣"),
        ]
        if not hasattr(self, "_laser_color_idx"):
            self._laser_color_idx = 0
        self._laser_color_idx = (self._laser_color_idx + 1) % len(presets)
        hex_c, name, emoji = presets[self._laser_color_idx]
        self.set_laser_color(hex_c)
        return hex_c, name, emoji

    def move_laser(
        self,
        x: float,
        y: float,
        visible: bool = True,
        hand_id: str = "primary",
        color: QColor | str = "#ff0000",
        radius: float = 12.0,
        smooth: bool = True,
    ) -> None:
        """Update laser pointer position for a specific hand with tremor-filtering."""
        if color in ("#ff0000", "#ef4444") and hasattr(self, "_laser_color"):
            color = self._laser_color
        if radius == 12.0 and hasattr(self, "_laser_radius"):
            radius = self._laser_radius

        if smooth:
            if hand_id not in self._laser_filters:
                self._laser_filters[hand_id] = OneEuroFilter2D(min_cutoff=1.2, beta=0.01)
            t_ms = time.perf_counter() * 1000.0
            fx, fy = self._laser_filters[hand_id].filter(x, y, t_ms)
        else:
            fx, fy = x, y

        if hand_id not in self._laser_items:
            c = QColor(color) if isinstance(color, str) else color
            item = QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
            item.setPen(QPen(c, 2))
            item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 80)))
            item.setZValue(self.Z_LASER)
            item.setVisible(False)
            self.addItem(item)
            self._laser_items[hand_id] = item
            if hand_id == "primary":
                self._laser_item = item

        item = self._laser_items[hand_id]
        item.setPos(fx, fy)
        item.setVisible(visible)

        # Update dynamic fading neon laser trail
        if getattr(self, "_laser_trail_enabled", True) and visible:
            if hand_id not in self._laser_trails:
                from src.ui.laser_trail_item import LaserTrailItem
                trail = LaserTrailItem(
                    color=color,
                    lifetime_sec=getattr(self, "_laser_trail_lifetime", 1.5),
                )
                self.addItem(trail)
                self._laser_trails[hand_id] = trail
            self._laser_trails[hand_id].add_point(fx, fy)

        # Update Cartesian coordinate HUD if active
        if visible and (hand_id == "primary" or len(self._laser_items) <= 1):
            self.update_coordinate_hud(fx, fy)

    def hide_laser(self, hand_id: Optional[str] = None) -> None:
        """Hide laser pointer for a specific hand or all hands."""
        if hand_id is None:
            for item in self._laser_items.values():
                item.setVisible(False)
            for f in self._laser_filters.values():
                f.reset()
        elif hand_id in self._laser_items:
            self._laser_items[hand_id].setVisible(False)
            if hand_id in self._laser_filters:
                self._laser_filters[hand_id].reset()

    def mouseMoveEvent(self, event: Any) -> None:
        """Track mouse movement to update floating coordinate HUD."""
        pos = event.scenePos()
        self.update_coordinate_hud(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    # ── Dynamic Fading Laser Trail Controls ────────────────────────────────────

    def set_laser_trail_enabled(self, enabled: bool) -> None:
        """Enable or disable fading neon trail behind the laser pointer."""
        self._laser_trail_enabled = bool(enabled)
        if not enabled:
            self.clear_laser_trails()

    def is_laser_trail_enabled(self) -> bool:
        """Check if fading neon laser trail is active."""
        return getattr(self, "_laser_trail_enabled", True)

    def set_laser_trail_lifetime(self, lifetime_sec: float) -> None:
        """Set how long the neon trail remains before dissolving (seconds)."""
        self._laser_trail_lifetime = max(0.2, float(lifetime_sec))
        for trail in getattr(self, "_laser_trails", {}).values():
            trail.set_lifetime(self._laser_trail_lifetime)

    def laser_trail_lifetime(self) -> float:
        """Get trail dissolution duration in seconds."""
        return getattr(self, "_laser_trail_lifetime", 1.5)

    def clear_laser_trails(self) -> None:
        """Clear all active laser trails immediately."""
        for trail in getattr(self, "_laser_trails", {}).values():
            trail.clear()

    def move_eraser_reticle(
        self,
        x: float,
        y: float,
        radius: float = 28.0,
        visible: bool = True,
        hand_id: str = "primary",
    ) -> None:
        """Show visual circular reticle at (x, y) with active eraser radius."""
        if hand_id not in self._eraser_items:
            item = QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
            pen = QPen(QColor(234, 67, 53, 220), 2, Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(234, 67, 53, 35)))
            item.setZValue(self.Z_PREVIEW + 2)
            item.setVisible(False)
            self.addItem(item)
            self._eraser_items[hand_id] = item

        item = self._eraser_items[hand_id]
        rect = item.rect()
        if abs(rect.width() - 2 * radius) > 0.5:
            item.setRect(-radius, -radius, 2 * radius, 2 * radius)
        item.setPos(x, y)
        item.setVisible(visible)

    def hide_eraser_reticle(self, hand_id: Optional[str] = None) -> None:
        """Hide eraser reticle for a specific hand or all hands."""
        if hand_id is None:
            for item in self._eraser_items.values():
                item.setVisible(False)
        elif hand_id in self._eraser_items:
            self._eraser_items[hand_id].setVisible(False)

    # ── Background page ───────────────────────────────────────────────────────

    def set_background_page(self, img_bgr: np.ndarray | QPixmap) -> None:
        """
        Set or replace the background document page image.
        Accepts either a BGR numpy array or a pre-rendered QPixmap.
        Scales to fill the board while preserving aspect ratio.
        """
        if self._bg_item is not None:
            self.removeItem(self._bg_item)
            self._bg_item = None

        if isinstance(img_bgr, QPixmap):
            scaled = img_bgr
            if scaled.width() != BOARD_WIDTH and scaled.height() != BOARD_HEIGHT:
                scaled = scaled.scaled(
                    BOARD_WIDTH, BOARD_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        else:
            pixmap = _numpy_bgr_to_qpixmap(img_bgr)
            scaled = pixmap.scaled(
                BOARD_WIDTH, BOARD_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._bg_item = QGraphicsPixmapItem(scaled)
        # Center on board
        ox = (BOARD_WIDTH  - scaled.width())  / 2
        oy = (BOARD_HEIGHT - scaled.height()) / 2
        self._bg_item.setPos(ox, oy)
        self._bg_item.setZValue(self.Z_BACKGROUND)
        self.addItem(self._bg_item)

    def clear_background(self) -> None:
        if self._bg_item is not None:
            self.removeItem(self._bg_item)
            self._bg_item = None

    # ── Multi-Hand Live drawing previews ───────────────────────────────────────

    def begin_stroke_preview(
        self,
        x: float,
        y: float,
        color: QColor,
        hand_id: str = "primary",
        width: float = 2.5,
    ) -> None:
        """Start a new in-progress stroke preview for a specific hand."""
        self.end_stroke_preview(hand_id)
        path = QPainterPath()
        path.moveTo(x, y)
        pen = QPen(
            color,
            width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self.Z_PREVIEW)
        self.addItem(item)

        self._live_previews[hand_id] = {
            "points": [(x, y)],
            "item": item,
            "color": color,
            "width": width,
        }
        if hand_id == "primary":
            self._preview_item = item
            self._live_points = [(x, y)]

    def update_stroke_preview(self, x: float, y: float, hand_id: str = "primary") -> None:
        """Extend the live preview stroke with a new filtered point for hand_id."""
        preview = self._live_previews.get(hand_id)
        if preview is None or not preview.get("points"):
            return
        preview["points"].append((x, y))
        pts = preview["points"]
        path = QPainterPath()
        path.moveTo(pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            path.lineTo(px, py)
        preview["item"].setPath(path)

        if hand_id == "primary":
            self._live_points = list(pts)

    def end_stroke_preview(self, hand_id: str = "primary") -> None:
        """Remove the preview stroke for a specific hand."""
        preview = self._live_previews.pop(hand_id, None)
        if preview is not None:
            item = preview.get("item")
            if item is not None and item.scene() is self:
                self.removeItem(item)

        if hand_id == "primary":
            if self._preview_item is not None and self._preview_item.scene() is self:
                self.removeItem(self._preview_item)
            self._preview_item = None
            self._live_points.clear()

    def end_all_stroke_previews(self) -> None:
        """Remove previews for all hands."""
        for hid in list(self._live_previews.keys()):
            self.end_stroke_preview(hid)
        self._preview_item = None
        self._live_points.clear()

    # ── Committed strokes ─────────────────────────────────────────────────────

    def commit_stroke(self, stroke: InkStroke) -> None:
        """
        Add a fully processed InkStroke to the scene.
        Removes the live preview for this stroke's hand_id without disturbing
        other hands.
        """
        self.end_stroke_preview(stroke.hand_id)

        if stroke.snap is not None and stroke.snap.shape_type != ShapeType.FREEFORM:
            item = self._render_snapped_shape(stroke)
            stroke._shape_item = item
            try:
                from src.utils.audio_feedback import AudioFeedbackManager
                AudioFeedbackManager.instance().play_snap()
            except Exception:
                pass
        else:
            item = self._render_bezier_stroke(stroke)
            stroke._path_item = item

        self._strokes[stroke.stroke_id] = stroke
        self.stroke_committed.emit(stroke.stroke_id)

    def _render_bezier_stroke(self, stroke: InkStroke) -> QGraphicsPathItem:
        """Render a freehand stroke as a single QPainterPath."""
        path = _build_bezier_path(stroke.segments)
        avg_width = float(np.mean(stroke.widths)) if stroke.widths else 2.5
        pen = QPen(stroke.color, avg_width, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self.Z_INK)
        item.setData(0, stroke.stroke_id)
        self.addItem(item)
        return item

    def _render_snapped_shape(self, stroke: InkStroke) -> QGraphicsItem:
        """Render a snapped geometric shape."""
        snap = stroke.snap
        color = _SNAP_COLOR.get(snap.shape_type, QColor("#1a73e8"))
        pen = QPen(color, 2.5, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        item: QGraphicsItem

        st = snap.shape_type
        p = snap.params

        if st == ShapeType.LINE:
            start = p.get("start")
            end = p.get("end")
            if start is not None and end is not None:
                item = QGraphicsLineItem(start[0], start[1], end[0], end[1])
                item.setPen(pen)
            elif snap.render_points is not None and len(snap.render_points) >= 2:
                p0 = snap.render_points[0]
                p1 = snap.render_points[-1]
                item = QGraphicsLineItem(p0[0], p0[1], p1[0], p1[1])
                item.setPen(pen)
            else:
                pts = p.get("points", [])
                if len(pts) >= 2:
                    item = QGraphicsLineItem(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])
                    item.setPen(pen)
                else:
                    item = self._render_bezier_stroke(stroke)

        elif st == ShapeType.ARROW:
            item = self._build_arrow_item(p, pen, snap.render_points)

        elif st in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
            center = p.get("center")
            if center is not None:
                cx, cy = float(center[0]), float(center[1])
                rx = float(p.get("semi_major", 40))
                ry = float(p.get("semi_minor", 40))
                angle = float(p.get("angle_deg", 0))
            else:
                cx = float(p.get("center_x", 0))
                cy = float(p.get("center_y", 0))
                rx = float(p.get("radius_x", p.get("radius", 40)))
                ry = float(p.get("radius_y", p.get("radius", 40)))
                angle = float(p.get("angle", 0))

            if snap.render_points is not None and len(snap.render_points) >= 3:
                path = QPainterPath()
                path.moveTo(snap.render_points[0][0], snap.render_points[0][1])
                for pt in snap.render_points[1:]:
                    path.lineTo(pt[0], pt[1])
                path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
            else:
                item = QGraphicsEllipseItem(cx - rx, cy - ry, 2 * rx, 2 * ry)
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
                if abs(angle) > 0.5:
                    item.setTransformOriginPoint(cx, cy)
                    item.setRotation(angle)

        elif st == ShapeType.RECTANGLE:
            corners = p.get("corners")
            if corners is not None and len(corners) == 4:
                path = QPainterPath()
                path.moveTo(corners[0][0], corners[0][1])
                for c in corners[1:]:
                    path.lineTo(c[0], c[1])
                path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
            elif snap.render_points is not None and len(snap.render_points) >= 4:
                path = QPainterPath()
                path.moveTo(snap.render_points[0][0], snap.render_points[0][1])
                for pt in snap.render_points[1:]:
                    path.lineTo(pt[0], pt[1])
                path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
            else:
                verts = p.get("vertices", [])
                if len(verts) == 4:
                    xs = [v[0] for v in verts]
                    ys = [v[1] for v in verts]
                    item = QGraphicsRectItem(min(xs), min(ys),
                                             max(xs) - min(xs), max(ys) - min(ys))
                    item.setPen(pen)
                    item.setBrush(Qt.BrushStyle.NoBrush)
                else:
                    item = self._render_bezier_stroke(stroke)

        elif st == ShapeType.TRIANGLE:
            corners = p.get("corners")
            if corners is not None and len(corners) == 3:
                path = QPainterPath()
                path.moveTo(corners[0][0], corners[0][1])
                for c in corners[1:]:
                    path.lineTo(c[0], c[1])
                path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
            elif snap.render_points is not None and len(snap.render_points) >= 3:
                path = QPainterPath()
                path.moveTo(snap.render_points[0][0], snap.render_points[0][1])
                for pt in snap.render_points[1:]:
                    path.lineTo(pt[0], pt[1])
                path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
            else:
                verts = p.get("vertices", [])
                path = QPainterPath()
                if len(verts) >= 3:
                    path.moveTo(verts[0][0], verts[0][1])
                    for v in verts[1:]:
                        path.lineTo(v[0], v[1])
                    path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)

        else:
            item = self._render_bezier_stroke(stroke)

        item.setZValue(self.Z_SHAPE)
        item.setData(0, stroke.stroke_id)
        self.addItem(item)
        return item

    def _build_arrow_item(
        self,
        params: dict,
        pen: QPen,
        render_points: Optional[np.ndarray] = None,
    ) -> QGraphicsPathItem:
        """Build an arrow path: shaft + two arrowhead lines."""
        start = params.get("start")
        end = params.get("tip") or params.get("shaft_end")
        if start is None or end is None:
            if render_points is not None and len(render_points) >= 2:
                start = render_points[0]
                end = render_points[-1]
            else:
                pts = params.get("points", [])
                if len(pts) >= 2:
                    start = pts[0]
                    end = pts[-1]
                else:
                    return QGraphicsPathItem()

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return QGraphicsPathItem()

        ux, uy = dx / length, dy / length
        arrow_len = min(20.0, length * 0.25)
        angle_rad = math.radians(25)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # Two arrowhead directions
        def _rotate(vx: float, vy: float, c: float, s: float) -> tuple[float, float]:
            return vx * c - vy * s, vx * s + vy * c

        l1x, l1y = _rotate(-ux, -uy,  cos_a, sin_a)
        l2x, l2y = _rotate(-ux, -uy,  cos_a, -sin_a)

        path = QPainterPath()
        path.moveTo(start[0], start[1])
        path.lineTo(end[0], end[1])
        path.moveTo(end[0], end[1])
        path.lineTo(end[0] + l1x * arrow_len, end[1] + l1y * arrow_len)
        path.moveTo(end[0], end[1])
        path.lineTo(end[0] + l2x * arrow_len, end[1] + l2y * arrow_len)

        item = QGraphicsPathItem(path)
        item.setPen(pen)
        return item

    def remove_stroke(self, stroke_id: str) -> bool:
        """Remove a stroke by ID. Returns True if found and removed."""
        stroke = self._strokes.pop(stroke_id, None)
        if stroke is None:
            return False
        for attr in ("_path_item", "_shape_item"):
            item = getattr(stroke, attr, None)
            if item is not None and item.scene() is self:
                self.removeItem(item)
        return True

    def erase_strokes_at(self, x: float, y: float, radius: float = 28.0) -> list[str]:
        """
        Erase any strokes passing near (x, y) using continuous Bézier segment-distance
        and snapped shape geometry.
        Returns:
            list of removed stroke IDs.
        """
        erased_ids = []
        target = np.array([x, y], dtype=np.float64)
        r2 = radius * radius

        for sid, stroke in list(self._strokes.items()):
            hit = False

            # Check snapped geometric shape if present
            if stroke.snap is not None and stroke.snap.shape_type != ShapeType.FREEFORM:
                st = stroke.snap.shape_type
                p = stroke.snap.params
                if st in (ShapeType.LINE, ShapeType.ARROW):
                    start = p.get("start")
                    end = p.get("end") or p.get("tip") or p.get("shaft_end")
                    if start is not None and end is not None:
                        d = _dist_point_to_segment_2d(x, y, start[0], start[1], end[0], end[1])
                        if d <= radius:
                            hit = True
                elif st in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
                    center = p.get("center") or (p.get("center_x", 0), p.get("center_y", 0))
                    rad_val = float(p.get("radius") or p.get("radius_x") or p.get("semi_major", 40))
                    dist_to_center = math.hypot(x - center[0], y - center[1])
                    if abs(dist_to_center - rad_val) <= radius or dist_to_center <= radius:
                        hit = True
                elif st in (ShapeType.RECTANGLE, ShapeType.TRIANGLE):
                    verts = p.get("corners") or p.get("vertices") or []
                    if len(verts) >= 3:
                        for i in range(len(verts)):
                            v1 = verts[i]
                            v2 = verts[(i + 1) % len(verts)]
                            if _dist_point_to_segment_2d(x, y, v1[0], v1[1], v2[0], v2[1]) <= radius:
                                hit = True
                                break

            # Check Bézier segments with fine sampling & segment distance
            if not hit:
                for seg in stroke.segments:
                    # Quick bounding check on segment endpoints & control points
                    pts = (seg.p0, seg.p1, seg.p2, seg.p3)
                    min_x = min(p[0] for p in pts) - radius
                    max_x = max(p[0] for p in pts) + radius
                    min_y = min(p[1] for p in pts) - radius
                    max_y = max(p[1] for p in pts) + radius
                    if not (min_x <= x <= max_x and min_y <= y <= max_y):
                        continue

                    # Direct control point check
                    for p in pts:
                        if float(np.sum((p - target) ** 2)) <= r2:
                            hit = True
                            break
                    if hit:
                        break

                    # Dense curve segment-distance check (5 samples -> 4 subsegments)
                    curve_samples = [
                        _eval_cubic_bezier(seg.p0, seg.p1, seg.p2, seg.p3, t)
                        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
                    ]
                    for i in range(len(curve_samples) - 1):
                        sa = curve_samples[i]
                        sb = curve_samples[i + 1]
                        if _dist_point_to_segment_2d(x, y, sa[0], sa[1], sb[0], sb[1]) <= radius:
                            hit = True
                            break
                    if hit:
                        break

            if hit:
                self.remove_stroke(sid)
                erased_ids.append(sid)

        if erased_ids:
            try:
                from src.utils.audio_feedback import AudioFeedbackManager
                AudioFeedbackManager.instance().play_erase()
            except Exception:
                pass

        return erased_ids

    def erase_object_at(self, x: float, y: float, radius: float = 28.0) -> Optional[str]:
        """Erase the single closest stroke or annotation at (x, y)."""
        erased = self.erase_strokes_at(x, y, radius=radius)
        if erased:
            return erased[0]

        target = QPointF(x, y)
        for item in list(self._overlay_items):
            if item.scene() is self and item.sceneBoundingRect().contains(target):
                self.removeItem(item)
                self._overlay_items.remove(item)
                return "overlay_item"
        return None

    def erase_area(self, rect: QRectF) -> list[str]:
        """Erase all strokes intersecting the given rectangle."""
        erased_ids = []
        for sid, stroke in list(self._strokes.items()):
            item = stroke._shape_item or stroke._path_item
            if item is not None:
                if rect.intersects(item.sceneBoundingRect()):
                    self.remove_stroke(sid)
                    erased_ids.append(sid)
        return erased_ids

    def clear_last_stroke(self, hand_id: Optional[str] = None) -> Optional[str]:
        """Remove the most recently added stroke (optionally by specific hand)."""
        if not self._strokes:
            return None
        if hand_id is None:
            sid = next(reversed(self._strokes))
            self.remove_stroke(sid)
            return sid
        for sid in reversed(list(self._strokes.keys())):
            if self._strokes[sid].hand_id == hand_id:
                self.remove_stroke(sid)
                return sid
        return None

    def clear_page_annotations(self) -> list[str]:
        """Remove all ink strokes and overlays on current page, strictly preserving background."""
        removed_ids = list(self._strokes.keys())
        self.clear_all_strokes()
        for item in list(self._overlay_items):
            if item.scene() is self:
                self.removeItem(item)
        self._overlay_items.clear()
        return removed_ids

    def clear_all_strokes(self) -> None:
        """Remove all ink strokes from the scene (keep background)."""
        for sid in list(self._strokes.keys()):
            self.remove_stroke(sid)

    # ── OCR overlays ─────────────────────────────────────────────────────────

    def add_text_overlay(
        self,
        text: str,
        x: float,
        y: float,
        font_size: int = 18,
        color: QColor | None = None,
    ) -> QGraphicsTextItem:
        """Place a text label on the board (replaces raw strokes after OCR)."""
        if color is None:
            color = QColor("#1a73e8")
        item = QGraphicsTextItem(text)
        item.setDefaultTextColor(color)
        font = QFont("Segoe UI", font_size)
        item.setFont(font)
        item.setPos(x, y)
        item.setZValue(self.Z_OCR_OVERLAY)
        self.addItem(item)
        self._overlay_items.append(item)
        return item

    def add_latex_overlay(
        self,
        rgba_img: np.ndarray,
        x: float,
        y: float,
        max_width: float = 600.0,
    ) -> QGraphicsPixmapItem:
        """Render a LaTeX RGBA image onto the board at (x, y)."""
        pixmap = _numpy_rgba_to_qpixmap(rgba_img)
        if pixmap.width() > max_width:
            pixmap = pixmap.scaledToWidth(
                int(max_width), Qt.TransformationMode.SmoothTransformation
            )
        item = QGraphicsPixmapItem(pixmap)
        item.setPos(x, y)
        item.setZValue(self.Z_OCR_OVERLAY)
        self.addItem(item)
        self._overlay_items.append(item)
        return item

    # ── Selection rectangle ───────────────────────────────────────────────────

    def show_selection_rect(self, rect: QRectF) -> None:
        """Draw a dashed selection rectangle (SELECT gesture)."""
        if self._select_rect_item is not None:
            self.removeItem(self._select_rect_item)
        pen = QPen(QColor("#1a73e8"), 1.5, Qt.PenStyle.DashLine)
        self._select_rect_item = QGraphicsRectItem(rect)
        self._select_rect_item.setPen(pen)
        self._select_rect_item.setBrush(QBrush(QColor(26, 115, 232, 30)))
        self._select_rect_item.setZValue(self.Z_LASER - 1)
        self.addItem(self._select_rect_item)
        self.selection_rect_changed.emit(rect)

    def hide_selection_rect(self) -> None:
        if self._select_rect_item is not None:
            self.removeItem(self._select_rect_item)
            self._select_rect_item = None

    # ── Interactive Item & Selection Manipulation ─────────────────────────────

    def get_items_in_rect(self, rect: QRectF) -> list[QGraphicsItem]:
        """
        Return all user-drawn items (strokes, shapes, STEM diagrams, overlays)
        intersecting rect, excluding background, cursor items, and selection boxes.
        """
        excluded = {
            getattr(self, "_bg_item", None),
            getattr(self, "_bg_pixmap_item", None),
            getattr(self, "_laser_item", None),
            getattr(self, "_laser_glow", None),
            getattr(self, "_pen_dot_item", None),
            getattr(self, "_select_rect_item", None),
            getattr(self, "_ruler_item", None),
            getattr(self, "_protractor_item", None),
            getattr(self, "_compass_item", None),
            getattr(self, "_spotlight_item", None),
            getattr(self, "_loupe_item", None),
            getattr(self, "_timer_item", None),
            getattr(self, "_metronome_item", None),
            getattr(self, "_coord_hud_item", None),
            getattr(self, "_angle_tool_item", None),
            getattr(self, "_triangle_tool_item", None),
            getattr(self, "_function_plotter_item", None),
            getattr(self, "_incline_tool_item", None),
            getattr(self, "_optics_refraction_item", None),
            getattr(self, "_standing_wave_item", None),
        }
        for item in getattr(self, "_eraser_items", {}).values():
            excluded.add(item)
        for item in getattr(self, "_laser_items", {}).values():
            excluded.add(item)
        for item in getattr(self, "_laser_glows", {}).values():
            excluded.add(item)
        for w in getattr(self, "_circuit_wires", []):
            excluded.add(w)

        items_in_rect: list[QGraphicsItem] = []
        for itm in self.items(rect, Qt.ItemSelectionMode.IntersectsItemShape):
            if itm not in excluded and itm is not None and itm.parentItem() is None:
                if itm is not getattr(self, "_bg_item", None):
                    items_in_rect.append(itm)
        return items_in_rect

    def get_strokes_in_rect(self, rect: QRectF) -> list[InkStroke]:
        """Return all InkStroke instances that intersect the given rectangle."""
        matched: list[InkStroke] = []
        for s in self._strokes.values():
            if s._path_item and s._path_item.sceneBoundingRect().intersects(rect):
                matched.append(s)
            elif s._shape_item and s._shape_item.sceneBoundingRect().intersects(rect):
                matched.append(s)
            else:
                for seg in s.segments:
                    p0 = QPointF(seg.p0[0], seg.p0[1])
                    p3 = QPointF(seg.p3[0], seg.p3[1])
                    if rect.contains(p0) or rect.contains(p3):
                        matched.append(s)
                        break
        return matched

    def translate_items(self, items: list[QGraphicsItem], dx: float, dy: float) -> None:
        """Translate a collection of graphics items by (dx, dy)."""
        offset = QPointF(dx, dy)
        for itm in items:
            if itm.scene() is self:
                itm.setPos(itm.pos() + offset)

    def delete_items(self, items: list[QGraphicsItem]) -> tuple[list[InkStroke], list[QGraphicsItem]]:
        """
        Remove a collection of items (strokes, shapes, STEM items, overlays) from the scene.
        Returns:
            (deleted_strokes, deleted_non_stroke_items)
        """
        del_strokes: list[InkStroke] = []
        del_other_items: list[QGraphicsItem] = []

        stroke_path_map = {s._path_item: s for s in self._strokes.values() if s._path_item}
        stroke_shape_map = {s._shape_item: s for s in self._strokes.values() if s._shape_item}

        for itm in items:
            stroke = stroke_path_map.get(itm) or stroke_shape_map.get(itm)
            if stroke:
                if stroke not in del_strokes:
                    self.remove_stroke(stroke.stroke_id)
                    del_strokes.append(stroke)
            else:
                if itm.scene() is self:
                    itm.setVisible(False)
                    self.removeItem(itm)
                    del_other_items.append(itm)

        return del_strokes, del_other_items


    # ── Serialization helpers ─────────────────────────────────────────────────

    def get_strokes_for_export(self) -> list[dict]:
        """
        Serialize all strokes to a list of dicts for .airteach package export.
        Each dict matches the ink/strokes.json schema.
        """
        result = []
        for stroke in self._strokes.values():
            segs = []
            for seg in stroke.segments:
                segs.append({
                    "p0":  [float(seg.p0[0]),  float(seg.p0[1])],
                    "cp1": [float(seg.p1[0]), float(seg.p1[1])],
                    "cp2": [float(seg.p2[0]), float(seg.p2[1])],
                    "p1":  [float(seg.p3[0]),  float(seg.p3[1])],
                })
            entry: dict = {
                "stroke_id":     stroke.stroke_id,
                "segments":      segs,
                "widths":        [float(w) for w in stroke.widths],
                "color_hex":     stroke.color.name(),
                "timestamps_ms": stroke.timestamps_ms,
            }
            if stroke.snap is not None:
                entry["snap"] = {
                    "shape_type": stroke.snap.shape_type.name,
                    "parameters": stroke.snap.params,
                }
            result.append(entry)
        return result

    def snapshot_as_pixmap(self) -> QPixmap:
        """Render the entire board to a QPixmap (for export / thumbnail)."""
        pixmap = QPixmap(BOARD_WIDTH, BOARD_HEIGHT)
        pixmap.fill(QColor("#f8f9fa"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.render(painter)
        painter.end()
        return pixmap
