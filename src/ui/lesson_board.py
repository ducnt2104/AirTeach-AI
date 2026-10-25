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
        self._transformer_attention_item: Optional[QGraphicsItem] = None
        self._special_relativity_item: Optional[QGraphicsItem] = None
        self._quantum_tunneling_item: Optional[QGraphicsItem] = None
        self._coriolis_earth_item: Optional[QGraphicsItem] = None
        self._acoustic_sonar_item: Optional[QGraphicsItem] = None
        self._cloud_chamber_item: Optional[QGraphicsItem] = None
        self._orbital_mechanics_item: Optional[QGraphicsItem] = None
        self._heat_transfer_item: Optional[QGraphicsItem] = None
        self._quantum_diffraction_item: Optional[QGraphicsItem] = None
        self._laval_nozzle_item: Optional[QGraphicsItem] = None
        self._ramachandran_plot_item: Optional[QGraphicsItem] = None

        # Interactive Physics Vector Force Decomposer & Inclined Plane
        self._incline_tool_item: Optional[QGraphicsItem] = None

        # Interactive Physics Optics Snell's Law & Refraction Simulator
        self._optics_refraction_item: Optional[QGraphicsItem] = None

        # Interactive Physics Wave Motion & Standing Wave Resonance Simulator
        self._standing_wave_item: Optional[QGraphicsItem] = None

        # Interactive Chemistry Molecular 3D VSEPR Geometry Inspector
        self._vsepr_geometry_item: Optional[QGraphicsItem] = None

        # Interactive Biology DNA Double Helix Replication Simulator
        self._dna_replication_item: Optional[QGraphicsItem] = None

        # Interactive Astronomy Celestial Mechanics & Planetary Orbit Kepler Simulator
        self._kepler_orbit_item: Optional[QGraphicsItem] = None

        # Interactive Computer Science Graph Theory & Pathfinding Simulator
        self._graph_pathfinding_item: Optional[QGraphicsItem] = None

        # Interactive Physics Electric Field Lines & Point Charges Coulomb Simulator
        self._electric_field_item: Optional[QGraphicsItem] = None

        # Interactive Physics Magnetism & Lorentz Force Particle Deflection Simulator
        self._lorentz_force_item: Optional[QGraphicsItem] = None

        # Interactive Biology Ecosystem Trophic Food Web & Lotka-Volterra Simulator
        self._trophic_web_item: Optional[QGraphicsItem] = None

        # Interactive Chemistry & Physics Ideal Gas Laws & Kinetic Theory Simulator
        self._gas_laws_item: Optional[QGraphicsItem] = None

        # Interactive Chemistry Acid-Base Titration & pH Buffer Curve Simulator
        self._titration_item: Optional[QGraphicsItem] = None

        # Interactive Physics & Mechanical Harmonic Oscillator & Resonance Simulator
        self._harmonic_oscillator_item: Optional[QGraphicsItem] = None

        # Interactive Nuclear Physics & Radioactive Decay Chain Law Simulator
        self._radioactive_decay_item: Optional[QGraphicsItem] = None

        # Interactive Physical Chemistry Chemical Kinetics & Equilibrium Simulator
        self._chemical_kinetics_item: Optional[QGraphicsItem] = None

        # Interactive Computer Science & Mathematics Cryptography RSA & Modular Arithmetic Engine
        self._cryptography_item: Optional[QGraphicsItem] = None

        # Interactive Physics & Astronomy Gravitational N-Body Planetary Orbit & Lagrange Points Simulator
        self._gravitational_orbits_item: Optional[QGraphicsItem] = None

        # Interactive Special Relativity Minkowski Spacetime & Lorentz Boost Simulator
        self._spacetime_relativity_item: Optional[QGraphicsItem] = None

        # Interactive Wave Optics & Young's Double-Slit Interference Simulator
        self._young_interference_item: Optional[QGraphicsItem] = None

        # Interactive AC Series RLC Circuit & Fresnel Phasor Simulator
        self._rlc_circuit_item: Optional[QGraphicsItem] = None

        # Interactive Geometric Optics Thin Lens & Optical Bench Simulator
        self._thin_lens_item: Optional[QGraphicsItem] = None
        self._bragg_diffraction_item: Optional[QGraphicsItem] = None
        self._buffer_system_item: Optional[QGraphicsItem] = None
        self._stern_gerlach_item: Optional[QGraphicsItem] = None
        self._enzyme_kinetics_item: Optional[QGraphicsItem] = None
        self._poiseuille_flow_item: Optional[QGraphicsItem] = None
        self._cosmic_expansion_item: Optional[QGraphicsItem] = None
        self._seismic_waves_item: Optional[QGraphicsItem] = None
        self._mass_spectrometer_item: Optional[QGraphicsItem] = None
        self._spectrophotometry_item: Optional[QGraphicsItem] = None
        self._hall_effect_item: Optional[QGraphicsItem] = None
        self._compton_scattering_item: Optional[QGraphicsItem] = None
        self._nuclear_binding_energy_item: Optional[QGraphicsItem] = None
        self._gravitational_lensing_item: Optional[QGraphicsItem] = None
        self._chemical_equilibrium_item: Optional[QGraphicsItem] = None
        self._hodgkin_huxley_item: Optional[QGraphicsItem] = None
        self._mst_visualizer_item: Optional[QGraphicsItem] = None
        self._radiation_shielding_item: Optional[QGraphicsItem] = None
        self._maxwell_boltzmann_item: Optional[QGraphicsItem] = None
        self._molecular_orbital_item: Optional[QGraphicsItem] = None
        self._rlc_resonance_item: Optional[QGraphicsItem] = None
        self._stress_strain_item: Optional[QGraphicsItem] = None
        self._kepler_orbits_item: Optional[QGraphicsItem] = None
        self._logic_circuits_item: Optional[QGraphicsItem] = None
        # Interactive Modern Physics Photoelectric Effect Simulator
        self._photoelectric_effect_item: Optional[QGraphicsItem] = None
        self._punnett_genetics_item: Optional[QGraphicsItem] = None
        self._reaction_kinetics_item: Optional[QGraphicsItem] = None
        self._calculus_taylor_item: Optional[QGraphicsItem] = None
        self._michelson_interferometer_item: Optional[QGraphicsItem] = None
        self._huffman_coding_item: Optional[QGraphicsItem] = None
        self._action_potential_item: Optional[QGraphicsItem] = None
        self._cyclotron_item: Optional[QGraphicsItem] = None
        self._aho_corasick_item: Optional[QGraphicsItem] = None
        self._phase_diagram_item: Optional[QGraphicsItem] = None
        self._em_wave_item: Optional[QGraphicsItem] = None
        self._hardy_weinberg_item: Optional[QGraphicsItem] = None
        self._fft_item: Optional[QGraphicsItem] = None
        self._fabry_perot_item: Optional[QGraphicsItem] = None
        self._hplc_item: Optional[QGraphicsItem] = None
        self._photoelectric_item: Optional[QGraphicsItem] = None
        self._pcr_item: Optional[QGraphicsItem] = None
        self._thin_lens_item: Optional[QGraphicsItem] = None
        # Interactive Thermodynamics Carnot Heat Engine Simulator
        self._carnot_engine_item: Optional[QGraphicsItem] = None
        # Interactive Quantum Physics Bohr Atom Simulator
        self._bohr_atom_item: Optional[QGraphicsItem] = None
        self._automata_item: Optional[QGraphicsItem] = None
        self._sorting_item: Optional[QGraphicsItem] = None
        self._punnett_genetics_item: Optional[QGraphicsItem] = None
        self._organic_chemistry_item: Optional[QGraphicsItem] = None
        self._fluid_venturi_item: Optional[QGraphicsItem] = None
        self._binary_tree_item: Optional[QGraphicsItem] = None
        self._electrochemistry_item: Optional[QGraphicsItem] = None
        self._circular_motion_item: Optional[QGraphicsItem] = None
        self._ph_titration_item: Optional[QGraphicsItem] = None
        self._electromagnetic_induction_item: Optional[QGraphicsItem] = None
        self._dynamic_programming_item: Optional[QGraphicsItem] = None
        self._projectile_motion_item: Optional[QGraphicsItem] = None
        self._photosynthesis_item: Optional[QGraphicsItem] = None
        self._gcms_item: Optional[QGraphicsItem] = None
        self._cnn_visualizer_item: Optional[QGraphicsItem] = None
        self._airfoil_item: Optional[QGraphicsItem] = None
        self._periodic_table_item: Optional[QGraphicsItem] = None
        self._cellular_respiration_item: Optional[QGraphicsItem] = None
        self._protein_folding_item: Optional[QGraphicsItem] = None
        self._mlp_backprop_item: Optional[QGraphicsItem] = None
        self._max_flow_item: Optional[QGraphicsItem] = None
        self._oxyz_geometry_item: Optional[QGraphicsItem] = None
        self._semiconductor_diode_item: Optional[QGraphicsItem] = None
        self._nuclear_reactions_item: Optional[QGraphicsItem] = None
        self._doppler_effect_item: Optional[QGraphicsItem] = None
        self._red_black_tree_item: Optional[QGraphicsItem] = None
        self._millikan_oil_drop_item: Optional[QGraphicsItem] = None
        self._quantum_well_item: Optional[QGraphicsItem] = None
        self._hohmann_transfer_item: Optional[QGraphicsItem] = None
        self._photoelectric_effect_item: Optional[QGraphicsItem] = None

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

    def show_transformer_attention(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Transformer multi-head self-attention visualizer."""
        if getattr(self, "_transformer_attention_item", None) is None:
            from src.ui.interactive_transformer_attention import InteractiveTransformerAttentionItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - InteractiveTransformerAttentionItem.WIDTH / 2.0, 120.0)
            self._transformer_attention_item = InteractiveTransformerAttentionItem(pos=spawn_pos)
            self.addItem(self._transformer_attention_item)
        self._transformer_attention_item.setVisible(visible)
        if visible and pos is not None:
            self._transformer_attention_item.setPos(pos[0], pos[1])
        self.update()
        return self._transformer_attention_item if visible else None

    def toggle_transformer_attention(self) -> bool:
        """Toggle interactive Transformer self-attention visualizer visibility."""
        if getattr(self, "_transformer_attention_item", None) is None or not self._transformer_attention_item.isVisible():
            self.show_transformer_attention(True)
            return True
        else:
            self.show_transformer_attention(False)
            return False

    def is_transformer_attention_visible(self) -> bool:
        """Return True if interactive Transformer visualizer is active on board."""
        return getattr(self, "_transformer_attention_item", None) is not None and self._transformer_attention_item.isVisible()

    def get_transformer_attention_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Transformer visualizer instance if created."""
        return getattr(self, "_transformer_attention_item", None)

    def show_special_relativity(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Special Relativity & Minkowski Spacetime visualizer."""
        if getattr(self, "_special_relativity_item", None) is None:
            from src.ui.interactive_special_relativity import InteractiveSpecialRelativityItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - InteractiveSpecialRelativityItem.WIDTH / 2.0, 110.0)
            self._special_relativity_item = InteractiveSpecialRelativityItem(pos=spawn_pos)
            self.addItem(self._special_relativity_item)
        self._special_relativity_item.setVisible(visible)
        if visible and pos is not None:
            self._special_relativity_item.setPos(pos[0], pos[1])
        self.update()
        return self._special_relativity_item if visible else None

    def toggle_special_relativity(self) -> bool:
        """Toggle interactive Special Relativity visualizer visibility."""
        if getattr(self, "_special_relativity_item", None) is None or not self._special_relativity_item.isVisible():
            self.show_special_relativity(True)
            return True
        else:
            self.show_special_relativity(False)
            return False

    def is_special_relativity_visible(self) -> bool:
        """Return True if interactive Special Relativity visualizer is active on board."""
        return getattr(self, "_special_relativity_item", None) is not None and self._special_relativity_item.isVisible()

    def get_special_relativity_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Special Relativity visualizer instance if created."""
        return getattr(self, "_special_relativity_item", None)

    # ── Interactive Quantum Mechanics & Tunneling Simulator ─────────────────────

    def show_quantum_tunneling(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Quantum Tunneling & Potential Well visualizer."""
        if getattr(self, "_quantum_tunneling_item", None) is None:
            from src.ui.interactive_quantum_tunneling import InteractiveQuantumTunnelingItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - InteractiveQuantumTunnelingItem.WIDTH / 2.0, 110.0)
            self._quantum_tunneling_item = InteractiveQuantumTunnelingItem(pos=spawn_pos)
            self.addItem(self._quantum_tunneling_item)
        self._quantum_tunneling_item.setVisible(visible)
        if visible and pos is not None:
            self._quantum_tunneling_item.setPos(pos[0], pos[1])
        self.update()
        return self._quantum_tunneling_item if visible else None

    def toggle_quantum_tunneling(self) -> bool:
        """Toggle interactive Quantum Tunneling visualizer visibility."""
        if getattr(self, "_quantum_tunneling_item", None) is None or not self._quantum_tunneling_item.isVisible():
            self.show_quantum_tunneling(True)
            return True
        else:
            self.show_quantum_tunneling(False)
            return False

    def is_quantum_tunneling_visible(self) -> bool:
        """Return True if interactive Quantum Tunneling visualizer is active on board."""
        return getattr(self, "_quantum_tunneling_item", None) is not None and self._quantum_tunneling_item.isVisible()

    def get_quantum_tunneling_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Quantum Tunneling visualizer instance if created."""
        return getattr(self, "_quantum_tunneling_item", None)

    # ── Interactive Atmospheric Physics: Coriolis & Geostrophic Simulator ───────

    def show_coriolis_earth(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Coriolis Effect & Geostrophic Flow visualizer."""
        if getattr(self, "_coriolis_earth_item", None) is None:
            from src.ui.interactive_coriolis_earth import InteractiveCoriolisEarthItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - InteractiveCoriolisEarthItem.WIDTH / 2.0, 110.0)
            self._coriolis_earth_item = InteractiveCoriolisEarthItem(pos=spawn_pos)
            self.addItem(self._coriolis_earth_item)
        self._coriolis_earth_item.setVisible(visible)
        if visible and pos is not None:
            self._coriolis_earth_item.setPos(pos[0], pos[1])
        self.update()
        return self._coriolis_earth_item if visible else None

    def toggle_coriolis_earth(self) -> bool:
        """Toggle interactive Coriolis Effect visualizer visibility."""
        if getattr(self, "_coriolis_earth_item", None) is None or not self._coriolis_earth_item.isVisible():
            self.show_coriolis_earth(True)
            return True
        else:
            self.show_coriolis_earth(False)
            return False

    def is_coriolis_earth_visible(self) -> bool:
        """Return True if interactive Coriolis Effect visualizer is active on board."""
        return getattr(self, "_coriolis_earth_item", None) is not None and self._coriolis_earth_item.isVisible()

    def get_coriolis_earth_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Coriolis Effect visualizer instance if created."""
        return getattr(self, "_coriolis_earth_item", None)

    # ── Interactive Acoustic Wave Physics & Sonar Simulator ──────────────────
    def show_acoustic_sonar(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Acoustic Doppler, Ultrasound & Sonar visualizer."""
        if getattr(self, "_acoustic_sonar_item", None) is None:
            from src.ui.interactive_acoustic_sonar import InteractiveAcousticSonarItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - 940.0 / 2.0, 110.0)
            self._acoustic_sonar_item = InteractiveAcousticSonarItem(board=self, pos=spawn_pos)
            self.addItem(self._acoustic_sonar_item)

        self._acoustic_sonar_item.setVisible(visible)
        if visible and pos is not None:
            self._acoustic_sonar_item.setPos(pos[0], pos[1])
        self.update()
        return self._acoustic_sonar_item if visible else None

    def toggle_acoustic_sonar(self) -> bool:
        """Toggle interactive Acoustic Doppler & Sonar visualizer visibility."""
        if getattr(self, "_acoustic_sonar_item", None) is None or not self._acoustic_sonar_item.isVisible():
            self.show_acoustic_sonar(True)
            return True
        else:
            self.show_acoustic_sonar(False)
            return False

    def is_acoustic_sonar_visible(self) -> bool:
        """Return True if interactive Acoustic Doppler visualizer is active on board."""
        return getattr(self, "_acoustic_sonar_item", None) is not None and self._acoustic_sonar_item.isVisible()

    def get_acoustic_sonar_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Acoustic Doppler visualizer instance if created."""
        return getattr(self, "_acoustic_sonar_item", None)

    # ── Interactive Nuclear Physics & Cloud Chamber Simulator ──────────
    def show_cloud_chamber(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Wilson Cloud Chamber & Particle Tracks visualizer."""
        if getattr(self, "_cloud_chamber_item", None) is None:
            from src.ui.interactive_cloud_chamber import InteractiveCloudChamberItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - 940.0 / 2.0, 110.0)
            self._cloud_chamber_item = InteractiveCloudChamberItem(board=self, pos=spawn_pos)
            self.addItem(self._cloud_chamber_item)

        self._cloud_chamber_item.setVisible(visible)
        if visible and pos is not None:
            self._cloud_chamber_item.setPos(pos[0], pos[1])
        self.update()
        return self._cloud_chamber_item if visible else None

    def toggle_cloud_chamber(self) -> bool:
        """Toggle interactive Wilson Cloud Chamber visualizer visibility."""
        if getattr(self, "_cloud_chamber_item", None) is None or not self._cloud_chamber_item.isVisible():
            self.show_cloud_chamber(True)
            return True
        else:
            self.show_cloud_chamber(False)
            return False

    def is_cloud_chamber_visible(self) -> bool:
        """Return True if interactive Wilson Cloud Chamber is active on board."""
        return getattr(self, "_cloud_chamber_item", None) is not None and self._cloud_chamber_item.isVisible()

    def get_cloud_chamber_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Wilson Cloud Chamber instance if created."""
        return getattr(self, "_cloud_chamber_item", None)

    # ── Interactive Astrodynamics & Orbital Mechanics Simulator ─────────
    def show_orbital_mechanics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Hohmann Transfer & Astrodynamics visualizer."""
        if getattr(self, "_orbital_mechanics_item", None) is None:
            from src.ui.interactive_orbital_mechanics import InteractiveOrbitalMechanicsItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - 940.0 / 2.0, 110.0)
            self._orbital_mechanics_item = InteractiveOrbitalMechanicsItem(board=self, pos=spawn_pos)
            self.addItem(self._orbital_mechanics_item)

        self._orbital_mechanics_item.setVisible(visible)
        if visible and pos is not None:
            self._orbital_mechanics_item.setPos(pos[0], pos[1])
        self.update()
        return self._orbital_mechanics_item if visible else None

    def toggle_orbital_mechanics(self) -> bool:
        """Toggle interactive Hohmann Transfer & Orbital Mechanics visualizer visibility."""
        if getattr(self, "_orbital_mechanics_item", None) is None or not self._orbital_mechanics_item.isVisible():
            self.show_orbital_mechanics(True)
            return True
        else:
            self.show_orbital_mechanics(False)
            return False

    def is_orbital_mechanics_visible(self) -> bool:
        """Return True if interactive Hohmann Transfer simulator is active on board."""
        return getattr(self, "_orbital_mechanics_item", None) is not None and self._orbital_mechanics_item.isVisible()

    def get_orbital_mechanics_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Hohmann Transfer instance if created."""
        return getattr(self, "_orbital_mechanics_item", None)

    # ── Interactive Thermodynamics & Heat Transfer Simulator ────────────
    def show_heat_transfer(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive 2D Heat Diffusion & Convection simulator."""
        if getattr(self, "_heat_transfer_item", None) is None:
            from src.ui.interactive_heat_transfer import InteractiveHeatTransferItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - 900.0 / 2.0, 110.0)
            self._heat_transfer_item = InteractiveHeatTransferItem(board=self)
            self._heat_transfer_item.setPos(spawn_pos[0], spawn_pos[1])
            self.addItem(self._heat_transfer_item)

        self._heat_transfer_item.setVisible(visible)
        if visible and pos is not None:
            self._heat_transfer_item.setPos(pos[0], pos[1])
        self.update()
        return self._heat_transfer_item if visible else None

    def toggle_heat_transfer(self) -> bool:
        """Toggle interactive 2D Heat Diffusion & Convection simulator visibility."""
        if getattr(self, "_heat_transfer_item", None) is None or not self._heat_transfer_item.isVisible():
            self.show_heat_transfer(True)
            return True
        else:
            self.show_heat_transfer(False)
            return False

    def is_heat_transfer_visible(self) -> bool:
        """Return True if interactive 2D Heat Transfer simulator is active on board."""
        return getattr(self, "_heat_transfer_item", None) is not None and self._heat_transfer_item.isVisible()

    def get_heat_transfer_item(self) -> Optional[QGraphicsItem]:
        """Return interactive 2D Heat Transfer instance if created."""
        return getattr(self, "_heat_transfer_item", None)

    # ── Interactive Quantum Double-Slit & de Broglie Matter Wave Simulator ──
    def show_quantum_diffraction(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Quantum Double-Slit & de Broglie matter wave visualizer."""
        if getattr(self, "_quantum_diffraction_item", None) is None:
            from src.ui.interactive_quantum_diffraction import InteractiveQuantumDiffractionItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - 920.0 / 2.0, 110.0)
            self._quantum_diffraction_item = InteractiveQuantumDiffractionItem(board=self)
            self._quantum_diffraction_item.setPos(spawn_pos[0], spawn_pos[1])
            self.addItem(self._quantum_diffraction_item)

        self._quantum_diffraction_item.setVisible(visible)
        if visible and pos is not None:
            self._quantum_diffraction_item.setPos(pos[0], pos[1])
        self.update()
        return self._quantum_diffraction_item if visible else None

    def toggle_quantum_diffraction(self) -> bool:
        """Toggle interactive Quantum Double-Slit simulator visibility."""
        if getattr(self, "_quantum_diffraction_item", None) is None or not self._quantum_diffraction_item.isVisible():
            self.show_quantum_diffraction(True)
            return True
        else:
            self.show_quantum_diffraction(False)
            return False

    def is_quantum_diffraction_visible(self) -> bool:
        """Return True if interactive Quantum Double-Slit simulator is active on board."""
        return getattr(self, "_quantum_diffraction_item", None) is not None and self._quantum_diffraction_item.isVisible()

    def get_quantum_diffraction_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Quantum Double-Slit instance if created."""
        return getattr(self, "_quantum_diffraction_item", None)

    # ── Interactive Aerodynamics & de Laval Nozzle Simulator ────────────
    def show_laval_nozzle(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive de Laval Nozzle & Supersonic shock diamonds visualizer."""
        if getattr(self, "_laval_nozzle_item", None) is None:
            from src.ui.interactive_laval_nozzle import InteractiveLavalNozzleItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - 920.0 / 2.0, 110.0)
            self._laval_nozzle_item = InteractiveLavalNozzleItem(board=self)
            self._laval_nozzle_item.setPos(spawn_pos[0], spawn_pos[1])
            self.addItem(self._laval_nozzle_item)

        self._laval_nozzle_item.setVisible(visible)
        if visible and pos is not None:
            self._laval_nozzle_item.setPos(pos[0], pos[1])
        self.update()
        return self._laval_nozzle_item if visible else None

    def toggle_laval_nozzle(self) -> bool:
        """Toggle interactive de Laval Nozzle simulator visibility."""
        if getattr(self, "_laval_nozzle_item", None) is None or not self._laval_nozzle_item.isVisible():
            self.show_laval_nozzle(True)
            return True
        else:
            self.show_laval_nozzle(False)
            return False

    def is_laval_nozzle_visible(self) -> bool:
        """Return True if interactive de Laval Nozzle simulator is active on board."""
        return getattr(self, "_laval_nozzle_item", None) is not None and self._laval_nozzle_item.isVisible()

    def get_laval_nozzle_item(self) -> Optional[QGraphicsItem]:
        """Return interactive de Laval Nozzle instance if created."""
        return getattr(self, "_laval_nozzle_item", None)

    # ── Interactive Structural Biochemistry & Ramachandran Plot Simulator ──
    def show_ramachandran_plot(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Ramachandran Plot & Protein Folding visualizer."""
        if getattr(self, "_ramachandran_plot_item", None) is None:
            from src.ui.interactive_ramachandran_plot import InteractiveRamachandranPlotItem
            spawn_pos = pos if pos is not None else (BOARD_WIDTH / 2.0 - 920.0 / 2.0, 110.0)
            self._ramachandran_plot_item = InteractiveRamachandranPlotItem(board=self)
            self._ramachandran_plot_item.setPos(spawn_pos[0], spawn_pos[1])
            self.addItem(self._ramachandran_plot_item)

        self._ramachandran_plot_item.setVisible(visible)
        if visible and pos is not None:
            self._ramachandran_plot_item.setPos(pos[0], pos[1])
        self.update()
        return self._ramachandran_plot_item if visible else None

    def toggle_ramachandran_plot(self) -> bool:
        """Toggle interactive Ramachandran Plot simulator visibility."""
        if getattr(self, "_ramachandran_plot_item", None) is None or not self._ramachandran_plot_item.isVisible():
            self.show_ramachandran_plot(True)
            return True
        else:
            self.show_ramachandran_plot(False)
            return False

    def is_ramachandran_plot_visible(self) -> bool:
        """Return True if interactive Ramachandran Plot simulator is active on board."""
        return getattr(self, "_ramachandran_plot_item", None) is not None and self._ramachandran_plot_item.isVisible()

    def get_ramachandran_plot_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Ramachandran Plot instance if created."""
        return getattr(self, "_ramachandran_plot_item", None)

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

    # ── Interactive Chemistry Molecular 3D VSEPR Geometry Inspector ────────────

    def show_vsepr_geometry(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
        model_key: str = "AX4",
    ) -> None:
        """Show or hide the interactive chemistry 3D VSEPR molecular geometry item."""
        if self._vsepr_geometry_item is None:
            from src.ui.interactive_vsepr_geometry import InteractiveVseprGeometryItem
            spawn_pos = pos if pos is not None else (500.0, 330.0)
            self._vsepr_geometry_item = InteractiveVseprGeometryItem(
                pos=spawn_pos,
                model_key=model_key,
            )
            self.addItem(self._vsepr_geometry_item)

        self._vsepr_geometry_item.setVisible(visible)
        if visible:
            if pos is not None:
                self._vsepr_geometry_item.setPos(pos[0], pos[1])
            if hasattr(self._vsepr_geometry_item, "set_model"):
                self._vsepr_geometry_item.set_model(model_key)
        self.update()

    def toggle_vsepr_geometry(self) -> bool:
        """Toggle interactive VSEPR molecular geometry visibility; returns new state."""
        if self._vsepr_geometry_item is None or not self._vsepr_geometry_item.isVisible():
            self.show_vsepr_geometry(True)
            return True
        else:
            self.show_vsepr_geometry(False)
            return False

    def is_vsepr_geometry_visible(self) -> bool:
        """Return True if interactive 3D VSEPR geometry simulator is active on board."""
        return self._vsepr_geometry_item is not None and self._vsepr_geometry_item.isVisible()

    def get_vsepr_geometry_item(self) -> Optional[QGraphicsItem]:
        """Return interactive VSEPR geometry instance if created."""
        return self._vsepr_geometry_item

    # ── Interactive Biology DNA Double Helix Replication Simulator ─────────────

    def show_dna_replication(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> None:
        """Show or hide the interactive biology DNA replication simulation item."""
        if self._dna_replication_item is None:
            from src.ui.interactive_dna_replication import InteractiveDnaReplicationItem
            spawn_pos = pos if pos is not None else (500.0, 320.0)
            self._dna_replication_item = InteractiveDnaReplicationItem(pos=spawn_pos)
            self.addItem(self._dna_replication_item)

        self._dna_replication_item.setVisible(visible)
        if visible and pos is not None:
            self._dna_replication_item.setPos(pos[0], pos[1])
        self.update()

    def toggle_dna_replication(self) -> bool:
        """Toggle interactive DNA replication simulator visibility; returns new state."""
        if self._dna_replication_item is None or not self._dna_replication_item.isVisible():
            self.show_dna_replication(True)
            return True
        else:
            self.show_dna_replication(False)
            return False

    def is_dna_replication_visible(self) -> bool:
        """Return True if interactive DNA replication simulator is active on board."""
        return self._dna_replication_item is not None and self._dna_replication_item.isVisible()

    def get_dna_replication_item(self) -> Optional[QGraphicsItem]:
        """Return interactive DNA replication instance if created."""
        return self._dna_replication_item

    def show_kepler_orbit(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
        preset: str = "EARTH",
    ) -> None:
        """Show or hide the interactive astronomy Kepler orbit celestial mechanics simulator item."""
        if self._kepler_orbit_item is None:
            from src.ui.interactive_kepler_orbit import InteractiveKeplerOrbitItem
            spawn_pos = pos if pos is not None else (510.0, 330.0)
            self._kepler_orbit_item = InteractiveKeplerOrbitItem(pos=spawn_pos)
            self.addItem(self._kepler_orbit_item)

        self._kepler_orbit_item.setVisible(visible)
        if visible:
            if hasattr(self._kepler_orbit_item, "set_preset") and preset in ("EARTH", "MARS", "MERCURY", "HALLEY"):
                self._kepler_orbit_item.set_preset(preset)
            if pos is not None:
                self._kepler_orbit_item.setPos(pos[0], pos[1])
        self.update()

    def toggle_kepler_orbit(self) -> bool:
        """Toggle interactive Kepler orbit simulator visibility; returns new state."""
        if self._kepler_orbit_item is None or not self._kepler_orbit_item.isVisible():
            self.show_kepler_orbit(True)
            return True
        else:
            self.show_kepler_orbit(False)
            return False

    def is_kepler_orbit_visible(self) -> bool:
        """Return True if interactive Kepler orbit simulator is active on board."""
        return self._kepler_orbit_item is not None and self._kepler_orbit_item.isVisible()

    def get_kepler_orbit_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Kepler orbit instance if created."""
        return self._kepler_orbit_item

    def show_graph_pathfinding(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
        preset: str = "NETWORK_7",
        algorithm: str = "DIJKSTRA",
    ) -> None:
        """Show or hide the interactive Computer Science graph theory & pathfinding simulator item."""
        if self._graph_pathfinding_item is None:
            from src.ui.interactive_graph_pathfinding import InteractiveGraphPathfindingItem
            spawn_pos = pos if pos is not None else (500.0, 320.0)
            self._graph_pathfinding_item = InteractiveGraphPathfindingItem(pos=spawn_pos)
            self.addItem(self._graph_pathfinding_item)

        self._graph_pathfinding_item.setVisible(visible)
        if visible:
            if hasattr(self._graph_pathfinding_item, "set_preset") and preset in ("NETWORK_7", "CYCLE_6", "BINARY_TREE"):
                self._graph_pathfinding_item.set_preset(preset)
            if hasattr(self._graph_pathfinding_item, "set_algorithm") and algorithm.upper() in ("DIJKSTRA", "ASTAR"):
                self._graph_pathfinding_item.set_algorithm(algorithm)
            if pos is not None:
                self._graph_pathfinding_item.setPos(pos[0], pos[1])
        self.update()

    def toggle_graph_pathfinding(self) -> bool:
        """Toggle interactive graph pathfinding simulator visibility; returns new state."""
        if self._graph_pathfinding_item is None or not self._graph_pathfinding_item.isVisible():
            self.show_graph_pathfinding(True)
            return True
        else:
            self.show_graph_pathfinding(False)
            return False

    def is_graph_pathfinding_visible(self) -> bool:
        """Return True if interactive graph pathfinding simulator is active on board."""
        return self._graph_pathfinding_item is not None and self._graph_pathfinding_item.isVisible()

    def get_graph_pathfinding_item(self) -> Optional[QGraphicsItem]:
        """Return interactive graph pathfinding instance if created."""
        return self._graph_pathfinding_item

    def show_electric_field(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
        preset: str = "DIPOLE",
    ) -> None:
        """Show or hide the interactive Physics electric field lines & Coulomb's law simulator item."""
        if self._electric_field_item is None:
            from src.ui.interactive_electric_field import InteractiveElectricFieldItem
            spawn_pos = pos if pos is not None else (500.0, 320.0)
            self._electric_field_item = InteractiveElectricFieldItem(pos=spawn_pos)
            self.addItem(self._electric_field_item)

        self._electric_field_item.setVisible(visible)
        if visible:
            if hasattr(self._electric_field_item, "set_preset") and preset in ("DIPOLE", "TWO_POSITIVE", "TWO_NEGATIVE", "QUADRUPOLE"):
                self._electric_field_item.set_preset(preset)
            if pos is not None:
                self._electric_field_item.setPos(pos[0], pos[1])
        self.update()

    def toggle_electric_field(self) -> bool:
        """Toggle interactive electric field simulator visibility; returns new state."""
        if self._electric_field_item is None or not self._electric_field_item.isVisible():
            self.show_electric_field(True)
            return True
        else:
            self.show_electric_field(False)
            return False

    def is_electric_field_visible(self) -> bool:
        """Return True if interactive electric field simulator is active on board."""
        return self._electric_field_item is not None and self._electric_field_item.isVisible()

    def get_electric_field_item(self) -> Optional[QGraphicsItem]:
        """Return interactive electric field instance if created."""
        return self._electric_field_item

    def show_lorentz_force(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
        particle_type: str = "ELECTRON",
        b_direction: str = "INTO",
    ) -> None:
        """Show or hide the interactive Physics magnetism & Lorentz force deflection simulator item."""
        if self._lorentz_force_item is None:
            from src.ui.interactive_lorentz_force import InteractiveLorentzForceItem
            spawn_pos = pos if pos is not None else (500.0, 320.0)
            self._lorentz_force_item = InteractiveLorentzForceItem(pos=spawn_pos)
            self.addItem(self._lorentz_force_item)

        self._lorentz_force_item.setVisible(visible)
        if visible:
            if hasattr(self._lorentz_force_item, "set_particle_type") and particle_type in ("ELECTRON", "PROTON", "ALPHA"):
                self._lorentz_force_item.set_particle_type(particle_type)
            if hasattr(self._lorentz_force_item, "_b_direction") and b_direction in ("INTO", "OUT"):
                self._lorentz_force_item._b_direction = b_direction
            if pos is not None:
                self._lorentz_force_item.setPos(pos[0], pos[1])
        self.update()

    def toggle_lorentz_force(self) -> bool:
        """Toggle interactive Lorentz force simulator visibility; returns new state."""
        if self._lorentz_force_item is None or not self._lorentz_force_item.isVisible():
            self.show_lorentz_force(True)
            return True
        else:
            self.show_lorentz_force(False)
            return False

    def is_lorentz_force_visible(self) -> bool:
        """Return True if interactive Lorentz force simulator is active on board."""
        return self._lorentz_force_item is not None and self._lorentz_force_item.isVisible()

    def get_lorentz_force_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Lorentz force instance if created."""
        return self._lorentz_force_item

    def show_trophic_web(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Biology Ecosystem Trophic Food Web simulator."""
        if self._trophic_web_item is None:
            from src.ui.interactive_trophic_web import InteractiveTrophicWebItem
            spawn_pos = pos if pos is not None else (500.0, 320.0)
            self._trophic_web_item = InteractiveTrophicWebItem(pos=spawn_pos)
            self.addItem(self._trophic_web_item)

        self._trophic_web_item.setVisible(visible)
        if visible and pos is not None:
            self._trophic_web_item.setPos(pos[0], pos[1])
        self.update()
        return self._trophic_web_item if visible else None

    def toggle_trophic_web(self) -> bool:
        """Toggle interactive trophic food web simulator visibility; returns new state."""
        if self._trophic_web_item is None or not self._trophic_web_item.isVisible():
            self.show_trophic_web(True)
            return True
        else:
            self.show_trophic_web(False)
            return False

    def is_trophic_web_visible(self) -> bool:
        """Return True if interactive trophic food web simulator is active on board."""
        return self._trophic_web_item is not None and self._trophic_web_item.isVisible()

    def get_trophic_web(self) -> Optional[QGraphicsItem]:
        """Return interactive trophic food web instance if active and visible."""
        if self._trophic_web_item is not None and self._trophic_web_item.isVisible():
            return self._trophic_web_item
        return None

    def get_trophic_web_item(self) -> Optional[QGraphicsItem]:
        """Return interactive trophic food web instance if created."""
        return self._trophic_web_item

    def show_gas_laws(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Ideal Gas Laws & Kinetic Theory simulator item."""
        if self._gas_laws_item is None:
            from src.ui.interactive_gas_laws import InteractiveGasLawsItem
            spawn_pos = pos if pos is not None else (480.0, 300.0)
            self._gas_laws_item = InteractiveGasLawsItem(pos=spawn_pos)
            self.addItem(self._gas_laws_item)

        self._gas_laws_item.setVisible(visible)
        if visible and pos is not None:
            self._gas_laws_item.setPos(pos[0], pos[1])
        self.update()
        return self._gas_laws_item if visible else None

    def toggle_gas_laws(self) -> bool:
        """Toggle interactive ideal gas laws simulator visibility; returns new state."""
        if self._gas_laws_item is None or not self._gas_laws_item.isVisible():
            self.show_gas_laws(True)
            return True
        else:
            self.show_gas_laws(False)
            return False

    def is_gas_laws_visible(self) -> bool:
        """Return True if interactive gas laws simulator is active on board."""
        return self._gas_laws_item is not None and self._gas_laws_item.isVisible()

    def get_gas_laws(self) -> Optional[QGraphicsItem]:
        """Return interactive gas laws instance if active and visible."""
        if self._gas_laws_item is not None and self._gas_laws_item.isVisible():
            return self._gas_laws_item
        return None

    def get_gas_laws_item(self) -> Optional[QGraphicsItem]:
        """Return interactive gas laws instance if created."""
        return self._gas_laws_item

    def show_titration(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Acid-Base Titration & pH Buffer simulator."""
        if self._titration_item is None:
            from src.ui.interactive_acid_base_titration import InteractiveAcidBaseTitrationItem
            spawn_pos = pos if pos is not None else (460.0, 280.0)
            self._titration_item = InteractiveAcidBaseTitrationItem(pos=spawn_pos)
            self.addItem(self._titration_item)

        self._titration_item.setVisible(visible)
        if visible and pos is not None:
            self._titration_item.setPos(pos[0], pos[1])
        self.update()
        return self._titration_item if visible else None

    def toggle_titration(self) -> bool:
        """Toggle interactive titration simulator visibility; returns new state."""
        if self._titration_item is None or not self._titration_item.isVisible():
            self.show_titration(True)
            return True
        else:
            self.show_titration(False)
            return False

    def is_titration_visible(self) -> bool:
        """Return True if interactive titration simulator is active on board."""
        return self._titration_item is not None and self._titration_item.isVisible()

    def get_titration(self) -> Optional[QGraphicsItem]:
        """Return interactive titration instance if active and visible."""
        if self._titration_item is not None and self._titration_item.isVisible():
            return self._titration_item
        return None

    def get_titration_item(self) -> Optional[QGraphicsItem]:
        """Return interactive titration instance if created."""
        return self._titration_item

    def show_harmonic_oscillator(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Harmonic Oscillator & Resonance simulator."""
        if self._harmonic_oscillator_item is None:
            from src.ui.interactive_harmonic_oscillator import InteractiveHarmonicOscillatorItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._harmonic_oscillator_item = InteractiveHarmonicOscillatorItem(pos=spawn_pos)
            self.addItem(self._harmonic_oscillator_item)

        self._harmonic_oscillator_item.setVisible(visible)
        if visible and pos is not None:
            self._harmonic_oscillator_item.setPos(pos[0], pos[1])
        self.update()
        return self._harmonic_oscillator_item if visible else None

    def toggle_harmonic_oscillator(self) -> bool:
        """Toggle interactive harmonic oscillator simulator visibility; returns new state."""
        if self._harmonic_oscillator_item is None or not self._harmonic_oscillator_item.isVisible():
            self.show_harmonic_oscillator(True)
            return True
        else:
            self.show_harmonic_oscillator(False)
            return False

    def is_harmonic_oscillator_visible(self) -> bool:
        """Return True if interactive harmonic oscillator simulator is active on board."""
        return self._harmonic_oscillator_item is not None and self._harmonic_oscillator_item.isVisible()

    def get_harmonic_oscillator(self) -> Optional[QGraphicsItem]:
        """Return interactive harmonic oscillator instance if active and visible."""
        if self._harmonic_oscillator_item is not None and self._harmonic_oscillator_item.isVisible():
            return self._harmonic_oscillator_item
        return None

    def get_harmonic_oscillator_item(self) -> Optional[QGraphicsItem]:
        """Return interactive harmonic oscillator instance if created."""
        return self._harmonic_oscillator_item

    def show_radioactive_decay(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Radioactive Decay & Nuclear Physics simulator."""
        if self._radioactive_decay_item is None:
            from src.ui.interactive_radioactive_decay import InteractiveRadioactiveDecayItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._radioactive_decay_item = InteractiveRadioactiveDecayItem(pos=spawn_pos)
            self.addItem(self._radioactive_decay_item)

        self._radioactive_decay_item.setVisible(visible)
        if visible and pos is not None:
            self._radioactive_decay_item.setPos(pos[0], pos[1])
        self.update()
        return self._radioactive_decay_item if visible else None

    def toggle_radioactive_decay(self) -> bool:
        """Toggle interactive radioactive decay simulator visibility; returns new state."""
        if self._radioactive_decay_item is None or not self._radioactive_decay_item.isVisible():
            self.show_radioactive_decay(True)
            return True
        else:
            self.show_radioactive_decay(False)
            return False

    def is_radioactive_decay_visible(self) -> bool:
        """Return True if interactive radioactive decay simulator is active on board."""
        return self._radioactive_decay_item is not None and self._radioactive_decay_item.isVisible()

    def get_radioactive_decay(self) -> Optional[QGraphicsItem]:
        """Return interactive radioactive decay instance if active and visible."""
        if self._radioactive_decay_item is not None and self._radioactive_decay_item.isVisible():
            return self._radioactive_decay_item
        return None

    def get_radioactive_decay_item(self) -> Optional[QGraphicsItem]:
        """Return interactive radioactive decay instance if created."""
        return self._radioactive_decay_item

    def show_chemical_kinetics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Chemical Kinetics & Equilibrium simulator."""
        if self._chemical_kinetics_item is None:
            from src.ui.interactive_chemical_kinetics import InteractiveChemicalKineticsItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._chemical_kinetics_item = InteractiveChemicalKineticsItem(pos=spawn_pos)
            self.addItem(self._chemical_kinetics_item)

        self._chemical_kinetics_item.setVisible(visible)
        if visible and pos is not None:
            self._chemical_kinetics_item.setPos(pos[0], pos[1])
        self.update()
        return self._chemical_kinetics_item if visible else None

    def toggle_chemical_kinetics(self) -> bool:
        """Toggle interactive chemical kinetics simulator visibility; returns new state."""
        if self._chemical_kinetics_item is None or not self._chemical_kinetics_item.isVisible():
            self.show_chemical_kinetics(True)
            return True
        else:
            self.show_chemical_kinetics(False)
            return False

    def is_chemical_kinetics_visible(self) -> bool:
        """Return True if interactive chemical kinetics simulator is active on board."""
        return self._chemical_kinetics_item is not None and self._chemical_kinetics_item.isVisible()

    def get_chemical_kinetics(self) -> Optional[QGraphicsItem]:
        """Return interactive chemical kinetics instance if active and visible."""
        if self._chemical_kinetics_item is not None and self._chemical_kinetics_item.isVisible():
            return self._chemical_kinetics_item
        return None

    def get_chemical_kinetics_item(self) -> Optional[QGraphicsItem]:
        """Return interactive chemical kinetics instance if created."""
        return self._chemical_kinetics_item

    def show_cryptography(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive RSA Cryptography & Modular Arithmetic simulator."""
        if self._cryptography_item is None:
            from src.ui.interactive_cryptography import InteractiveCryptographyItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._cryptography_item = InteractiveCryptographyItem(pos=spawn_pos)
            self.addItem(self._cryptography_item)

        self._cryptography_item.setVisible(visible)
        if visible and pos is not None:
            self._cryptography_item.setPos(pos[0], pos[1])
        self.update()
        return self._cryptography_item if visible else None

    def toggle_cryptography(self) -> bool:
        """Toggle interactive cryptography simulator visibility; returns new state."""
        if self._cryptography_item is None or not self._cryptography_item.isVisible():
            self.show_cryptography(True)
            return True
        else:
            self.show_cryptography(False)
            return False

    def is_cryptography_visible(self) -> bool:
        """Return True if interactive cryptography simulator is active on board."""
        return self._cryptography_item is not None and self._cryptography_item.isVisible()

    def get_cryptography(self) -> Optional[QGraphicsItem]:
        """Return interactive cryptography instance if active and visible."""
        if self._cryptography_item is not None and self._cryptography_item.isVisible():
            return self._cryptography_item
        return None

    def get_cryptography_item(self) -> Optional[QGraphicsItem]:
        """Return interactive cryptography instance if created."""
        return self._cryptography_item

    def show_gravitational_orbits(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Gravitational N-Body Orbits & Lagrange simulator."""
        if self._gravitational_orbits_item is None:
            from src.ui.interactive_gravitational_orbits import InteractiveGravitationalOrbitsItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._gravitational_orbits_item = InteractiveGravitationalOrbitsItem(pos=spawn_pos)
            self.addItem(self._gravitational_orbits_item)

        self._gravitational_orbits_item.setVisible(visible)
        if visible and pos is not None:
            self._gravitational_orbits_item.setPos(pos[0], pos[1])
        self.update()
        return self._gravitational_orbits_item if visible else None

    def toggle_gravitational_orbits(self) -> bool:
        """Toggle interactive gravitational orbits simulator visibility; returns new state."""
        if self._gravitational_orbits_item is None or not self._gravitational_orbits_item.isVisible():
            self.show_gravitational_orbits(True)
            return True
        else:
            self.show_gravitational_orbits(False)
            return False

    def is_gravitational_orbits_visible(self) -> bool:
        """Return True if interactive gravitational orbits simulator is active on board."""
        return self._gravitational_orbits_item is not None and self._gravitational_orbits_item.isVisible()

    def get_gravitational_orbits(self) -> Optional[QGraphicsItem]:
        """Return interactive gravitational orbits instance if active and visible."""
        if self._gravitational_orbits_item is not None and self._gravitational_orbits_item.isVisible():
            return self._gravitational_orbits_item
        return None

    def get_gravitational_orbits_item(self) -> Optional[QGraphicsItem]:
        """Return interactive gravitational orbits instance if created."""
        return self._gravitational_orbits_item

    def show_spacetime_relativity(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Special Relativity Minkowski Spacetime simulator."""
        if self._spacetime_relativity_item is None:
            from src.ui.interactive_spacetime_relativity import InteractiveSpacetimeRelativityItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._spacetime_relativity_item = InteractiveSpacetimeRelativityItem(pos=spawn_pos)
            self.addItem(self._spacetime_relativity_item)

        self._spacetime_relativity_item.setVisible(visible)
        if visible and pos is not None:
            self._spacetime_relativity_item.setPos(pos[0], pos[1])
        self.update()
        return self._spacetime_relativity_item if visible else None

    def toggle_spacetime_relativity(self) -> bool:
        """Toggle interactive spacetime relativity simulator visibility; returns new state."""
        if self._spacetime_relativity_item is None or not self._spacetime_relativity_item.isVisible():
            self.show_spacetime_relativity(True)
            return True
        else:
            self.show_spacetime_relativity(False)
            return False

    def is_spacetime_relativity_visible(self) -> bool:
        """Return True if interactive spacetime relativity simulator is active on board."""
        return self._spacetime_relativity_item is not None and self._spacetime_relativity_item.isVisible()

    def get_spacetime_relativity(self) -> Optional[QGraphicsItem]:
        """Return interactive spacetime relativity instance if active and visible."""
        if self._spacetime_relativity_item is not None and self._spacetime_relativity_item.isVisible():
            return self._spacetime_relativity_item
        return None

    def get_spacetime_relativity_item(self) -> Optional[QGraphicsItem]:
        """Return interactive spacetime relativity instance if created."""
        return self._spacetime_relativity_item

    def show_young_interference(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Wave Optics & Young's Double-Slit simulator."""
        if self._young_interference_item is None:
            from src.ui.interactive_young_interference import InteractiveYoungInterferenceItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._young_interference_item = InteractiveYoungInterferenceItem(pos=spawn_pos)
            self.addItem(self._young_interference_item)

        self._young_interference_item.setVisible(visible)
        if visible and pos is not None:
            self._young_interference_item.setPos(pos[0], pos[1])
        self.update()
        return self._young_interference_item if visible else None

    def toggle_young_interference(self) -> bool:
        """Toggle interactive wave optics interference simulator visibility; returns new state."""
        if self._young_interference_item is None or not self._young_interference_item.isVisible():
            self.show_young_interference(True)
            return True
        else:
            self.show_young_interference(False)
            return False

    def is_young_interference_visible(self) -> bool:
        """Return True if interactive wave optics interference simulator is active on board."""
        return self._young_interference_item is not None and self._young_interference_item.isVisible()

    def get_young_interference(self) -> Optional[QGraphicsItem]:
        """Return interactive wave optics interference instance if active and visible."""
        if self._young_interference_item is not None and self._young_interference_item.isVisible():
            return self._young_interference_item
        return None

    def get_young_interference_item(self) -> Optional[QGraphicsItem]:
        """Return interactive wave optics interference instance if created."""
        return self._young_interference_item

    def show_rlc_circuit(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive AC Series RLC Circuit & Phasor simulator."""
        if self._rlc_circuit_item is None:
            from src.ui.interactive_rlc_circuit import InteractiveRlcCircuitItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._rlc_circuit_item = InteractiveRlcCircuitItem(pos=spawn_pos)
            self.addItem(self._rlc_circuit_item)

        self._rlc_circuit_item.setVisible(visible)
        if visible and pos is not None:
            self._rlc_circuit_item.setPos(pos[0], pos[1])
        self.update()
        return self._rlc_circuit_item if visible else None

    def toggle_rlc_circuit(self) -> bool:
        """Toggle interactive RLC circuit simulator visibility; returns new state."""
        if self._rlc_circuit_item is None or not self._rlc_circuit_item.isVisible():
            self.show_rlc_circuit(True)
            return True
        else:
            self.show_rlc_circuit(False)
            return False

    def is_rlc_circuit_visible(self) -> bool:
        """Return True if interactive RLC circuit simulator is active on board."""
        return self._rlc_circuit_item is not None and self._rlc_circuit_item.isVisible()

    def get_rlc_circuit(self) -> Optional[QGraphicsItem]:
        """Return interactive RLC circuit instance if active and visible."""
        if self._rlc_circuit_item is not None and self._rlc_circuit_item.isVisible():
            return self._rlc_circuit_item
        return None

    def get_rlc_circuit_item(self) -> Optional[QGraphicsItem]:
        """Return interactive RLC circuit instance if created."""
        return self._rlc_circuit_item

    def show_thin_lens(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Geometric Optics Thin Lens & Optical Bench simulator."""
        if self._thin_lens_item is None:
            from src.ui.interactive_thin_lens import InteractiveThinLensBenchItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._thin_lens_item = InteractiveThinLensBenchItem(pos=spawn_pos)
            self.addItem(self._thin_lens_item)

        self._thin_lens_item.setVisible(visible)
        if visible and pos is not None:
            self._thin_lens_item.setPos(pos[0], pos[1])
        self.update()
        return self._thin_lens_item if visible else None

    def toggle_thin_lens(self) -> bool:
        """Toggle interactive thin lens simulator visibility; returns new state."""
        if self._thin_lens_item is None or not self._thin_lens_item.isVisible():
            self.show_thin_lens(True)
            return True
        else:
            self.show_thin_lens(False)
            return False

    def is_thin_lens_visible(self) -> bool:
        """Return True if interactive thin lens simulator is active on board."""
        return self._thin_lens_item is not None and self._thin_lens_item.isVisible()

    def get_thin_lens(self) -> Optional[QGraphicsItem]:
        """Return interactive thin lens instance if active and visible."""
        if self._thin_lens_item is not None and self._thin_lens_item.isVisible():
            return self._thin_lens_item
        return None

    def get_thin_lens_item(self) -> Optional[QGraphicsItem]:
        """Return interactive thin lens instance if created."""
        return self._thin_lens_item

    def show_photoelectric_effect(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Modern Physics Photoelectric Effect simulator."""
        if self._photoelectric_effect_item is None:
            from src.ui.interactive_photoelectric_effect import InteractivePhotoelectricEffectItem
            spawn_pos = pos if pos is not None else (400.0, 200.0)
            self._photoelectric_effect_item = InteractivePhotoelectricEffectItem(pos=spawn_pos)
            self.addItem(self._photoelectric_effect_item)

        self._photoelectric_effect_item.setVisible(visible)
        if visible and pos is not None:
            self._photoelectric_effect_item.setPos(pos[0], pos[1])
        self.update()
        return self._photoelectric_effect_item if visible else None

    def toggle_photoelectric_effect(self) -> bool:
        """Toggle interactive photoelectric effect simulator visibility; returns new state."""
        if self._photoelectric_effect_item is None or not self._photoelectric_effect_item.isVisible():
            self.show_photoelectric_effect(True)
            return True
        else:
            self.show_photoelectric_effect(False)
            return False

    def is_photoelectric_effect_visible(self) -> bool:
        """Return True if interactive photoelectric effect simulator is active on board."""
        return self._photoelectric_effect_item is not None and self._photoelectric_effect_item.isVisible()

    def get_photoelectric_effect(self) -> Optional[QGraphicsItem]:
        """Return interactive photoelectric effect instance if active and visible."""
        if self._photoelectric_effect_item is not None and self._photoelectric_effect_item.isVisible():
            return self._photoelectric_effect_item
        return None

    def get_photoelectric_effect_item(self) -> Optional[QGraphicsItem]:
        """Return interactive photoelectric effect instance if created."""
        return self._photoelectric_effect_item

    def show_carnot_engine(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Thermodynamics Carnot Heat Engine simulator."""
        if self._carnot_engine_item is None:
            from src.ui.interactive_carnot_engine import InteractiveCarnotEngineItem
            spawn_pos = pos if pos is not None else (380.0, 180.0)
            self._carnot_engine_item = InteractiveCarnotEngineItem(pos=spawn_pos)
            self.addItem(self._carnot_engine_item)

        self._carnot_engine_item.setVisible(visible)
        if visible and pos is not None:
            self._carnot_engine_item.setPos(pos[0], pos[1])
        self.update()
        return self._carnot_engine_item if visible else None

    def toggle_carnot_engine(self) -> bool:
        """Toggle interactive Carnot heat engine simulator visibility; returns new state."""
        if self._carnot_engine_item is None or not self._carnot_engine_item.isVisible():
            self.show_carnot_engine(True)
            return True
        else:
            self.show_carnot_engine(False)
            return False

    def is_carnot_engine_visible(self) -> bool:
        """Return True if interactive Carnot engine simulator is active on board."""
        return self._carnot_engine_item is not None and self._carnot_engine_item.isVisible()

    def get_carnot_engine(self) -> Optional[QGraphicsItem]:
        """Return interactive Carnot engine instance if active and visible."""
        if self._carnot_engine_item is not None and self._carnot_engine_item.isVisible():
            return self._carnot_engine_item
        return None

    def get_carnot_engine_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Carnot engine instance if created."""
        return self._carnot_engine_item

    def show_bohr_atom(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Quantum Physics Bohr Atom simulator."""
        if self._bohr_atom_item is None:
            from src.ui.interactive_bohr_atom import InteractiveBohrAtomItem
            spawn_pos = pos if pos is not None else (360.0, 180.0)
            self._bohr_atom_item = InteractiveBohrAtomItem(pos=spawn_pos)
            self.addItem(self._bohr_atom_item)

        self._bohr_atom_item.setVisible(visible)
        if visible and pos is not None:
            self._bohr_atom_item.setPos(pos[0], pos[1])
        self.update()
        return self._bohr_atom_item if visible else None

    def toggle_bohr_atom(self) -> bool:
        """Toggle interactive Bohr atom simulator visibility; returns new state."""
        if self._bohr_atom_item is None or not self._bohr_atom_item.isVisible():
            self.show_bohr_atom(True)
            return True
        else:
            self.show_bohr_atom(False)
            return False

    def is_bohr_atom_visible(self) -> bool:
        """Return True if interactive Bohr atom simulator is active on board."""
        return self._bohr_atom_item is not None and self._bohr_atom_item.isVisible()

    def get_bohr_atom(self) -> Optional[QGraphicsItem]:
        """Return interactive Bohr atom instance if active and visible."""
        if self._bohr_atom_item is not None and self._bohr_atom_item.isVisible():
            return self._bohr_atom_item
        return None

    def get_bohr_atom_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Bohr atom instance if created."""
        return self._bohr_atom_item

    def show_automata(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Computer Science Finite State Machine (Automata) simulator."""
        if self._automata_item is None:
            from src.ui.interactive_automata import InteractiveAutomataItem
            spawn_pos = pos if pos is not None else (340.0, 160.0)
            self._automata_item = InteractiveAutomataItem(pos=spawn_pos)
            self.addItem(self._automata_item)

        self._automata_item.setVisible(visible)
        if visible and pos is not None:
            self._automata_item.setPos(pos[0], pos[1])
        self.update()
        return self._automata_item if visible else None

    def toggle_automata(self) -> bool:
        """Toggle interactive Automata simulator visibility; returns new state."""
        if self._automata_item is None or not self._automata_item.isVisible():
            self.show_automata(True)
            return True
        else:
            self.show_automata(False)
            return False

    def is_automata_visible(self) -> bool:
        """Return True if interactive Automata simulator is active on board."""
        return self._automata_item is not None and self._automata_item.isVisible()

    def get_automata(self) -> Optional[QGraphicsItem]:
        """Return interactive Automata instance if active and visible."""
        if self._automata_item is not None and self._automata_item.isVisible():
            return self._automata_item
        return None

    def get_automata_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Automata instance if created."""
        return self._automata_item

    def show_sorting(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Computer Science Sorting Algorithm visualizer."""
        if self._sorting_item is None:
            from src.ui.interactive_sorting import InteractiveSortingItem
            spawn_pos = pos if pos is not None else (320.0, 150.0)
            self._sorting_item = InteractiveSortingItem(pos=spawn_pos)
            self.addItem(self._sorting_item)

        self._sorting_item.setVisible(visible)
        if visible and pos is not None:
            self._sorting_item.setPos(pos[0], pos[1])
        self.update()
        return self._sorting_item if visible else None

    def toggle_sorting(self) -> bool:
        """Toggle interactive Sorting visualizer visibility; returns new state."""
        if self._sorting_item is None or not self._sorting_item.isVisible():
            self.show_sorting(True)
            return True
        else:
            self.show_sorting(False)
            return False

    def is_sorting_visible(self) -> bool:
        """Return True if interactive Sorting visualizer is active on board."""
        return self._sorting_item is not None and self._sorting_item.isVisible()

    def get_sorting(self) -> Optional[QGraphicsItem]:
        """Return interactive Sorting instance if active and visible."""
        if self._sorting_item is not None and self._sorting_item.isVisible():
            return self._sorting_item
        return None

    def get_sorting_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Sorting instance if created."""
        return self._sorting_item

    def show_punnett_genetics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Biology Mendel Genetics & Punnett Square visualizer."""
        if self._punnett_genetics_item is None:
            from src.ui.interactive_genetics_punnett import InteractiveGeneticsPunnettItem
            spawn_pos = pos if pos is not None else (330.0, 140.0)
            self._punnett_genetics_item = InteractiveGeneticsPunnettItem(pos=spawn_pos)
            self.addItem(self._punnett_genetics_item)

        self._punnett_genetics_item.setVisible(visible)
        if visible and pos is not None:
            self._punnett_genetics_item.setPos(pos[0], pos[1])
        self.update()
        return self._punnett_genetics_item if visible else None

    def toggle_punnett_genetics(self) -> bool:
        """Toggle interactive Punnett genetics visualizer visibility; returns new state."""
        if self._punnett_genetics_item is None or not self._punnett_genetics_item.isVisible():
            self.show_punnett_genetics(True)
            return True
        else:
            self.show_punnett_genetics(False)
            return False

    def is_punnett_genetics_visible(self) -> bool:
        """Return True if interactive Punnett genetics visualizer is active on board."""
        return self._punnett_genetics_item is not None and self._punnett_genetics_item.isVisible()

    def get_punnett_genetics(self) -> Optional[QGraphicsItem]:
        """Return interactive Punnett genetics instance if active and visible."""
        if self._punnett_genetics_item is not None and self._punnett_genetics_item.isVisible():
            return self._punnett_genetics_item
        return None

    def get_punnett_genetics_item(self) -> Optional[QGraphicsItem]:
        """Return interactive Punnett genetics instance if created."""
        return self._punnett_genetics_item

    def show_organic_chemistry(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Organic Chemistry Hydrocarbon & Isomer Simulator."""
        if self._organic_chemistry_item is None:
            from src.ui.interactive_organic_chemistry import InteractiveOrganicChemistryItem
            spawn_pos = pos if pos is not None else (310.0, 130.0)
            self._organic_chemistry_item = InteractiveOrganicChemistryItem(pos=spawn_pos)
            self.addItem(self._organic_chemistry_item)

        self._organic_chemistry_item.setVisible(visible)
        if visible and pos is not None:
            self._organic_chemistry_item.setPos(pos[0], pos[1])
        self.update()
        return self._organic_chemistry_item if visible else None

    def toggle_organic_chemistry(self) -> bool:
        """Toggle interactive organic chemistry simulator visibility; returns new state."""
        if self._organic_chemistry_item is None or not self._organic_chemistry_item.isVisible():
            self.show_organic_chemistry(True)
            return True
        else:
            self.show_organic_chemistry(False)
            return False

    def is_organic_chemistry_visible(self) -> bool:
        """Return True if interactive organic chemistry simulator is active on board."""
        return self._organic_chemistry_item is not None and self._organic_chemistry_item.isVisible()

    def get_organic_chemistry(self) -> Optional[QGraphicsItem]:
        """Return interactive organic chemistry instance if active and visible."""
        if self._organic_chemistry_item is not None and self._organic_chemistry_item.isVisible():
            return self._organic_chemistry_item
        return None

    def get_organic_chemistry_item(self) -> Optional[QGraphicsItem]:
        """Return interactive organic chemistry instance if created."""
        return self._organic_chemistry_item

    def show_fluid_venturi(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Physics Fluid Dynamics Venturi Tube Simulator."""
        if self._fluid_venturi_item is None:
            from src.ui.interactive_fluid_venturi import InteractiveFluidVenturiItem
            spawn_pos = pos if pos is not None else (310.0, 130.0)
            self._fluid_venturi_item = InteractiveFluidVenturiItem(pos=spawn_pos)
            self.addItem(self._fluid_venturi_item)

        self._fluid_venturi_item.setVisible(visible)
        if visible and pos is not None:
            self._fluid_venturi_item.setPos(pos[0], pos[1])
        self.update()
        return self._fluid_venturi_item if visible else None

    def toggle_fluid_venturi(self) -> bool:
        """Toggle interactive fluid venturi simulator visibility; returns new state."""
        if self._fluid_venturi_item is None or not self._fluid_venturi_item.isVisible():
            self.show_fluid_venturi(True)
            return True
        else:
            self.show_fluid_venturi(False)
            return False

    def is_fluid_venturi_visible(self) -> bool:
        """Return True if interactive fluid venturi simulator is active on board."""
        return self._fluid_venturi_item is not None and self._fluid_venturi_item.isVisible()

    def get_fluid_venturi(self) -> Optional[QGraphicsItem]:
        """Return interactive fluid venturi instance if active and visible."""
        if self._fluid_venturi_item is not None and self._fluid_venturi_item.isVisible():
            return self._fluid_venturi_item
        return None

    def get_fluid_venturi_item(self) -> Optional[QGraphicsItem]:
        """Return interactive fluid venturi instance if created."""
        return self._fluid_venturi_item

    def show_binary_tree(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Informatics Binary Tree & Huffman Simulator."""
        if self._binary_tree_item is None:
            from src.ui.interactive_binary_tree import InteractiveBinaryTreeItem
            spawn_pos = pos if pos is not None else (310.0, 130.0)
            self._binary_tree_item = InteractiveBinaryTreeItem(pos=spawn_pos)
            self.addItem(self._binary_tree_item)

        self._binary_tree_item.setVisible(visible)
        if visible and pos is not None:
            self._binary_tree_item.setPos(pos[0], pos[1])
        self.update()
        return self._binary_tree_item if visible else None

    def toggle_binary_tree(self) -> bool:
        """Toggle interactive binary tree simulator visibility; returns new state."""
        if self._binary_tree_item is None or not self._binary_tree_item.isVisible():
            self.show_binary_tree(True)
            return True
        else:
            self.show_binary_tree(False)
            return False

    def is_binary_tree_visible(self) -> bool:
        """Return True if interactive binary tree simulator is active on board."""
        return self._binary_tree_item is not None and self._binary_tree_item.isVisible()

    def get_binary_tree(self) -> Optional[QGraphicsItem]:
        """Return interactive binary tree instance if active and visible."""
        if self._binary_tree_item is not None and self._binary_tree_item.isVisible():
            return self._binary_tree_item
        return None

    def get_binary_tree_item(self) -> Optional[QGraphicsItem]:
        """Return interactive binary tree instance if created."""
        return self._binary_tree_item

    def show_electrochemistry(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Chemistry Electrochemistry & Electrolysis Cell Simulator."""
        if self._electrochemistry_item is None:
            from src.ui.interactive_electrochemistry import InteractiveElectrochemistryItem
            spawn_pos = pos if pos is not None else (290.0, 110.0)
            self._electrochemistry_item = InteractiveElectrochemistryItem(pos=spawn_pos)
            self.addItem(self._electrochemistry_item)

        self._electrochemistry_item.setVisible(visible)
        if visible and pos is not None:
            self._electrochemistry_item.setPos(pos[0], pos[1])
        self.update()
        return self._electrochemistry_item if visible else None

    def toggle_electrochemistry(self) -> bool:
        """Toggle interactive electrochemistry cell simulator visibility; returns new state."""
        if self._electrochemistry_item is None or not self._electrochemistry_item.isVisible():
            self.show_electrochemistry(True)
            return True
        else:
            self.show_electrochemistry(False)
            return False

    def is_electrochemistry_visible(self) -> bool:
        """Return True if interactive electrochemistry simulator is active on board."""
        return self._electrochemistry_item is not None and self._electrochemistry_item.isVisible()

    def get_electrochemistry(self) -> Optional[QGraphicsItem]:
        """Return interactive electrochemistry instance if active and visible."""
        if self._electrochemistry_item is not None and self._electrochemistry_item.isVisible():
            return self._electrochemistry_item
        return None

    def get_electrochemistry_item(self) -> Optional[QGraphicsItem]:
        """Return interactive electrochemistry instance if created."""
        return self._electrochemistry_item

    def show_circular_motion(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Physics Circular Motion & Centripetal Simulator."""
        if self._circular_motion_item is None:
            from src.ui.interactive_circular_motion import InteractiveCircularMotionItem
            spawn_pos = pos if pos is not None else (290.0, 110.0)
            self._circular_motion_item = InteractiveCircularMotionItem(pos=spawn_pos)
            self.addItem(self._circular_motion_item)

        self._circular_motion_item.setVisible(visible)
        if visible and pos is not None:
            self._circular_motion_item.setPos(pos[0], pos[1])
        self.update()
        return self._circular_motion_item if visible else None

    def toggle_circular_motion(self) -> bool:
        """Toggle interactive circular motion simulator visibility; returns new state."""
        if self._circular_motion_item is None or not self._circular_motion_item.isVisible():
            self.show_circular_motion(True)
            return True
        else:
            self.show_circular_motion(False)
            return False

    def is_circular_motion_visible(self) -> bool:
        """Return True if interactive circular motion simulator is active on board."""
        return self._circular_motion_item is not None and self._circular_motion_item.isVisible()

    def get_circular_motion(self) -> Optional[QGraphicsItem]:
        """Return interactive circular motion instance if active and visible."""
        if self._circular_motion_item is not None and self._circular_motion_item.isVisible():
            return self._circular_motion_item
        return None

    def get_circular_motion_item(self) -> Optional[QGraphicsItem]:
        """Return interactive circular motion instance if created."""
        return self._circular_motion_item

    def show_ph_titration(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Chemistry Acid-Base Neutralization & pH Titration Simulator."""
        if self._ph_titration_item is None:
            from src.ui.interactive_ph_titration import InteractiveAcidBaseTitrationItem
            spawn_pos = pos if pos is not None else (290.0, 110.0)
            self._ph_titration_item = InteractiveAcidBaseTitrationItem(pos=spawn_pos)
            self.addItem(self._ph_titration_item)

        self._ph_titration_item.setVisible(visible)
        if visible and pos is not None:
            self._ph_titration_item.setPos(pos[0], pos[1])
        self.update()
        return self._ph_titration_item if visible else None

    def toggle_ph_titration(self) -> bool:
        """Toggle interactive pH titration simulator visibility; returns new state."""
        if self._ph_titration_item is None or not self._ph_titration_item.isVisible():
            self.show_ph_titration(True)
            return True
        else:
            self.show_ph_titration(False)
            return False

    def is_ph_titration_visible(self) -> bool:
        """Return True if interactive pH titration simulator is active on board."""
        return self._ph_titration_item is not None and self._ph_titration_item.isVisible()

    def get_ph_titration(self) -> Optional[QGraphicsItem]:
        """Return interactive pH titration instance if active and visible."""
        if self._ph_titration_item is not None and self._ph_titration_item.isVisible():
            return self._ph_titration_item
        return None

    def get_ph_titration_item(self) -> Optional[QGraphicsItem]:
        """Return interactive pH titration instance if created."""
        return self._ph_titration_item

    def show_electromagnetic_induction(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Physics Electromagnetic Induction & Faraday Simulator."""
        if self._electromagnetic_induction_item is None:
            from src.ui.interactive_electromagnetic_induction import InteractiveElectromagneticInductionItem
            spawn_pos = pos if pos is not None else (290.0, 110.0)
            self._electromagnetic_induction_item = InteractiveElectromagneticInductionItem(pos=spawn_pos)
            self.addItem(self._electromagnetic_induction_item)

        self._electromagnetic_induction_item.setVisible(visible)
        if visible and pos is not None:
            self._electromagnetic_induction_item.setPos(pos[0], pos[1])
        self.update()
        return self._electromagnetic_induction_item if visible else None

    def toggle_electromagnetic_induction(self) -> bool:
        """Toggle interactive electromagnetic induction simulator visibility; returns new state."""
        if self._electromagnetic_induction_item is None or not self._electromagnetic_induction_item.isVisible():
            self.show_electromagnetic_induction(True)
            return True
        else:
            self.show_electromagnetic_induction(False)
            return False

    def is_electromagnetic_induction_visible(self) -> bool:
        """Return True if interactive electromagnetic induction simulator is active on board."""
        return self._electromagnetic_induction_item is not None and self._electromagnetic_induction_item.isVisible()

    def get_electromagnetic_induction(self) -> Optional[QGraphicsItem]:
        """Return interactive electromagnetic induction instance if active and visible."""
        if self._electromagnetic_induction_item is not None and self._electromagnetic_induction_item.isVisible():
            return self._electromagnetic_induction_item
        return None

    def get_electromagnetic_induction_item(self) -> Optional[QGraphicsItem]:
        """Return interactive electromagnetic induction instance if created."""
        return self._electromagnetic_induction_item

    def show_gcms(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show, hide, or reposition the interactive GC-MS simulator."""
        if self._gcms_item is None:
            from src.ui.interactive_gcms import InteractiveGCMSItem
            spawn_pos = QPointF(pos[0], pos[1]) if pos else QPointF(200.0, 80.0)
            self._gcms_item = InteractiveGCMSItem(pos=spawn_pos)
            self.addItem(self._gcms_item)

        self._gcms_item.setVisible(visible)
        if pos:
            self._gcms_item.setPos(pos[0], pos[1])
        return self._gcms_item if visible else None

    def toggle_gcms(self) -> bool:
        """Toggle visibility of interactive GC-MS simulator."""
        if self._gcms_item is None or not self._gcms_item.isVisible():
            self.show_gcms(True)
            return True
        else:
            self.show_gcms(False)
            return False

    def is_gcms_visible(self) -> bool:
        """Check if interactive GC-MS simulator is visible."""
        return self._gcms_item is not None and self._gcms_item.isVisible()

    def get_gcms(self) -> Optional[QGraphicsItem]:
        """Get visible interactive GC-MS simulator item."""
        if self._gcms_item is not None and self._gcms_item.isVisible():
            return self._gcms_item
        return None

    def get_gcms_item(self) -> Optional[QGraphicsItem]:
        """Get interactive GC-MS simulator item regardless of visibility."""
        return self._gcms_item

    def show_photosynthesis(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show, hide, or reposition the interactive Photosynthesis simulator."""
        if self._photosynthesis_item is None:
            from src.ui.interactive_photosynthesis import InteractivePhotosynthesisItem
            spawn_pos = QPointF(pos[0], pos[1]) if pos else QPointF(220.0, 90.0)
            self._photosynthesis_item = InteractivePhotosynthesisItem(pos=spawn_pos)
            self.addItem(self._photosynthesis_item)

        self._photosynthesis_item.setVisible(visible)
        if pos:
            self._photosynthesis_item.setPos(pos[0], pos[1])
        return self._photosynthesis_item if visible else None

    def toggle_photosynthesis(self) -> bool:
        """Toggle visibility of interactive Photosynthesis simulator."""
        if self._photosynthesis_item is None or not self._photosynthesis_item.isVisible():
            self.show_photosynthesis(True)
            return True
        else:
            self.show_photosynthesis(False)
            return False

    def is_photosynthesis_visible(self) -> bool:
        """Check if interactive Photosynthesis simulator is visible."""
        return self._photosynthesis_item is not None and self._photosynthesis_item.isVisible()

    def get_photosynthesis(self) -> Optional[QGraphicsItem]:
        """Get visible interactive Photosynthesis simulator item."""
        if self._photosynthesis_item is not None and self._photosynthesis_item.isVisible():
            return self._photosynthesis_item
        return None

    def get_photosynthesis_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Photosynthesis simulator item regardless of visibility."""
        return self._photosynthesis_item

    def show_projectile_motion(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show, hide, or reposition the interactive Projectile Motion simulator."""
        if self._projectile_motion_item is None:
            from src.ui.interactive_projectile_motion import InteractiveProjectileMotionItem
            spawn_pos = QPointF(pos[0], pos[1]) if pos else QPointF(200.0, 150.0)
            self._projectile_motion_item = InteractiveProjectileMotionItem(pos=spawn_pos)
            self.addItem(self._projectile_motion_item)

        self._projectile_motion_item.setVisible(visible)
        if pos:
            self._projectile_motion_item.setPos(pos[0], pos[1])
        return self._projectile_motion_item if visible else None

    def toggle_projectile_motion(self) -> bool:
        """Toggle visibility of interactive Projectile Motion simulator."""
        if self._projectile_motion_item is None or not self._projectile_motion_item.isVisible():
            self.show_projectile_motion(True)
            return True
        else:
            self.show_projectile_motion(False)
            return False

    def is_projectile_motion_visible(self) -> bool:
        """Check if interactive Projectile Motion simulator is visible."""
        return self._projectile_motion_item is not None and self._projectile_motion_item.isVisible()

    def get_projectile_motion(self) -> Optional[QGraphicsItem]:
        """Get visible interactive Projectile Motion simulator item."""
        if self._projectile_motion_item is not None and self._projectile_motion_item.isVisible():
            return self._projectile_motion_item
        return None

    def get_projectile_motion_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Projectile Motion simulator item regardless of visibility."""
        return self._projectile_motion_item

    def show_cnn_visualizer(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show, hide, or reposition the interactive CNN 2D Convolution simulator."""
        if self._cnn_visualizer_item is None:
            from src.ui.interactive_cnn_visualizer import InteractiveCNNVisualizerItem
            spawn_pos = QPointF(pos[0], pos[1]) if pos else QPointF(180.0, 100.0)
            self._cnn_visualizer_item = InteractiveCNNVisualizerItem(pos=spawn_pos)
            self.addItem(self._cnn_visualizer_item)

        self._cnn_visualizer_item.setVisible(visible)
        if pos:
            self._cnn_visualizer_item.setPos(pos[0], pos[1])
        return self._cnn_visualizer_item if visible else None

    def toggle_cnn_visualizer(self) -> bool:
        """Toggle visibility of interactive CNN 2D Convolution simulator."""
        if self._cnn_visualizer_item is None or not self._cnn_visualizer_item.isVisible():
            self.show_cnn_visualizer(True)
            return True
        else:
            self.show_cnn_visualizer(False)
            return False

    def is_cnn_visualizer_visible(self) -> bool:
        """Check if interactive CNN 2D Convolution simulator is visible."""
        return self._cnn_visualizer_item is not None and self._cnn_visualizer_item.isVisible()

    def get_cnn_visualizer(self) -> Optional[QGraphicsItem]:
        """Get visible interactive CNN 2D Convolution simulator item."""
        if self._cnn_visualizer_item is not None and self._cnn_visualizer_item.isVisible():
            return self._cnn_visualizer_item
        return None

    def get_cnn_visualizer_item(self) -> Optional[QGraphicsItem]:
        """Get interactive CNN 2D Convolution simulator item regardless of visibility."""
        return self._cnn_visualizer_item

    def show_airfoil_aerodynamics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show, hide, or reposition the interactive Airfoil Aerodynamics simulator."""
        if self._airfoil_item is None:
            from src.ui.interactive_airfoil_aerodynamics import InteractiveAirfoilAerodynamicsItem
            spawn_pos = QPointF(pos[0], pos[1]) if pos else QPointF(180.0, 100.0)
            self._airfoil_item = InteractiveAirfoilAerodynamicsItem(pos=spawn_pos)
            self.addItem(self._airfoil_item)

        self._airfoil_item.setVisible(visible)
        if pos:
            self._airfoil_item.setPos(pos[0], pos[1])
        return self._airfoil_item if visible else None

    def toggle_airfoil_aerodynamics(self) -> bool:
        """Toggle visibility of interactive Airfoil Aerodynamics simulator."""
        if self._airfoil_item is None or not self._airfoil_item.isVisible():
            self.show_airfoil_aerodynamics(True)
            return True
        else:
            self.show_airfoil_aerodynamics(False)
            return False

    def is_airfoil_aerodynamics_visible(self) -> bool:
        """Check if interactive Airfoil Aerodynamics simulator is visible."""
        return self._airfoil_item is not None and self._airfoil_item.isVisible()

    def get_airfoil_aerodynamics(self) -> Optional[QGraphicsItem]:
        """Get visible interactive Airfoil Aerodynamics simulator item."""
        if self._airfoil_item is not None and self._airfoil_item.isVisible():
            return self._airfoil_item
        return None

    def get_airfoil_aerodynamics_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Airfoil Aerodynamics simulator item regardless of visibility."""
        return self._airfoil_item

    def show_periodic_table(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show, hide, or reposition the interactive Periodic Table simulator."""
        if self._periodic_table_item is None:
            from src.ui.interactive_periodic_table import InteractivePeriodicTableItem
            spawn_pos = QPointF(pos[0], pos[1]) if pos else QPointF(180.0, 100.0)
            self._periodic_table_item = InteractivePeriodicTableItem(pos=spawn_pos)
            self.addItem(self._periodic_table_item)

        self._periodic_table_item.setVisible(visible)
        if pos:
            self._periodic_table_item.setPos(pos[0], pos[1])
        return self._periodic_table_item if visible else None

    def toggle_periodic_table(self) -> bool:
        """Toggle visibility of interactive Periodic Table simulator."""
        if self._periodic_table_item is None or not self._periodic_table_item.isVisible():
            self.show_periodic_table(True)
            return True
        else:
            self.show_periodic_table(False)
            return False

    def is_periodic_table_visible(self) -> bool:
        """Check if interactive Periodic Table simulator is visible."""
        return self._periodic_table_item is not None and self._periodic_table_item.isVisible()

    def get_periodic_table(self) -> Optional[QGraphicsItem]:
        """Get visible interactive Periodic Table simulator item."""
        if self._periodic_table_item is not None and self._periodic_table_item.isVisible():
            return self._periodic_table_item
        return None

    def get_periodic_table_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Periodic Table simulator item regardless of visibility."""
        return self._periodic_table_item

    def show_cellular_respiration(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Cellular Respiration & ETC simulator."""
        if self._cellular_respiration_item is None:
            from src.ui.interactive_cellular_respiration import InteractiveCellularRespirationItem
            spawn_pos = pos if pos is not None else (160.0, 75.0)
            self._cellular_respiration_item = InteractiveCellularRespirationItem(pos=spawn_pos)
            self.addItem(self._cellular_respiration_item)
        else:
            self._cellular_respiration_item.setVisible(visible)
            if pos is not None:
                self._cellular_respiration_item.setPos(pos[0], pos[1])
        return self._cellular_respiration_item if visible else None

    def toggle_cellular_respiration(self) -> bool:
        """Toggle interactive Cellular Respiration simulator visibility."""
        if self._cellular_respiration_item is None or not self._cellular_respiration_item.isVisible():
            self.show_cellular_respiration(True)
            return True
        else:
            self.show_cellular_respiration(False)
            return False

    def is_cellular_respiration_visible(self) -> bool:
        """Check if interactive Cellular Respiration simulator is currently visible."""
        return self._cellular_respiration_item is not None and self._cellular_respiration_item.isVisible()

    def get_cellular_respiration(self) -> Optional[QGraphicsItem]:
        """Get interactive Cellular Respiration simulator item if visible."""
        if self._cellular_respiration_item is not None and self._cellular_respiration_item.isVisible():
            return self._cellular_respiration_item
        return None

    def get_cellular_respiration_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Cellular Respiration simulator item regardless of visibility."""
        return self._cellular_respiration_item

    def show_protein_folding(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Protein Folding & Ramachandran simulator."""
        if self._protein_folding_item is None:
            from src.ui.interactive_protein_folding import InteractiveProteinFoldingItem
            spawn_pos = pos if pos is not None else (160.0, 70.0)
            self._protein_folding_item = InteractiveProteinFoldingItem(pos=spawn_pos)
            self.addItem(self._protein_folding_item)
        else:
            self._protein_folding_item.setVisible(visible)
            if pos is not None:
                self._protein_folding_item.setPos(pos[0], pos[1])
        return self._protein_folding_item if visible else None

    def toggle_protein_folding(self) -> bool:
        """Toggle interactive Protein Folding simulator visibility."""
        if self._protein_folding_item is None or not self._protein_folding_item.isVisible():
            self.show_protein_folding(True)
            return True
        else:
            self.show_protein_folding(False)
            return False

    def is_protein_folding_visible(self) -> bool:
        """Check if interactive Protein Folding simulator is currently visible."""
        return self._protein_folding_item is not None and self._protein_folding_item.isVisible()

    def get_protein_folding(self) -> Optional[QGraphicsItem]:
        """Get interactive Protein Folding simulator item if visible."""
        if self._protein_folding_item is not None and self._protein_folding_item.isVisible():
            return self._protein_folding_item
        return None

    def get_protein_folding_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Protein Folding simulator item regardless of visibility."""
        return self._protein_folding_item

    def show_mlp_backprop(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive MLP Backprop & Decision Boundary simulator."""
        if self._mlp_backprop_item is None:
            from src.ui.interactive_mlp_backprop import InteractiveMLPBackpropItem
            spawn_pos = pos if pos is not None else (150.0, 65.0)
            self._mlp_backprop_item = InteractiveMLPBackpropItem(pos=spawn_pos)
            self.addItem(self._mlp_backprop_item)
        else:
            self._mlp_backprop_item.setVisible(visible)
            if pos is not None:
                self._mlp_backprop_item.setPos(pos[0], pos[1])
        return self._mlp_backprop_item if visible else None

    def toggle_mlp_backprop(self) -> bool:
        """Toggle interactive MLP Backprop simulator visibility."""
        if self._mlp_backprop_item is None or not self._mlp_backprop_item.isVisible():
            self.show_mlp_backprop(True)
            return True
        else:
            self.show_mlp_backprop(False)
            return False

    def is_mlp_backprop_visible(self) -> bool:
        """Check if interactive MLP Backprop simulator is currently visible."""
        return self._mlp_backprop_item is not None and self._mlp_backprop_item.isVisible()

    def get_mlp_backprop(self) -> Optional[QGraphicsItem]:
        """Get interactive MLP Backprop simulator item if visible."""
        if self._mlp_backprop_item is not None and self._mlp_backprop_item.isVisible():
            return self._mlp_backprop_item
        return None

    def get_mlp_backprop_item(self) -> Optional[QGraphicsItem]:
        """Get interactive MLP Backprop simulator item regardless of visibility."""
        return self._mlp_backprop_item

    def show_max_flow(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Max Flow Network & Min-Cut simulator."""
        if self._max_flow_item is None:
            from src.ui.interactive_max_flow import InteractiveMaxFlowNetworkItem
            spawn_pos = pos if pos is not None else (150.0, 65.0)
            self._max_flow_item = InteractiveMaxFlowNetworkItem(pos=spawn_pos)
            self.addItem(self._max_flow_item)
        else:
            self._max_flow_item.setVisible(visible)
            if pos is not None:
                self._max_flow_item.setPos(pos[0], pos[1])
        return self._max_flow_item if visible else None

    def toggle_max_flow(self) -> bool:
        """Toggle interactive Max Flow Network simulator visibility."""
        if self._max_flow_item is None or not self._max_flow_item.isVisible():
            self.show_max_flow(True)
            return True
        else:
            self.show_max_flow(False)
            return False

    def is_max_flow_visible(self) -> bool:
        """Check if interactive Max Flow Network simulator is currently visible."""
        return self._max_flow_item is not None and self._max_flow_item.isVisible()

    def get_max_flow(self) -> Optional[QGraphicsItem]:
        """Get interactive Max Flow Network simulator item if visible."""
        if self._max_flow_item is not None and self._max_flow_item.isVisible():
            return self._max_flow_item
        return None

    def get_max_flow_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Max Flow Network simulator item regardless of visibility."""
        return self._max_flow_item

    def show_oxyz_geometry(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive 3D Oxyz Geometry simulator."""
        if self._oxyz_geometry_item is None:
            from src.ui.interactive_oxyz_geometry import InteractiveOxyzGeometryItem
            spawn_pos = pos if pos is not None else (150.0, 65.0)
            self._oxyz_geometry_item = InteractiveOxyzGeometryItem(pos=spawn_pos)
            self.addItem(self._oxyz_geometry_item)
        else:
            self._oxyz_geometry_item.setVisible(visible)
            if pos is not None:
                self._oxyz_geometry_item.setPos(pos[0], pos[1])
        return self._oxyz_geometry_item if visible else None

    def toggle_oxyz_geometry(self) -> bool:
        """Toggle interactive 3D Oxyz Geometry simulator visibility."""
        if self._oxyz_geometry_item is None or not self._oxyz_geometry_item.isVisible():
            self.show_oxyz_geometry(True)
            return True
        else:
            self.show_oxyz_geometry(False)
            return False

    def is_oxyz_geometry_visible(self) -> bool:
        """Check if interactive 3D Oxyz Geometry simulator is currently visible."""
        return self._oxyz_geometry_item is not None and self._oxyz_geometry_item.isVisible()

    def get_oxyz_geometry(self) -> Optional[QGraphicsItem]:
        """Get interactive 3D Oxyz Geometry simulator item if visible."""
        if self._oxyz_geometry_item is not None and self._oxyz_geometry_item.isVisible():
            return self._oxyz_geometry_item
        return None

    def get_oxyz_geometry_item(self) -> Optional[QGraphicsItem]:
        """Get interactive 3D Oxyz Geometry simulator item regardless of visibility."""
        return self._oxyz_geometry_item

    def show_semiconductor_diode(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Semiconductor P-N Diode simulator."""
        if self._semiconductor_diode_item is None:
            from src.ui.interactive_semiconductor_diode import InteractiveSemiconductorDiodeItem
            spawn_pos = pos if pos is not None else (150.0, 65.0)
            self._semiconductor_diode_item = InteractiveSemiconductorDiodeItem(pos=spawn_pos)
            self.addItem(self._semiconductor_diode_item)
        else:
            self._semiconductor_diode_item.setVisible(visible)
            if pos is not None:
                self._semiconductor_diode_item.setPos(pos[0], pos[1])
        return self._semiconductor_diode_item if visible else None

    def toggle_semiconductor_diode(self) -> bool:
        """Toggle interactive Semiconductor P-N Diode simulator visibility."""
        if self._semiconductor_diode_item is None or not self._semiconductor_diode_item.isVisible():
            self.show_semiconductor_diode(True)
            return True
        else:
            self.show_semiconductor_diode(False)
            return False

    def is_semiconductor_diode_visible(self) -> bool:
        """Check if interactive Semiconductor P-N Diode simulator is currently visible."""
        return self._semiconductor_diode_item is not None and self._semiconductor_diode_item.isVisible()

    def get_semiconductor_diode(self) -> Optional[QGraphicsItem]:
        """Get interactive Semiconductor P-N Diode simulator item if visible."""
        if self._semiconductor_diode_item is not None and self._semiconductor_diode_item.isVisible():
            return self._semiconductor_diode_item
        return None

    def get_semiconductor_diode_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Semiconductor P-N Diode simulator item regardless of visibility."""
        return self._semiconductor_diode_item

    def show_nuclear_reactions(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Nuclear Physics & Reactions simulator."""
        if self._nuclear_reactions_item is None:
            from src.ui.interactive_nuclear_reactions import InteractiveNuclearReactionsItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._nuclear_reactions_item = InteractiveNuclearReactionsItem(pos=spawn_pos)
            self.addItem(self._nuclear_reactions_item)
        else:
            self._nuclear_reactions_item.setVisible(visible)
            if pos is not None:
                self._nuclear_reactions_item.setPos(pos[0], pos[1])
        return self._nuclear_reactions_item if visible else None

    def toggle_nuclear_reactions(self) -> bool:
        """Toggle interactive Nuclear Physics & Reactions simulator visibility."""
        if self._nuclear_reactions_item is None or not self._nuclear_reactions_item.isVisible():
            self.show_nuclear_reactions(True)
            return True
        else:
            self.show_nuclear_reactions(False)
            return False

    def is_nuclear_reactions_visible(self) -> bool:
        """Check if interactive Nuclear Physics & Reactions simulator is currently visible."""
        return self._nuclear_reactions_item is not None and self._nuclear_reactions_item.isVisible()

    def get_nuclear_reactions(self) -> Optional[QGraphicsItem]:
        """Get interactive Nuclear Physics & Reactions simulator item if visible."""
        if self._nuclear_reactions_item is not None and self._nuclear_reactions_item.isVisible():
            return self._nuclear_reactions_item
        return None

    def get_nuclear_reactions_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Nuclear Physics & Reactions simulator item regardless of visibility."""
        return self._nuclear_reactions_item

    def show_doppler_effect(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Doppler Effect & Mach Cone simulator."""
        if self._doppler_effect_item is None:
            from src.ui.interactive_doppler_effect import InteractiveDopplerEffectItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._doppler_effect_item = InteractiveDopplerEffectItem(pos=spawn_pos)
            self.addItem(self._doppler_effect_item)
        else:
            self._doppler_effect_item.setVisible(visible)
            if pos is not None:
                self._doppler_effect_item.setPos(pos[0], pos[1])
        return self._doppler_effect_item if visible else None

    def toggle_doppler_effect(self) -> bool:
        """Toggle interactive Doppler Effect & Mach Cone simulator visibility."""
        if self._doppler_effect_item is None or not self._doppler_effect_item.isVisible():
            self.show_doppler_effect(True)
            return True
        else:
            self.show_doppler_effect(False)
            return False

    def is_doppler_effect_visible(self) -> bool:
        """Check if interactive Doppler Effect & Mach Cone simulator is currently visible."""
        return self._doppler_effect_item is not None and self._doppler_effect_item.isVisible()

    def get_doppler_effect(self) -> Optional[QGraphicsItem]:
        """Get interactive Doppler Effect & Mach Cone simulator item if visible."""
        if self._doppler_effect_item is not None and self._doppler_effect_item.isVisible():
            return self._doppler_effect_item
        return None

    def get_doppler_effect_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Doppler Effect & Mach Cone simulator item regardless of visibility."""
        return self._doppler_effect_item

    def show_red_black_tree(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Red-Black Self-Balancing Tree simulator."""
        if self._red_black_tree_item is None:
            from src.ui.interactive_red_black_tree import InteractiveRedBlackTreeItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._red_black_tree_item = InteractiveRedBlackTreeItem(pos=spawn_pos)
            self.addItem(self._red_black_tree_item)
        else:
            self._red_black_tree_item.setVisible(visible)
            if pos is not None:
                self._red_black_tree_item.setPos(pos[0], pos[1])
        return self._red_black_tree_item if visible else None

    def toggle_red_black_tree(self) -> bool:
        """Toggle interactive Red-Black Self-Balancing Tree simulator visibility."""
        if self._red_black_tree_item is None or not self._red_black_tree_item.isVisible():
            self.show_red_black_tree(True)
            return True
        else:
            self.show_red_black_tree(False)
            return False

    def is_red_black_tree_visible(self) -> bool:
        """Check if interactive Red-Black Self-Balancing Tree simulator is currently visible."""
        return self._red_black_tree_item is not None and self._red_black_tree_item.isVisible()

    def get_red_black_tree(self) -> Optional[QGraphicsItem]:
        """Get interactive Red-Black Self-Balancing Tree simulator item if visible."""
        if self._red_black_tree_item is not None and self._red_black_tree_item.isVisible():
            return self._red_black_tree_item
        return None

    def get_red_black_tree_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Red-Black Self-Balancing Tree simulator item regardless of visibility."""
        return self._red_black_tree_item

    def show_millikan_oil_drop(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Millikan Oil Drop & Elementary Charge simulator."""
        if self._millikan_oil_drop_item is None:
            from src.ui.interactive_millikan_oil_drop import InteractiveMillikanOilDropItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._millikan_oil_drop_item = InteractiveMillikanOilDropItem(pos=spawn_pos)
            self.addItem(self._millikan_oil_drop_item)
        else:
            self._millikan_oil_drop_item.setVisible(visible)
            if pos is not None:
                self._millikan_oil_drop_item.setPos(pos[0], pos[1])
        return self._millikan_oil_drop_item if visible else None

    def toggle_millikan_oil_drop(self) -> bool:
        """Toggle interactive Millikan Oil Drop & Elementary Charge simulator visibility."""
        if self._millikan_oil_drop_item is None or not self._millikan_oil_drop_item.isVisible():
            self.show_millikan_oil_drop(True)
            return True
        else:
            self.show_millikan_oil_drop(False)
            return False

    def is_millikan_oil_drop_visible(self) -> bool:
        """Check if interactive Millikan Oil Drop & Elementary Charge simulator is currently visible."""
        return self._millikan_oil_drop_item is not None and self._millikan_oil_drop_item.isVisible()

    def get_millikan_oil_drop(self) -> Optional[QGraphicsItem]:
        """Get interactive Millikan Oil Drop & Elementary Charge simulator item if visible."""
        if self._millikan_oil_drop_item is not None and self._millikan_oil_drop_item.isVisible():
            return self._millikan_oil_drop_item
        return None

    def get_millikan_oil_drop_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Millikan Oil Drop & Elementary Charge simulator item regardless of visibility."""
        return self._millikan_oil_drop_item

    def show_quantum_well(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Quantum Well & Particle in a Box simulator."""
        if self._quantum_well_item is None:
            from src.ui.interactive_quantum_well import InteractiveQuantumWellItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._quantum_well_item = InteractiveQuantumWellItem(pos=spawn_pos)
            self.addItem(self._quantum_well_item)
        else:
            self._quantum_well_item.setVisible(visible)
            if pos is not None:
                self._quantum_well_item.setPos(pos[0], pos[1])
        return self._quantum_well_item if visible else None

    def toggle_quantum_well(self) -> bool:
        """Toggle interactive Quantum Well & Particle in a Box simulator visibility."""
        if self._quantum_well_item is None or not self._quantum_well_item.isVisible():
            self.show_quantum_well(True)
            return True
        else:
            self.show_quantum_well(False)
            return False

    def is_quantum_well_visible(self) -> bool:
        """Check if interactive Quantum Well & Particle in a Box simulator is currently visible."""
        return self._quantum_well_item is not None and self._quantum_well_item.isVisible()

    def get_quantum_well(self) -> Optional[QGraphicsItem]:
        """Get interactive Quantum Well & Particle in a Box simulator item if visible."""
        if self._quantum_well_item is not None and self._quantum_well_item.isVisible():
            return self._quantum_well_item
        return None

    def get_quantum_well_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Quantum Well & Particle in a Box simulator item regardless of visibility."""
        return self._quantum_well_item

    def show_hohmann_transfer(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Hohmann Transfer Orbit & Keplerian Mechanics simulator."""
        if self._hohmann_transfer_item is None:
            from src.ui.interactive_hohmann_transfer import InteractiveHohmannTransferItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._hohmann_transfer_item = InteractiveHohmannTransferItem(pos=spawn_pos)
            self.addItem(self._hohmann_transfer_item)
        else:
            self._hohmann_transfer_item.setVisible(visible)
            if pos is not None:
                self._hohmann_transfer_item.setPos(pos[0], pos[1])
        return self._hohmann_transfer_item if visible else None

    def toggle_hohmann_transfer(self) -> bool:
        """Toggle interactive Hohmann Transfer Orbit & Keplerian Mechanics simulator visibility."""
        if self._hohmann_transfer_item is None or not self._hohmann_transfer_item.isVisible():
            self.show_hohmann_transfer(True)
            return True
        else:
            self.show_hohmann_transfer(False)
            return False

    def is_hohmann_transfer_visible(self) -> bool:
        """Check if interactive Hohmann Transfer Orbit & Keplerian Mechanics simulator is currently visible."""
        return self._hohmann_transfer_item is not None and self._hohmann_transfer_item.isVisible()

    def get_hohmann_transfer(self) -> Optional[QGraphicsItem]:
        """Get interactive Hohmann Transfer Orbit & Keplerian Mechanics simulator item if visible."""
        if self._hohmann_transfer_item is not None and self._hohmann_transfer_item.isVisible():
            return self._hohmann_transfer_item
        return None

    def get_hohmann_transfer_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Hohmann Transfer Orbit & Keplerian Mechanics simulator item regardless of visibility."""
        return self._hohmann_transfer_item

    def show_photoelectric_effect(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Photoelectric Effect & Einstein Photon simulator."""
        if self._photoelectric_effect_item is None:
            from src.ui.interactive_photoelectric_effect import InteractivePhotoelectricEffectItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._photoelectric_effect_item = InteractivePhotoelectricEffectItem(pos=spawn_pos)
            self.addItem(self._photoelectric_effect_item)
        else:
            self._photoelectric_effect_item.setVisible(visible)
            if pos is not None:
                self._photoelectric_effect_item.setPos(pos[0], pos[1])
        return self._photoelectric_effect_item if visible else None

    def toggle_photoelectric_effect(self) -> bool:
        """Toggle interactive Photoelectric Effect & Einstein Photon simulator visibility."""
        if self._photoelectric_effect_item is None or not self._photoelectric_effect_item.isVisible():
            self.show_photoelectric_effect(True)
            return True
        else:
            self.show_photoelectric_effect(False)
            return False

    def is_photoelectric_effect_visible(self) -> bool:
        """Check if interactive Photoelectric Effect & Einstein Photon simulator is currently visible."""
        return self._photoelectric_effect_item is not None and self._photoelectric_effect_item.isVisible()

    def get_photoelectric_effect(self) -> Optional[QGraphicsItem]:
        """Get interactive Photoelectric Effect & Einstein Photon simulator item if visible."""
        if self._photoelectric_effect_item is not None and self._photoelectric_effect_item.isVisible():
            return self._photoelectric_effect_item
        return None

    def get_photoelectric_effect_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Photoelectric Effect & Einstein Photon simulator item regardless of visibility."""
        return self._photoelectric_effect_item

    def show_punnett_genetics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Mendelian Genetics & Punnett Square simulator."""
        if self._punnett_genetics_item is None:
            from src.ui.interactive_punnett_genetics import InteractivePunnettGeneticsItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._punnett_genetics_item = InteractivePunnettGeneticsItem(pos=spawn_pos)
            self.addItem(self._punnett_genetics_item)
        else:
            self._punnett_genetics_item.setVisible(visible)
            if pos is not None:
                self._punnett_genetics_item.setPos(pos[0], pos[1])
        return self._punnett_genetics_item if visible else None

    def toggle_punnett_genetics(self) -> bool:
        """Toggle interactive Mendelian Genetics & Punnett Square simulator visibility."""
        if self._punnett_genetics_item is None or not self._punnett_genetics_item.isVisible():
            self.show_punnett_genetics(True)
            return True
        else:
            self.show_punnett_genetics(False)
            return False

    def is_punnett_genetics_visible(self) -> bool:
        """Check if interactive Mendelian Genetics & Punnett Square simulator is currently visible."""
        return self._punnett_genetics_item is not None and self._punnett_genetics_item.isVisible()

    def get_punnett_genetics(self) -> Optional[QGraphicsItem]:
        """Get interactive Mendelian Genetics & Punnett Square simulator item if visible."""
        if self._punnett_genetics_item is not None and self._punnett_genetics_item.isVisible():
            return self._punnett_genetics_item
        return None

    def get_punnett_genetics_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Mendelian Genetics & Punnett Square simulator item regardless of visibility."""
        return self._punnett_genetics_item

    def show_reaction_kinetics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Chemical Kinetics & Arrhenius Law simulator."""
        if self._reaction_kinetics_item is None:
            from src.ui.interactive_reaction_kinetics import InteractiveReactionKineticsItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._reaction_kinetics_item = InteractiveReactionKineticsItem(pos=spawn_pos)
            self.addItem(self._reaction_kinetics_item)
        else:
            self._reaction_kinetics_item.setVisible(visible)
            if pos is not None:
                self._reaction_kinetics_item.setPos(pos[0], pos[1])
        return self._reaction_kinetics_item if visible else None

    def toggle_reaction_kinetics(self) -> bool:
        """Toggle interactive Chemical Kinetics & Arrhenius Law simulator visibility."""
        if self._reaction_kinetics_item is None or not self._reaction_kinetics_item.isVisible():
            self.show_reaction_kinetics(True)
            return True
        else:
            self.show_reaction_kinetics(False)
            return False

    def is_reaction_kinetics_visible(self) -> bool:
        """Check if interactive Chemical Kinetics & Arrhenius Law simulator is currently visible."""
        return self._reaction_kinetics_item is not None and self._reaction_kinetics_item.isVisible()

    def get_reaction_kinetics(self) -> Optional[QGraphicsItem]:
        """Get interactive Chemical Kinetics & Arrhenius Law simulator item if visible."""
        if self._reaction_kinetics_item is not None and self._reaction_kinetics_item.isVisible():
            return self._reaction_kinetics_item
        return None

    def get_reaction_kinetics_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Chemical Kinetics & Arrhenius Law simulator item regardless of visibility."""
        return self._reaction_kinetics_item

    def show_calculus_taylor(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Riemann Calculus & Taylor Polynomials simulator."""
        if self._calculus_taylor_item is None:
            from src.ui.interactive_calculus_taylor import InteractiveCalculusTaylorItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._calculus_taylor_item = InteractiveCalculusTaylorItem(pos=spawn_pos)
            self.addItem(self._calculus_taylor_item)
        else:
            self._calculus_taylor_item.setVisible(visible)
            if pos is not None:
                self._calculus_taylor_item.setPos(pos[0], pos[1])
        return self._calculus_taylor_item if visible else None

    def toggle_calculus_taylor(self) -> bool:
        """Toggle interactive Riemann Calculus & Taylor Polynomials simulator visibility."""
        if self._calculus_taylor_item is None or not self._calculus_taylor_item.isVisible():
            self.show_calculus_taylor(True)
            return True
        else:
            self.show_calculus_taylor(False)
            return False

    def is_calculus_taylor_visible(self) -> bool:
        """Check if interactive Riemann Calculus & Taylor Polynomials simulator is currently visible."""
        return self._calculus_taylor_item is not None and self._calculus_taylor_item.isVisible()

    def get_calculus_taylor(self) -> Optional[QGraphicsItem]:
        """Get interactive Riemann Calculus & Taylor Polynomials simulator item if visible."""
        if self._calculus_taylor_item is not None and self._calculus_taylor_item.isVisible():
            return self._calculus_taylor_item
        return None

    def get_calculus_taylor_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Riemann Calculus & Taylor Polynomials simulator item regardless of visibility."""
        return self._calculus_taylor_item

    def show_michelson_interferometer(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Michelson Interferometer & LIGO simulator."""
        if self._michelson_interferometer_item is None:
            from src.ui.interactive_michelson_interferometer import InteractiveMichelsonInterferometerItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._michelson_interferometer_item = InteractiveMichelsonInterferometerItem(pos=spawn_pos)
            self.addItem(self._michelson_interferometer_item)
        else:
            self._michelson_interferometer_item.setVisible(visible)
            if pos is not None:
                self._michelson_interferometer_item.setPos(pos[0], pos[1])
        return self._michelson_interferometer_item if visible else None

    def toggle_michelson_interferometer(self) -> bool:
        """Toggle interactive Michelson Interferometer & LIGO simulator visibility."""
        if self._michelson_interferometer_item is None or not self._michelson_interferometer_item.isVisible():
            self.show_michelson_interferometer(True)
            return True
        else:
            self.show_michelson_interferometer(False)
            return False

    def is_michelson_interferometer_visible(self) -> bool:
        """Check if interactive Michelson Interferometer & LIGO simulator is currently visible."""
        return self._michelson_interferometer_item is not None and self._michelson_interferometer_item.isVisible()

    def get_michelson_interferometer(self) -> Optional[QGraphicsItem]:
        """Get interactive Michelson Interferometer & LIGO simulator item if visible."""
        if self._michelson_interferometer_item is not None and self._michelson_interferometer_item.isVisible():
            return self._michelson_interferometer_item
        return None

    def get_michelson_interferometer_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Michelson Interferometer & LIGO simulator item regardless of visibility."""
        return self._michelson_interferometer_item

    def show_huffman_coding(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Huffman Coding & Shannon Entropy simulator."""
        if self._huffman_coding_item is None:
            from src.ui.interactive_huffman_coding import InteractiveHuffmanCodingItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._huffman_coding_item = InteractiveHuffmanCodingItem(pos=spawn_pos)
            self.addItem(self._huffman_coding_item)
        else:
            self._huffman_coding_item.setVisible(visible)
            if pos is not None:
                self._huffman_coding_item.setPos(pos[0], pos[1])
        return self._huffman_coding_item if visible else None

    def toggle_huffman_coding(self) -> bool:
        """Toggle interactive Huffman Coding & Shannon Entropy simulator visibility."""
        if self._huffman_coding_item is None or not self._huffman_coding_item.isVisible():
            self.show_huffman_coding(True)
            return True
        else:
            self.show_huffman_coding(False)
            return False

    def is_huffman_coding_visible(self) -> bool:
        """Check if interactive Huffman Coding & Shannon Entropy simulator is currently visible."""
        return self._huffman_coding_item is not None and self._huffman_coding_item.isVisible()

    def get_huffman_coding(self) -> Optional[QGraphicsItem]:
        """Get interactive Huffman Coding & Shannon Entropy simulator item if visible."""
        if self._huffman_coding_item is not None and self._huffman_coding_item.isVisible():
            return self._huffman_coding_item
        return None

    def get_huffman_coding_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Huffman Coding & Shannon Entropy simulator item regardless of visibility."""
        return self._huffman_coding_item

    def show_action_potential(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Hodgkin-Huxley Action Potential simulator."""
        if self._action_potential_item is None:
            from src.ui.interactive_action_potential import InteractiveActionPotentialItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._action_potential_item = InteractiveActionPotentialItem(pos=spawn_pos)
            self.addItem(self._action_potential_item)
        else:
            self._action_potential_item.setVisible(visible)
            if pos is not None:
                self._action_potential_item.setPos(pos[0], pos[1])
        return self._action_potential_item if visible else None

    def toggle_action_potential(self) -> bool:
        """Toggle interactive Hodgkin-Huxley Action Potential simulator visibility."""
        if self._action_potential_item is None or not self._action_potential_item.isVisible():
            self.show_action_potential(True)
            return True
        else:
            self.show_action_potential(False)
            return False

    def is_action_potential_visible(self) -> bool:
        """Check if interactive Hodgkin-Huxley Action Potential simulator is currently visible."""
        return self._action_potential_item is not None and self._action_potential_item.isVisible()

    def get_action_potential(self) -> Optional[QGraphicsItem]:
        """Get interactive Hodgkin-Huxley Action Potential simulator item if visible."""
        if self._action_potential_item is not None and self._action_potential_item.isVisible():
            return self._action_potential_item
        return None

    def get_action_potential_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Hodgkin-Huxley Action Potential simulator item regardless of visibility."""
        return self._action_potential_item

    def show_cyclotron(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Cyclotron Particle Accelerator simulator."""
        if self._cyclotron_item is None:
            from src.ui.interactive_cyclotron import InteractiveCyclotronItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._cyclotron_item = InteractiveCyclotronItem(pos=spawn_pos)
            self.addItem(self._cyclotron_item)
        else:
            self._cyclotron_item.setVisible(visible)
            if pos is not None:
                self._cyclotron_item.setPos(pos[0], pos[1])
        return self._cyclotron_item if visible else None

    def toggle_cyclotron(self) -> bool:
        """Toggle interactive Cyclotron Particle Accelerator simulator visibility."""
        if self._cyclotron_item is None or not self._cyclotron_item.isVisible():
            self.show_cyclotron(True)
            return True
        else:
            self.show_cyclotron(False)
            return False

    def is_cyclotron_visible(self) -> bool:
        """Check if interactive Cyclotron Particle Accelerator simulator is currently visible."""
        return self._cyclotron_item is not None and self._cyclotron_item.isVisible()

    def get_cyclotron(self) -> Optional[QGraphicsItem]:
        """Get interactive Cyclotron Particle Accelerator simulator item if visible."""
        if self._cyclotron_item is not None and self._cyclotron_item.isVisible():
            return self._cyclotron_item
        return None

    def get_cyclotron_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Cyclotron Particle Accelerator simulator item regardless of visibility."""
        return self._cyclotron_item

    def show_aho_corasick(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Aho-Corasick Multi-Pattern Automaton simulator."""
        if self._aho_corasick_item is None:
            from src.ui.interactive_aho_corasick import InteractiveAhoCorasickItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._aho_corasick_item = InteractiveAhoCorasickItem(pos=spawn_pos)
            self.addItem(self._aho_corasick_item)
        else:
            self._aho_corasick_item.setVisible(visible)
            if pos is not None:
                self._aho_corasick_item.setPos(pos[0], pos[1])
        return self._aho_corasick_item if visible else None

    def toggle_aho_corasick(self) -> bool:
        """Toggle interactive Aho-Corasick Multi-Pattern Automaton simulator visibility."""
        if self._aho_corasick_item is None or not self._aho_corasick_item.isVisible():
            self.show_aho_corasick(True)
            return True
        else:
            self.show_aho_corasick(False)
            return False

    def is_aho_corasick_visible(self) -> bool:
        """Check if interactive Aho-Corasick Multi-Pattern Automaton simulator is currently visible."""
        return self._aho_corasick_item is not None and self._aho_corasick_item.isVisible()

    def get_aho_corasick(self) -> Optional[QGraphicsItem]:
        """Get interactive Aho-Corasick Multi-Pattern Automaton simulator item if visible."""
        if self._aho_corasick_item is not None and self._aho_corasick_item.isVisible():
            return self._aho_corasick_item
        return None

    def get_aho_corasick_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Aho-Corasick Multi-Pattern Automaton simulator item regardless of visibility."""
        return self._aho_corasick_item

    def show_phase_diagram(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Thermodynamics P-T Phase Diagram simulator."""
        if self._phase_diagram_item is None:
            from src.ui.interactive_phase_diagram import InteractivePhaseDiagramItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._phase_diagram_item = InteractivePhaseDiagramItem(pos=spawn_pos)
            self.addItem(self._phase_diagram_item)
        else:
            self._phase_diagram_item.setVisible(visible)
            if pos is not None:
                self._phase_diagram_item.setPos(pos[0], pos[1])
        return self._phase_diagram_item if visible else None

    def toggle_phase_diagram(self) -> bool:
        """Toggle interactive Thermodynamics P-T Phase Diagram simulator visibility."""
        if self._phase_diagram_item is None or not self._phase_diagram_item.isVisible():
            self.show_phase_diagram(True)
            return True
        else:
            self.show_phase_diagram(False)
            return False

    def is_phase_diagram_visible(self) -> bool:
        """Check if interactive Thermodynamics P-T Phase Diagram simulator is currently visible."""
        return self._phase_diagram_item is not None and self._phase_diagram_item.isVisible()

    def get_phase_diagram(self) -> Optional[QGraphicsItem]:
        """Get interactive Thermodynamics P-T Phase Diagram simulator item if visible."""
        if self._phase_diagram_item is not None and self._phase_diagram_item.isVisible():
            return self._phase_diagram_item
        return None

    def get_phase_diagram_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Thermodynamics P-T Phase Diagram simulator item regardless of visibility."""
        return self._phase_diagram_item

    def show_em_wave(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Maxwell's Electromagnetic Waves simulator."""
        if self._em_wave_item is None:
            from src.ui.interactive_em_wave import InteractiveElectromagneticWaveItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._em_wave_item = InteractiveElectromagneticWaveItem(pos=spawn_pos)
            self.addItem(self._em_wave_item)
        else:
            self._em_wave_item.setVisible(visible)
            if pos is not None:
                self._em_wave_item.setPos(pos[0], pos[1])
        return self._em_wave_item if visible else None

    def toggle_em_wave(self) -> bool:
        """Toggle interactive Maxwell's Electromagnetic Waves simulator visibility."""
        if self._em_wave_item is None or not self._em_wave_item.isVisible():
            self.show_em_wave(True)
            return True
        else:
            self.show_em_wave(False)
            return False

    def is_em_wave_visible(self) -> bool:
        """Check if interactive Maxwell's Electromagnetic Waves simulator is currently visible."""
        return self._em_wave_item is not None and self._em_wave_item.isVisible()

    def get_em_wave(self) -> Optional[QGraphicsItem]:
        """Get interactive Maxwell's Electromagnetic Waves simulator item if visible."""
        if self._em_wave_item is not None and self._em_wave_item.isVisible():
            return self._em_wave_item
        return None

    def get_em_wave_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Maxwell's Electromagnetic Waves simulator item regardless of visibility."""
        return self._em_wave_item

    def show_hardy_weinberg(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Hardy-Weinberg Population Genetics simulator."""
        if self._hardy_weinberg_item is None:
            from src.ui.interactive_hardy_weinberg import InteractiveHardyWeinbergItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._hardy_weinberg_item = InteractiveHardyWeinbergItem(pos=spawn_pos)
            self.addItem(self._hardy_weinberg_item)
        else:
            self._hardy_weinberg_item.setVisible(visible)
            if pos is not None:
                self._hardy_weinberg_item.setPos(pos[0], pos[1])
        return self._hardy_weinberg_item if visible else None

    def toggle_hardy_weinberg(self) -> bool:
        """Toggle interactive Hardy-Weinberg Population Genetics simulator visibility."""
        if self._hardy_weinberg_item is None or not self._hardy_weinberg_item.isVisible():
            self.show_hardy_weinberg(True)
            return True
        else:
            self.show_hardy_weinberg(False)
            return False

    def is_hardy_weinberg_visible(self) -> bool:
        """Check if interactive Hardy-Weinberg Population Genetics simulator is currently visible."""
        return self._hardy_weinberg_item is not None and self._hardy_weinberg_item.isVisible()

    def get_hardy_weinberg(self) -> Optional[QGraphicsItem]:
        """Get interactive Hardy-Weinberg Population Genetics simulator item if visible."""
        if self._hardy_weinberg_item is not None and self._hardy_weinberg_item.isVisible():
            return self._hardy_weinberg_item
        return None

    def get_hardy_weinberg_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Hardy-Weinberg Population Genetics simulator item regardless of visibility."""
        return self._hardy_weinberg_item

    def show_fft(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Cooley-Tukey FFT & Spectrum Analyzer simulator."""
        if self._fft_item is None:
            from src.ui.interactive_fft import InteractiveFFTItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._fft_item = InteractiveFFTItem(pos=spawn_pos)
            self.addItem(self._fft_item)
        else:
            self._fft_item.setVisible(visible)
            if pos is not None:
                self._fft_item.setPos(pos[0], pos[1])
        return self._fft_item if visible else None

    def toggle_fft(self) -> bool:
        """Toggle interactive Cooley-Tukey FFT & Spectrum Analyzer simulator visibility."""
        if self._fft_item is None or not self._fft_item.isVisible():
            self.show_fft(True)
            return True
        else:
            self.show_fft(False)
            return False

    def is_fft_visible(self) -> bool:
        """Check if interactive Cooley-Tukey FFT & Spectrum Analyzer simulator is currently visible."""
        return self._fft_item is not None and self._fft_item.isVisible()

    def get_fft(self) -> Optional[QGraphicsItem]:
        """Get interactive Cooley-Tukey FFT & Spectrum Analyzer simulator item if visible."""
        if self._fft_item is not None and self._fft_item.isVisible():
            return self._fft_item
        return None

    def get_fft_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Cooley-Tukey FFT & Spectrum Analyzer simulator item regardless of visibility."""
        return self._fft_item

    def show_fabry_perot(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Fabry-Pérot Interferometer & Cavity Resonator simulator."""
        if self._fabry_perot_item is None:
            from src.ui.interactive_fabry_perot import InteractiveFabryPerotItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._fabry_perot_item = InteractiveFabryPerotItem(pos=spawn_pos)
            self.addItem(self._fabry_perot_item)
        else:
            self._fabry_perot_item.setVisible(visible)
            if pos is not None:
                self._fabry_perot_item.setPos(pos[0], pos[1])
        return self._fabry_perot_item if visible else None

    def toggle_fabry_perot(self) -> bool:
        """Toggle interactive Fabry-Pérot Interferometer & Cavity Resonator simulator visibility."""
        if self._fabry_perot_item is None or not self._fabry_perot_item.isVisible():
            self.show_fabry_perot(True)
            return True
        else:
            self.show_fabry_perot(False)
            return False

    def is_fabry_perot_visible(self) -> bool:
        """Check if interactive Fabry-Pérot Interferometer simulator is currently visible."""
        return self._fabry_perot_item is not None and self._fabry_perot_item.isVisible()

    def get_fabry_perot(self) -> Optional[QGraphicsItem]:
        """Get interactive Fabry-Pérot Interferometer simulator item if visible."""
        if self._fabry_perot_item is not None and self._fabry_perot_item.isVisible():
            return self._fabry_perot_item
        return None

    def get_fabry_perot_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Fabry-Pérot Interferometer simulator item regardless of visibility."""
        return self._fabry_perot_item

    def show_hplc(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive HPLC & Van Deemter Column Efficiency simulator."""
        if self._hplc_item is None:
            from src.ui.interactive_hplc import InteractiveHPLCItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._hplc_item = InteractiveHPLCItem(pos=spawn_pos)
            self.addItem(self._hplc_item)
        else:
            self._hplc_item.setVisible(visible)
            if pos is not None:
                self._hplc_item.setPos(pos[0], pos[1])
        return self._hplc_item if visible else None

    def toggle_hplc(self) -> bool:
        """Toggle interactive HPLC & Van Deemter Column Efficiency simulator visibility."""
        if self._hplc_item is None or not self._hplc_item.isVisible():
            self.show_hplc(True)
            return True
        else:
            self.show_hplc(False)
            return False

    def is_hplc_visible(self) -> bool:
        """Check if interactive HPLC simulator is currently visible."""
        return self._hplc_item is not None and self._hplc_item.isVisible()

    def get_hplc(self) -> Optional[QGraphicsItem]:
        """Get interactive HPLC simulator item if visible."""
        if self._hplc_item is not None and self._hplc_item.isVisible():
            return self._hplc_item
        return None

    def get_hplc_item(self) -> Optional[QGraphicsItem]:
        """Get interactive HPLC simulator item regardless of visibility."""
        return self._hplc_item

    def show_photoelectric(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Einstein Photoelectric Effect simulator."""
        if self._photoelectric_item is None:
            from src.ui.interactive_photoelectric_effect import InteractivePhotoelectricEffectItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._photoelectric_item = InteractivePhotoelectricEffectItem(pos=spawn_pos)
            self.addItem(self._photoelectric_item)
        else:
            self._photoelectric_item.setVisible(visible)
            if pos is not None:
                self._photoelectric_item.setPos(pos[0], pos[1])
        return self._photoelectric_item if visible else None

    def toggle_photoelectric(self) -> bool:
        """Toggle interactive Einstein Photoelectric Effect simulator visibility."""
        if self._photoelectric_item is None or not self._photoelectric_item.isVisible():
            self.show_photoelectric(True)
            return True
        else:
            self.show_photoelectric(False)
            return False

    def is_photoelectric_visible(self) -> bool:
        """Check if interactive Photoelectric Effect simulator is currently visible."""
        return self._photoelectric_item is not None and self._photoelectric_item.isVisible()

    def get_photoelectric(self) -> Optional[QGraphicsItem]:
        """Get interactive Photoelectric Effect simulator item if visible."""
        if self._photoelectric_item is not None and self._photoelectric_item.isVisible():
            return self._photoelectric_item
        return None

    def get_photoelectric_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Photoelectric Effect simulator item regardless of visibility."""
        return self._photoelectric_item

    def show_pcr(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Polymerase Chain Reaction (PCR) simulator."""
        if self._pcr_item is None:
            from src.ui.interactive_pcr import InteractivePCRItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._pcr_item = InteractivePCRItem(pos=spawn_pos)
            self.addItem(self._pcr_item)
        else:
            self._pcr_item.setVisible(visible)
            if pos is not None:
                self._pcr_item.setPos(pos[0], pos[1])
        return self._pcr_item if visible else None

    def toggle_pcr(self) -> bool:
        """Toggle interactive Polymerase Chain Reaction (PCR) simulator visibility."""
        if self._pcr_item is None or not self._pcr_item.isVisible():
            self.show_pcr(True)
            return True
        else:
            self.show_pcr(False)
            return False

    def is_pcr_visible(self) -> bool:
        """Check if interactive PCR simulator is currently visible."""
        return self._pcr_item is not None and self._pcr_item.isVisible()

    def get_pcr(self) -> Optional[QGraphicsItem]:
        """Get interactive PCR simulator item if visible."""
        if self._pcr_item is not None and self._pcr_item.isVisible():
            return self._pcr_item
        return None

    def get_pcr_item(self) -> Optional[QGraphicsItem]:
        """Get interactive PCR simulator item regardless of visibility."""
        return self._pcr_item

    def show_thin_lens(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Thin Lens Systems & Geometrical Optics simulator."""
        if self._thin_lens_item is None:
            from src.ui.interactive_thin_lens_system import InteractiveThinLensSystemItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._thin_lens_item = InteractiveThinLensSystemItem(pos=spawn_pos)
            self.addItem(self._thin_lens_item)
        else:
            self._thin_lens_item.setVisible(visible)
            if pos is not None:
                self._thin_lens_item.setPos(pos[0], pos[1])
        return self._thin_lens_item if visible else None

    def toggle_thin_lens(self) -> bool:
        """Toggle interactive Thin Lens Systems & Geometrical Optics simulator visibility."""
        if self._thin_lens_item is None or not self._thin_lens_item.isVisible():
            self.show_thin_lens(True)
            return True
        else:
            self.show_thin_lens(False)
            return False

    def is_thin_lens_visible(self) -> bool:
        """Check if interactive Thin Lens simulator is currently visible."""
        return self._thin_lens_item is not None and self._thin_lens_item.isVisible()

    def get_thin_lens(self) -> Optional[QGraphicsItem]:
        """Get interactive Thin Lens simulator item if visible."""
        if self._thin_lens_item is not None and self._thin_lens_item.isVisible():
            return self._thin_lens_item
        return None

    def get_thin_lens_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Thin Lens simulator item regardless of visibility."""
        return self._thin_lens_item

    def show_bragg_diffraction(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Bragg's Law & XRD Powder Crystallography simulator."""
        if self._bragg_diffraction_item is None:
            from src.ui.interactive_bragg_diffraction import InteractiveBraggDiffractionItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._bragg_diffraction_item = InteractiveBraggDiffractionItem(pos=spawn_pos)
            self._bragg_diffraction_item.stamp_requested.connect(self._on_stamp_bragg_diffraction)
            self.addItem(self._bragg_diffraction_item)
        else:
            self._bragg_diffraction_item.setVisible(visible)
            if pos is not None:
                self._bragg_diffraction_item.setPos(pos[0], pos[1])
        return self._bragg_diffraction_item if visible else None

    def toggle_bragg_diffraction(self) -> bool:
        """Toggle interactive Bragg's Law & XRD Powder Crystallography simulator visibility."""
        if self._bragg_diffraction_item is None or not self._bragg_diffraction_item.isVisible():
            self.show_bragg_diffraction(True)
            return True
        else:
            self.show_bragg_diffraction(False)
            return False

    def is_bragg_diffraction_visible(self) -> bool:
        """Return True if Bragg diffraction simulator is currently visible."""
        return self._bragg_diffraction_item is not None and self._bragg_diffraction_item.isVisible()

    def get_bragg_diffraction(self) -> Optional[QGraphicsItem]:
        """Return the Bragg diffraction item if currently visible, otherwise None."""
        if self._bragg_diffraction_item is not None and self._bragg_diffraction_item.isVisible():
            return self._bragg_diffraction_item
        return None

    def get_bragg_diffraction_item(self) -> Optional[QGraphicsItem]:
        """Return the Bragg diffraction item instance regardless of visibility."""
        return self._bragg_diffraction_item

    def _on_stamp_bragg_diffraction(self, data: dict) -> None:
        """Stamp vector diagram of Bragg XRD crystallography onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._bragg_diffraction_item.pos().x() + 340.0,
            self._bragg_diffraction_item.pos().y() + 190.0,
        ) if self._bragg_diffraction_item else (450.0, 250.0)
        items = PhysicsTools.create_bragg_diffraction_diagram(
            center=pos,
            preset_code=data.get("preset_code", "SILICON_POWDER_STANDARD"),
            crystal_type=data.get("crystal_type", "DIAMOND_CUBIC"),
            lattice_a_nm=data.get("lattice_a_nm", 0.5431),
            wavelength_nm=data.get("wavelength_nm", 0.15406),
            crystallite_d_nm=data.get("crystallite_d_nm", 45.0),
            plane_label=data.get("active_plane_label", "(111)"),
            d_spacing_nm=data.get("d_spacing_nm", 0.3135),
            bragg_theta_deg=data.get("bragg_theta_deg", 14.22),
            two_theta_deg=data.get("two_theta_deg", 28.44),
            scherrer_fwhm_deg=data.get("scherrer_fwhm_deg", 0.22),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_buffer_system(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Acid-Base Buffer Systems & Van Slyke Capacity simulator."""
        if self._buffer_system_item is None:
            from src.ui.interactive_buffer_system import InteractiveBufferSystemItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._buffer_system_item = InteractiveBufferSystemItem(pos=spawn_pos)
            self._buffer_system_item.stamp_requested.connect(self._on_stamp_buffer_system)
            self.addItem(self._buffer_system_item)
        else:
            self._buffer_system_item.setVisible(visible)
            if pos is not None:
                self._buffer_system_item.setPos(pos[0], pos[1])
        return self._buffer_system_item if visible else None

    def toggle_buffer_system(self) -> bool:
        """Toggle interactive Acid-Base Buffer Systems simulator visibility."""
        if self._buffer_system_item is None or not self._buffer_system_item.isVisible():
            self.show_buffer_system(True)
            return True
        else:
            self.show_buffer_system(False)
            return False

    def is_buffer_system_visible(self) -> bool:
        """Return True if Buffer System simulator is currently visible."""
        return self._buffer_system_item is not None and self._buffer_system_item.isVisible()

    def get_buffer_system(self) -> Optional[QGraphicsItem]:
        """Return the Buffer System item if currently visible, otherwise None."""
        if self._buffer_system_item is not None and self._buffer_system_item.isVisible():
            return self._buffer_system_item
        return None

    def get_buffer_system_item(self) -> Optional[QGraphicsItem]:
        """Return the Buffer System item instance regardless of visibility."""
        return self._buffer_system_item

    def _on_stamp_buffer_system(self, data: dict) -> None:
        """Stamp vector diagram of Acid-Base Buffer Systems onto the lesson board."""
        from src.ui.stem_tools import ChemistryTools
        pos = (
            self._buffer_system_item.pos().x() + 340.0,
            self._buffer_system_item.pos().y() + 190.0,
        ) if self._buffer_system_item else (450.0, 250.0)
        items = ChemistryTools.create_buffer_system_diagram(
            center=pos,
            preset_code=data.get("preset_code", "ACETIC_ACID_ACETATE_BUFFER"),
            acid_name=data.get("acid_name", "CH₃COOH"),
            base_name=data.get("base_name", "CH₃COO⁻"),
            pKa=data.get("pKa", 4.76),
            c_total_M=data.get("c_total_M", 0.20),
            ratio=data.get("ratio", 1.0),
            current_pH=data.get("current_pH", 4.76),
            van_slyke_capacity=data.get("van_slyke_capacity", 0.115),
            beta_max=data.get("beta_max", 0.115),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_stern_gerlach(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Stern-Gerlach Experiment & Quantum Spin 1/2 simulator."""
        if self._stern_gerlach_item is None:
            from src.ui.interactive_stern_gerlach import InteractiveSternGerlachItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._stern_gerlach_item = InteractiveSternGerlachItem(pos=spawn_pos)
            self._stern_gerlach_item.stamp_requested.connect(self._on_stamp_stern_gerlach)
            self.addItem(self._stern_gerlach_item)
        else:
            self._stern_gerlach_item.setVisible(visible)
            if pos is not None:
                self._stern_gerlach_item.setPos(pos[0], pos[1])
        return self._stern_gerlach_item if visible else None

    def toggle_stern_gerlach(self) -> bool:
        """Toggle interactive Stern-Gerlach Experiment simulator visibility."""
        if self._stern_gerlach_item is None or not self._stern_gerlach_item.isVisible():
            self.show_stern_gerlach(True)
            return True
        else:
            self.show_stern_gerlach(False)
            return False

    def is_stern_gerlach_visible(self) -> bool:
        """Return True if Stern-Gerlach simulator is currently visible."""
        return self._stern_gerlach_item is not None and self._stern_gerlach_item.isVisible()

    def get_stern_gerlach(self) -> Optional[QGraphicsItem]:
        """Return the Stern-Gerlach item if currently visible, otherwise None."""
        if self._stern_gerlach_item is not None and self._stern_gerlach_item.isVisible():
            return self._stern_gerlach_item
        return None

    def get_stern_gerlach_item(self) -> Optional[QGraphicsItem]:
        """Return the Stern-Gerlach item instance regardless of visibility."""
        return self._stern_gerlach_item

    def _on_stamp_stern_gerlach(self, data: dict) -> None:
        """Stamp vector diagram of Stern-Gerlach experiment onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._stern_gerlach_item.pos().x() + 340.0,
            self._stern_gerlach_item.pos().y() + 190.0,
        ) if self._stern_gerlach_item else (450.0, 250.0)
        items = PhysicsTools.create_stern_gerlach_diagram(
            center=pos,
            preset_code=data.get("preset_code", "SINGLE_STAGE_SILVER_ATOMS"),
            gradient_T_per_m=data.get("gradient_T_per_m", 1200.0),
            deflection_mm=data.get("deflection_mm", 2.15),
            p_up=data.get("p_up", 1.0),
            p_down=data.get("p_down", 1.0),
            is_classical=data.get("is_classical", False),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_enzyme_kinetics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Michaelis-Menten Enzyme Kinetics & Lineweaver-Burk simulator."""
        if self._enzyme_kinetics_item is None:
            from src.ui.interactive_enzyme_kinetics import InteractiveEnzymeKineticsItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._enzyme_kinetics_item = InteractiveEnzymeKineticsItem(pos=spawn_pos)
            self._enzyme_kinetics_item.stamp_requested.connect(self._on_stamp_enzyme_kinetics)
            self.addItem(self._enzyme_kinetics_item)
        self._enzyme_kinetics_item.setVisible(visible)
        if visible and pos is not None:
            self._enzyme_kinetics_item.setPos(pos[0], pos[1])
        return self._enzyme_kinetics_item if visible else None

    def toggle_enzyme_kinetics(self) -> bool:
        """Toggles visibility of the Enzyme Kinetics simulator."""
        if self._enzyme_kinetics_item is None or not self._enzyme_kinetics_item.isVisible():
            self.show_enzyme_kinetics(True)
            return True
        else:
            self.show_enzyme_kinetics(False)
            return False

    def is_enzyme_kinetics_visible(self) -> bool:
        """Returns True if the Enzyme Kinetics simulator is currently visible."""
        return self._enzyme_kinetics_item is not None and self._enzyme_kinetics_item.isVisible()

    def get_enzyme_kinetics(self) -> Optional[QGraphicsItem]:
        """Returns the enzyme kinetics item if visible."""
        if self._enzyme_kinetics_item is not None and self._enzyme_kinetics_item.isVisible():
            return self._enzyme_kinetics_item
        return None

    def get_enzyme_kinetics_item(self) -> Optional[QGraphicsItem]:
        """Returns the enzyme kinetics item instance regardless of visibility."""
        return self._enzyme_kinetics_item

    def _on_stamp_enzyme_kinetics(self, data: dict) -> None:
        """Stamp vector diagram of Enzyme Kinetics onto the lesson board."""
        from src.ui.stem_tools import BiologyTools
        pos = (
            self._enzyme_kinetics_item.pos().x() + 340.0,
            self._enzyme_kinetics_item.pos().y() + 190.0,
        ) if self._enzyme_kinetics_item else (450.0, 250.0)
        items = BiologyTools.create_enzyme_kinetics_diagram(
            center=pos,
            preset_code=data.get("preset_code", "CHYMOTRYPSIN_PEPTIDE_CLEAVAGE"),
            enzyme_name=data.get("enzyme_name", "Chymotrypsin"),
            substrate_name=data.get("substrate_name", "ATEE"),
            km_mM=data.get("km_mM", 1.5),
            vmax_uM_per_s=data.get("vmax_uM_per_s", 100.0),
            substrate_mM=data.get("substrate_mM", 1.5),
            v0_uM_per_s=data.get("v0_uM_per_s", 50.0),
            inhibition_type=data.get("inhibition_type", "NONE"),
            alpha=data.get("alpha", 1.0),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_poiseuille_flow(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Hagen-Poiseuille Viscous Tube Flow & Reynolds simulator."""
        if self._poiseuille_flow_item is None:
            from src.ui.interactive_poiseuille_flow import InteractivePoiseuilleFlowItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._poiseuille_flow_item = InteractivePoiseuilleFlowItem(pos=spawn_pos)
            self._poiseuille_flow_item.stamp_requested.connect(self._on_stamp_poiseuille_flow)
            self.addItem(self._poiseuille_flow_item)
        self._poiseuille_flow_item.setVisible(visible)
        if visible and pos is not None:
            self._poiseuille_flow_item.setPos(pos[0], pos[1])
        return self._poiseuille_flow_item if visible else None

    def toggle_poiseuille_flow(self) -> bool:
        """Toggles visibility of the Poiseuille Flow simulator."""
        if self._poiseuille_flow_item is None or not self._poiseuille_flow_item.isVisible():
            self.show_poiseuille_flow(True)
            return True
        else:
            self.show_poiseuille_flow(False)
            return False

    def is_poiseuille_flow_visible(self) -> bool:
        """Returns True if the Poiseuille Flow simulator is currently visible."""
        return self._poiseuille_flow_item is not None and self._poiseuille_flow_item.isVisible()

    def get_poiseuille_flow(self) -> Optional[QGraphicsItem]:
        """Returns the poiseuille flow item if visible."""
        if self._poiseuille_flow_item is not None and self._poiseuille_flow_item.isVisible():
            return self._poiseuille_flow_item
        return None

    def get_poiseuille_flow_item(self) -> Optional[QGraphicsItem]:
        """Returns the poiseuille flow item instance regardless of visibility."""
        return self._poiseuille_flow_item

    def _on_stamp_poiseuille_flow(self, data: dict) -> None:
        """Stamp vector diagram of Poiseuille flow onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._poiseuille_flow_item.pos().x() + 340.0,
            self._poiseuille_flow_item.pos().y() + 190.0,
        ) if self._poiseuille_flow_item else (450.0, 250.0)
        items = PhysicsTools.create_poiseuille_flow_diagram(
            center=pos,
            preset_code=data.get("preset_code", "WATER_CAPILLARY_LAMINAR"),
            fluid_name=data.get("fluid_name", "Nước Cất ở 20°C (Water)"),
            radius_mm=data.get("radius_mm", 0.5),
            delta_p_Pa=data.get("delta_p_Pa", 1200.0),
            viscosity_mPa_s=data.get("viscosity_mPa_s", 1.002),
            pipe_length_m=data.get("pipe_length_m", 0.20),
            flow_rate_mL_per_s=data.get("flow_rate_mL_per_s", 1.25),
            max_velocity_m_per_s=data.get("max_velocity_m_per_s", 0.75),
            average_velocity_m_per_s=data.get("average_velocity_m_per_s", 0.375),
            reynolds_number=data.get("reynolds_number", 450.0),
            flow_regime=data.get("flow_regime", "LAMINAR"),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_cosmic_expansion(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Hubble-Lemaître Law & Cosmic Expansion simulator."""
        if self._cosmic_expansion_item is None:
            from src.ui.interactive_cosmic_expansion import InteractiveCosmicExpansionItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._cosmic_expansion_item = InteractiveCosmicExpansionItem(pos=spawn_pos)
            self._cosmic_expansion_item.stamp_requested.connect(self._on_stamp_cosmic_expansion)
            self.addItem(self._cosmic_expansion_item)
        self._cosmic_expansion_item.setVisible(visible)
        if visible and pos is not None:
            self._cosmic_expansion_item.setPos(pos[0], pos[1])
        return self._cosmic_expansion_item if visible else None

    def toggle_cosmic_expansion(self) -> bool:
        """Toggles visibility of the Cosmic Expansion simulator."""
        if self._cosmic_expansion_item is None or not self._cosmic_expansion_item.isVisible():
            self.show_cosmic_expansion(True)
            return True
        else:
            self.show_cosmic_expansion(False)
            return False

    def is_cosmic_expansion_visible(self) -> bool:
        """Returns True if the Cosmic Expansion simulator is currently visible."""
        return self._cosmic_expansion_item is not None and self._cosmic_expansion_item.isVisible()

    def get_cosmic_expansion(self) -> Optional[QGraphicsItem]:
        """Returns the cosmic expansion item if visible."""
        if self._cosmic_expansion_item is not None and self._cosmic_expansion_item.isVisible():
            return self._cosmic_expansion_item
        return None

    def get_cosmic_expansion_item(self) -> Optional[QGraphicsItem]:
        """Returns the cosmic expansion item instance regardless of visibility."""
        return self._cosmic_expansion_item

    def _on_stamp_cosmic_expansion(self, data: dict) -> None:
        """Stamp vector diagram of Cosmic Expansion onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._cosmic_expansion_item.pos().x() + 340.0,
            self._cosmic_expansion_item.pos().y() + 190.0,
        ) if self._cosmic_expansion_item else (450.0, 250.0)
        items = PhysicsTools.create_cosmic_expansion_diagram(
            center=pos,
            preset_code=data.get("preset_code", "MODERN_PLANCK_LAMBDA_CDM"),
            target_name=data.get("target_name", "Thiên hà Chuẩn Cụm Coma"),
            hubble_constant=data.get("hubble_constant", 67.4),
            distance_mpc=data.get("distance_mpc", 100.0),
            recessional_velocity_km_s=data.get("recessional_velocity_km_s", 6740.0),
            redshift_z=data.get("redshift_z", 0.0225),
            hubble_time_Gyr=data.get("hubble_time_Gyr", 14.51),
            hubble_horizon_distance_Mpc=data.get("hubble_horizon_distance_Mpc", 4448.0),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_seismic_waves(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Seismic Waves & Epicenter Triangulation simulator."""
        if self._seismic_waves_item is None:
            from src.ui.interactive_seismic_waves import InteractiveSeismicWavesItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._seismic_waves_item = InteractiveSeismicWavesItem(pos=spawn_pos)
            self._seismic_waves_item.stamp_requested.connect(self._on_stamp_seismic_waves)
            self.addItem(self._seismic_waves_item)
        self._seismic_waves_item.setVisible(visible)
        if visible and pos is not None:
            self._seismic_waves_item.setPos(pos[0], pos[1])
        return self._seismic_waves_item if visible else None

    def toggle_seismic_waves(self) -> bool:
        """Toggles visibility of the Seismic Waves simulator."""
        if self._seismic_waves_item is None or not self._seismic_waves_item.isVisible():
            self.show_seismic_waves(True)
            return True
        else:
            self.show_seismic_waves(False)
            return False

    def is_seismic_waves_visible(self) -> bool:
        """Returns True if the Seismic Waves simulator is currently visible."""
        return self._seismic_waves_item is not None and self._seismic_waves_item.isVisible()

    def get_seismic_waves(self) -> Optional[QGraphicsItem]:
        """Returns the seismic waves item if visible."""
        if self._seismic_waves_item is not None and self._seismic_waves_item.isVisible():
            return self._seismic_waves_item
        return None

    def get_seismic_waves_item(self) -> Optional[QGraphicsItem]:
        """Returns the seismic waves item instance regardless of visibility."""
        return self._seismic_waves_item

    def _on_stamp_seismic_waves(self, data: dict) -> None:
        """Stamp vector diagram of Seismic Waves onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._seismic_waves_item.pos().x() + 340.0,
            self._seismic_waves_item.pos().y() + 190.0,
        ) if self._seismic_waves_item else (450.0, 250.0)
        items = PhysicsTools.create_seismic_waves_diagram(
            center=pos,
            preset_code=data.get("preset_code", "TOHOKU_2011_MEGATHRUST"),
            event_name=data.get("event_name", "Trận động đất Tohoku-Oki 2011"),
            epicenter_x=data.get("epicenter_x", 140.0),
            epicenter_y=data.get("epicenter_y", -60.0),
            magnitude=data.get("magnitude", 9.1),
            vp=data.get("vp", 7.6),
            vs=data.get("vs", 4.3),
            depth_km=data.get("depth_km", 29.0),
            dist_a_km=data.get("dist_a_km", 300.0),
            sp_lag_a_s=data.get("sp_lag_a_s", 30.0),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_mass_spectrometer(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Mass Spectrometer & Wien Filter simulator."""
        if self._mass_spectrometer_item is None:
            from src.ui.interactive_mass_spectrometer import InteractiveMassSpectrometerItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._mass_spectrometer_item = InteractiveMassSpectrometerItem(pos=spawn_pos)
            self._mass_spectrometer_item.stamp_requested.connect(self._on_stamp_mass_spectrometer)
            self.addItem(self._mass_spectrometer_item)
        self._mass_spectrometer_item.setVisible(visible)
        if visible and pos is not None:
            self._mass_spectrometer_item.setPos(pos[0], pos[1])
        return self._mass_spectrometer_item if visible else None

    def toggle_mass_spectrometer(self) -> bool:
        """Toggles visibility of the Mass Spectrometer simulator."""
        if self._mass_spectrometer_item is None or not self._mass_spectrometer_item.isVisible():
            self.show_mass_spectrometer(True)
            return True
        else:
            self.show_mass_spectrometer(False)
            return False

    def is_mass_spectrometer_visible(self) -> bool:
        """Returns True if the Mass Spectrometer simulator is currently visible."""
        return self._mass_spectrometer_item is not None and self._mass_spectrometer_item.isVisible()

    def get_mass_spectrometer(self) -> Optional[QGraphicsItem]:
        """Returns the mass spectrometer item if visible."""
        if self._mass_spectrometer_item is not None and self._mass_spectrometer_item.isVisible():
            return self._mass_spectrometer_item
        return None

    def get_mass_spectrometer_item(self) -> Optional[QGraphicsItem]:
        """Returns the mass spectrometer item instance regardless of visibility."""
        return self._mass_spectrometer_item

    def _on_stamp_mass_spectrometer(self, data: dict) -> None:
        """Stamp vector diagram of Mass Spectrometer onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._mass_spectrometer_item.pos().x() + 340.0,
            self._mass_spectrometer_item.pos().y() + 190.0,
        ) if self._mass_spectrometer_item else (450.0, 250.0)
        items = PhysicsTools.create_mass_spectrometer_diagram(
            center=pos,
            preset_code=data.get("preset_code", "CHLORINE_ISOTOPE_35_37"),
            element_name=data.get("element_name", "Clo (Chlorine)"),
            e_field_V_m=data.get("e_field_V_m", 50000.0),
            b1_Tesla=data.get("b1_Tesla", 0.20),
            b2_Tesla=data.get("b2_Tesla", 0.30),
            velocity_m_s=data.get("velocity_m_s", 250000.0),
            iso0_symbol=data.get("iso0_symbol", "³⁵Cl⁺"),
            iso0_radius_mm=data.get("iso0_radius_mm", 45.3),
            iso1_symbol=data.get("iso1_symbol", "³⁷Cl⁺"),
            iso1_radius_mm=data.get("iso1_radius_mm", 47.9),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_spectrophotometry(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Spectrophotometry & Beer-Lambert Law simulator."""
        if self._spectrophotometry_item is None:
            from src.ui.interactive_spectrophotometry import InteractiveSpectrophotometryItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._spectrophotometry_item = InteractiveSpectrophotometryItem(pos=spawn_pos)
            self._spectrophotometry_item.stamp_requested.connect(self._on_stamp_spectrophotometry)
            self.addItem(self._spectrophotometry_item)
        self._spectrophotometry_item.setVisible(visible)
        if visible and pos is not None:
            self._spectrophotometry_item.setPos(pos[0], pos[1])
        return self._spectrophotometry_item if visible else None

    def toggle_spectrophotometry(self) -> bool:
        """Toggles visibility of the Spectrophotometry simulator."""
        if self._spectrophotometry_item is None or not self._spectrophotometry_item.isVisible():
            self.show_spectrophotometry(True)
            return True
        else:
            self.show_spectrophotometry(False)
            return False

    def is_spectrophotometry_visible(self) -> bool:
        """Returns True if the Spectrophotometry simulator is currently visible."""
        return self._spectrophotometry_item is not None and self._spectrophotometry_item.isVisible()

    def get_spectrophotometry(self) -> Optional[QGraphicsItem]:
        """Returns the spectrophotometry item if visible."""
        if self._spectrophotometry_item is not None and self._spectrophotometry_item.isVisible():
            return self._spectrophotometry_item
        return None

    def get_spectrophotometry_item(self) -> Optional[QGraphicsItem]:
        """Returns the spectrophotometry item instance regardless of visibility."""
        return self._spectrophotometry_item

    def _on_stamp_spectrophotometry(self, data: dict) -> None:
        """Stamp vector diagram of Spectrophotometry onto the lesson board."""
        from src.ui.stem_tools import ChemistryTools
        pos = (
            self._spectrophotometry_item.pos().x() + 340.0,
            self._spectrophotometry_item.pos().y() + 190.0,
        ) if self._spectrophotometry_item else (450.0, 250.0)
        items = ChemistryTools.create_spectrophotometry_diagram(
            center=pos,
            preset_code=data.get("preset_code", "POTASSIUM_PERMANGANATE_KMnO4"),
            analyte_name=data.get("analyte_name", "KMnO₄ (Potassium Permanganate)"),
            conc_mM=data.get("conc_mM", 0.40),
            path_cm=data.get("path_cm", 1.0),
            wavelength_nm=data.get("wavelength_nm", 525.0),
            epsilon=data.get("epsilon", 2400.0),
            absorbance=data.get("absorbance", 0.96),
            transmittance_pct=data.get("transmittance_pct", 10.96),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_nuclear_binding_energy(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Nuclear Binding Energy & SEMF simulator."""
        if self._nuclear_binding_energy_item is None:
            from src.ui.interactive_nuclear_binding_energy import InteractiveNuclearBindingEnergyItem
            spawn_pos = pos if pos is not None else (130.0, 45.0)
            self._nuclear_binding_energy_item = InteractiveNuclearBindingEnergyItem(pos=spawn_pos)
            self._nuclear_binding_energy_item.stamp_requested.connect(self._on_stamp_nuclear_binding_energy)
            self.addItem(self._nuclear_binding_energy_item)
        self._nuclear_binding_energy_item.setVisible(visible)
        if visible and pos is not None:
            self._nuclear_binding_energy_item.setPos(pos[0], pos[1])
        return self._nuclear_binding_energy_item if visible else None

    def toggle_nuclear_binding_energy(self) -> bool:
        """Toggles the visibility of the Interactive Nuclear Binding Energy simulator."""
        is_vis = self._nuclear_binding_energy_item is not None and self._nuclear_binding_energy_item.isVisible()
        self.show_nuclear_binding_energy(not is_vis)
        return not is_vis

    def _on_stamp_nuclear_binding_energy(self, data: dict) -> None:
        """Stamp the Nuclear Binding Energy permanent vector diagram on the board."""
        from src.ui.stem_tools import PhysicsTools
        items = PhysicsTools.create_nuclear_binding_energy_diagram(
            center=data.get("center", (430.0, 310.0)),
            A=data.get("A", 56),
            Z=data.get("Z", 26),
            preset_name=data.get("preset_name", "Sắt-56 (Peak Stability)"),
            symbol=data.get("symbol", "⁵⁶₂₆Fe"),
            description=data.get("description", "Đỉnh bền vững hạt nhân cao nhất vũ trụ"),
            total_be_mev=data.get("total_be_mev", 492.2),
            be_per_nucleon_mev=data.get("be_per_nucleon_mev", 8.79),
            mass_defect_u=data.get("mass_defect_u", 0.5284),
            e_vol=data.get("e_vol", 884.8),
            e_surf=data.get("e_surf", -268.0),
            e_coul=data.get("e_coul", -121.3),
            e_asym=data.get("e_asym", -4.1),
            e_pair=data.get("e_pair", 1.6),
        )
        for itm in items:
            self.addItem(itm)
        if self._nuclear_binding_energy_item is not None:
            self._nuclear_binding_energy_item.setVisible(False)

    def show_compton_scattering(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Compton Scattering & Relativistic Collision simulator."""
        if self._compton_scattering_item is None:
            from src.ui.interactive_compton_scattering import InteractiveComptonScatteringItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._compton_scattering_item = InteractiveComptonScatteringItem(pos=spawn_pos)
            self._compton_scattering_item.stamp_requested.connect(self._on_stamp_compton_scattering)
            self.addItem(self._compton_scattering_item)
        self._compton_scattering_item.setVisible(visible)
        if visible and pos is not None:
            self._compton_scattering_item.setPos(pos[0], pos[1])
        return self._compton_scattering_item if visible else None

    def toggle_compton_scattering(self) -> bool:
        """Toggles the visibility of the Interactive Compton Scattering simulator."""
        is_vis = self._compton_scattering_item is not None and self._compton_scattering_item.isVisible()
        self.show_compton_scattering(not is_vis)
        return not is_vis

    def _on_stamp_compton_scattering(self, data: dict) -> None:
        """Stamp the Compton Scattering permanent vector diagram on the board."""
        from src.ui.stem_tools import PhysicsTools
        items = PhysicsTools.create_compton_scattering_diagram(
            center=data.get("center", (420.0, 320.0)),
            incident_energy_kev=data.get("incident_energy_kev", 511.0),
            scattering_angle_deg=data.get("scattering_angle_deg", 60.0),
            preset_name=data.get("preset_name", "Tán xạ Compton (Cobalt-60 / 511 keV)"),
            target_material=data.get("target_material", "Carbon Graphite (e⁻ tự do)"),
            delta_lambda_pm=data.get("delta_lambda_pm", 1.213),
            lambda_scatter_pm=data.get("lambda_scatter_pm", 3.640),
            energy_scatter_kev=data.get("energy_scatter_kev", 340.7),
            electron_kinetic_kev=data.get("electron_kinetic_kev", 170.3),
            phi_deg=data.get("phi_deg", 40.9),
        )
        for itm in items:
            self.addItem(itm)
        if self._compton_scattering_item is not None:
            self._compton_scattering_item.setVisible(False)

    def show_gravitational_lensing(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Gravitational Lensing & General Relativity simulator."""
        if self._gravitational_lensing_item is None:
            from src.ui.interactive_gravitational_lensing import InteractiveGravitationalLensingItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._gravitational_lensing_item = InteractiveGravitationalLensingItem(pos=spawn_pos)
            self._gravitational_lensing_item.stamp_requested.connect(self._on_stamp_gravitational_lensing)
            self.addItem(self._gravitational_lensing_item)
        self._gravitational_lensing_item.setVisible(visible)
        if visible and pos is not None:
            self._gravitational_lensing_item.setPos(pos[0], pos[1])
        return self._gravitational_lensing_item if visible else None

    def toggle_gravitational_lensing(self) -> bool:
        """Toggles the visibility of the Interactive Gravitational Lensing simulator."""
        is_vis = self._gravitational_lensing_item is not None and self._gravitational_lensing_item.isVisible()
        self.show_gravitational_lensing(not is_vis)
        return not is_vis

    def _on_stamp_gravitational_lensing(self, data: dict) -> None:
        """Stamp the Gravitational Lensing permanent vector diagram on the board."""
        from src.ui.stem_tools import PhysicsTools
        items = PhysicsTools.create_gravitational_lensing_diagram(
            center=data.get("center", (440.0, 310.0)),
            preset_name=data.get("preset_name", "Chiếc Nhẫn Einstein Hoàn Mỹ"),
            description=data.get("description", "Nguồn sáng thẳng hàng tuyệt đối với thấu kính hấp dẫn"),
            lens_mass_solar=data.get("lens_mass_solar", 1.0e12),
            source_offset_arcsec=data.get("source_offset_arcsec", 0.5),
            theta_e_arcsec=data.get("theta_e_arcsec", 1.60),
            theta_plus_arcsec=data.get("theta_plus_arcsec", 1.869),
            theta_minus_arcsec=data.get("theta_minus_arcsec", 1.369),
            mu_tot=data.get("mu_tot", 5.20),
            is_quadrupole=data.get("is_quadrupole", False),
        )
        for itm in items:
            self.addItem(itm)
        if self._gravitational_lensing_item is not None:
            self._gravitational_lensing_item.setVisible(False)

    def show_chemical_equilibrium(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Chemical Equilibrium & Le Chatelier simulator."""
        if self._chemical_equilibrium_item is None:
            from src.ui.interactive_chemical_equilibrium import InteractiveChemicalEquilibriumItem
            spawn_pos = pos if pos is not None else (135.0, 48.0)
            self._chemical_equilibrium_item = InteractiveChemicalEquilibriumItem(pos=spawn_pos)
            self._chemical_equilibrium_item.stamp_requested.connect(self._on_stamp_chemical_equilibrium)
            self.addItem(self._chemical_equilibrium_item)
        self._chemical_equilibrium_item.setVisible(visible)
        if visible and pos is not None:
            self._chemical_equilibrium_item.setPos(pos[0], pos[1])
        return self._chemical_equilibrium_item if visible else None

    def toggle_chemical_equilibrium(self) -> bool:
        """Toggles the visibility of the Interactive Chemical Equilibrium simulator."""
        is_vis = self._chemical_equilibrium_item is not None and self._chemical_equilibrium_item.isVisible()
        self.show_chemical_equilibrium(not is_vis)
        return not is_vis

    def _on_stamp_chemical_equilibrium(self, data: dict) -> None:
        """Stamp the Chemical Equilibrium permanent vector diagram on the board."""
        from src.ui.stem_tools import ChemistryTools
        items = ChemistryTools.create_chemical_equilibrium_diagram(
            center=data.get("center", (440.0, 310.0)),
            preset_key=data.get("preset_key", "HABER_BOSCH"),
            preset_name=data.get("preset_name", "Tổng Hợp Amoniac (Haber-Bosch)"),
            equation_display=data.get("equation_display", "N₂(k) + 3H₂(k) ⇌ 2NH₃(k)"),
            description=data.get("description", "Phản ứng tỏa nhiệt, giảm số mol khí"),
            temp_k=data.get("temp_k", 673.15),
            temp_c=data.get("temp_c", 400.0),
            pressure_atm=data.get("pressure_atm", 200.0),
            delta_g_kj=data.get("delta_g_kj", 41.5),
            kp=data.get("kp", 6.0e-4),
            kx=data.get("kx", 24.0),
            xi=data.get("xi", 0.52),
            yield_pct=data.get("yield_pct", 52.0),
            mole_fractions=data.get("mole_fractions"),
            catalyst_enabled=data.get("catalyst_enabled", True),
        )
        for itm in items:
            self.addItem(itm)
        if self._chemical_equilibrium_item is not None:
            self._chemical_equilibrium_item.setVisible(False)

    def show_hodgkin_huxley(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Hodgkin-Huxley Neuron ODE simulator."""
        if self._hodgkin_huxley_item is None:
            from src.ui.interactive_hodgkin_huxley import InteractiveHodgkinHuxleyItem
            spawn_pos = pos if pos is not None else (140.0, 52.0)
            self._hodgkin_huxley_item = InteractiveHodgkinHuxleyItem(pos=spawn_pos)
            self._hodgkin_huxley_item.stamp_requested.connect(self._on_stamp_hodgkin_huxley)
            self.addItem(self._hodgkin_huxley_item)
        self._hodgkin_huxley_item.setVisible(visible)
        if visible and pos is not None:
            self._hodgkin_huxley_item.setPos(pos[0], pos[1])
        return self._hodgkin_huxley_item if visible else None

    def toggle_hodgkin_huxley(self) -> bool:
        """Toggles the visibility of the Interactive Hodgkin-Huxley simulator."""
        is_vis = self._hodgkin_huxley_item is not None and self._hodgkin_huxley_item.isVisible()
        self.show_hodgkin_huxley(not is_vis)
        return not is_vis

    def _on_stamp_hodgkin_huxley(self, data: dict) -> None:
        """Stamp the Hodgkin-Huxley permanent vector diagram on the board."""
        from src.ui.stem_tools import BiologyTools
        items = BiologyTools.create_hodgkin_huxley_diagram(
            center=data.get("center", (440.0, 310.0)),
            preset_key=data.get("preset_key", "SINGLE_ACTION_POTENTIAL"),
            preset_name=data.get("preset_name", "Điện Thế Hoạt Động Đơn Cực (Action Potential)"),
            description=data.get("description", "Khử cực màng vượt ngưỡng bùng phát đỉnh spike"),
            stim_amp=data.get("stim_amp", 10.0),
            stim_duration_ms=data.get("stim_duration_ms", 2.0),
            v_max=data.get("v_max", 35.0),
            v_min=data.get("v_min", -75.0),
            has_spike=data.get("has_spike", True),
            ttx_blocked=data.get("ttx_blocked", False),
            tea_blocked=data.get("tea_blocked", False),
        )
        for itm in items:
            self.addItem(itm)
        if self._hodgkin_huxley_item is not None:
            self._hodgkin_huxley_item.setVisible(False)

    def show_mst_visualizer(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Minimum Spanning Tree (MST) simulator."""
        if self._mst_visualizer_item is None:
            from src.ui.interactive_mst_visualizer import InteractiveMstVisualizerItem
            spawn_pos = pos if pos is not None else (145.0, 50.0)
            self._mst_visualizer_item = InteractiveMstVisualizerItem(pos=spawn_pos)
            self._mst_visualizer_item.stamp_requested.connect(self._on_stamp_mst_visualizer)
            self.addItem(self._mst_visualizer_item)
        self._mst_visualizer_item.setVisible(visible)
        if visible and pos is not None:
            self._mst_visualizer_item.setPos(pos[0], pos[1])
        return self._mst_visualizer_item if visible else None

    def toggle_mst_visualizer(self) -> bool:
        """Toggles the visibility of the Interactive Minimum Spanning Tree simulator."""
        is_vis = self._mst_visualizer_item is not None and self._mst_visualizer_item.isVisible()
        self.show_mst_visualizer(not is_vis)
        return not is_vis

    def _on_stamp_mst_visualizer(self, data: dict) -> None:
        """Stamp the Minimum Spanning Tree permanent vector diagram on the board."""
        from src.ui.stem_tools import InformaticsTools
        items = InformaticsTools.create_mst_diagram(
            center=data.get("center", (440.0, 310.0)),
            preset_key=data.get("preset_key", "STANDARD_NETWORK"),
            preset_name=data.get("preset_name", "Mạng Chuẩn Sách Giáo Khoa (7 Đỉnh, 11 Cạnh)"),
            description=data.get("description", "Cây khung nhỏ nhất không chứa chu trình"),
            algorithm=data.get("algorithm", "KRUSKAL"),
            mst_weight=data.get("mst_weight", 37),
            num_nodes=data.get("num_nodes", 7),
            num_edges=data.get("num_edges", 11),
            accepted_edges=data.get("accepted_edges"),
            rejected_edges=data.get("rejected_edges"),
        )
        for itm in items:
            self.addItem(itm)
        if self._mst_visualizer_item is not None:
            self._mst_visualizer_item.setVisible(False)

    def show_hall_effect(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Hall Effect & Semiconductor Lorentz Drift simulator."""
        if self._hall_effect_item is None:
            from src.ui.interactive_hall_effect import InteractiveHallEffectItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._hall_effect_item = InteractiveHallEffectItem(pos=spawn_pos)
            self._hall_effect_item.stamp_requested.connect(self._on_stamp_hall_effect)
            self.addItem(self._hall_effect_item)
        self._hall_effect_item.setVisible(visible)
        if visible and pos is not None:
            self._hall_effect_item.setPos(pos[0], pos[1])
        return self._hall_effect_item if visible else None

    def toggle_hall_effect(self) -> bool:
        """Toggles visibility of the Hall Effect simulator."""
        if self._hall_effect_item is None or not self._hall_effect_item.isVisible():
            self.show_hall_effect(True)
            return True
        else:
            self.show_hall_effect(False)
            return False

    def is_hall_effect_visible(self) -> bool:
        """Returns True if the Hall Effect simulator is currently visible."""
        return self._hall_effect_item is not None and self._hall_effect_item.isVisible()

    def get_hall_effect(self) -> Optional[QGraphicsItem]:
        """Returns the Hall Effect item if visible."""
        if self._hall_effect_item is not None and self._hall_effect_item.isVisible():
            return self._hall_effect_item
        return None

    def get_hall_effect_item(self) -> Optional[QGraphicsItem]:
        """Returns the Hall Effect item instance regardless of visibility."""
        return self._hall_effect_item

    def _on_stamp_hall_effect(self, data: dict) -> None:
        """Stamp vector diagram of Hall Effect onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._hall_effect_item.pos().x() + 340.0,
            self._hall_effect_item.pos().y() + 190.0,
        ) if self._hall_effect_item else (450.0, 250.0)
        items = PhysicsTools.create_hall_effect_diagram(
            center=pos,
            preset_code=data.get("preset_code", "INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR"),
            material_name=data.get("material_name", "Indium Antimonide (InSb n-type)"),
            carrier_type=data.get("carrier_type", "electron"),
            current_mA=data.get("current_mA", 20.0),
            b_field_T=data.get("b_field_T", 0.50),
            thickness_um=data.get("thickness_um", 10.0),
            carrier_density_m3=data.get("carrier_density_m3", 2.0e22),
            hall_voltage_V=data.get("hall_voltage_V", -0.0156),
            hall_coefficient_m3_C=data.get("hall_coefficient_m3_C", -3.12e-4),
            drift_velocity_m_s=data.get("drift_velocity_m_s", 1248.0),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_radiation_shielding(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Nuclear Radiation Shielding simulator."""
        if self._radiation_shielding_item is None:
            from src.ui.interactive_radiation_shielding import InteractiveRadiationShieldingItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._radiation_shielding_item = InteractiveRadiationShieldingItem(pos=spawn_pos)
            self._radiation_shielding_item.stamp_requested.connect(self._on_stamp_radiation_shielding)
            self.addItem(self._radiation_shielding_item)
        self._radiation_shielding_item.setVisible(visible)
        if visible and pos is not None:
            self._radiation_shielding_item.setPos(pos[0], pos[1])
        return self._radiation_shielding_item if visible else None

    def toggle_radiation_shielding(self) -> bool:
        """Toggles visibility of the Radiation Shielding simulator."""
        if self._radiation_shielding_item is None or not self._radiation_shielding_item.isVisible():
            self.show_radiation_shielding(True)
            return True
        else:
            self.show_radiation_shielding(False)
            return False

    def is_radiation_shielding_visible(self) -> bool:
        """Returns True if the Radiation Shielding simulator is currently visible."""
        return self._radiation_shielding_item is not None and self._radiation_shielding_item.isVisible()

    def get_radiation_shielding(self) -> Optional[QGraphicsItem]:
        """Returns the Radiation Shielding item if visible."""
        if self._radiation_shielding_item is not None and self._radiation_shielding_item.isVisible():
            return self._radiation_shielding_item
        return None

    def get_radiation_shielding_item(self) -> Optional[QGraphicsItem]:
        """Returns the Radiation Shielding item instance regardless of visibility."""
        return self._radiation_shielding_item

    def _on_stamp_radiation_shielding(self, data: dict) -> None:
        """Stamp vector diagram of Radiation Shielding onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._radiation_shielding_item.pos().x() + 340.0,
            self._radiation_shielding_item.pos().y() + 190.0,
        ) if self._radiation_shielding_item else (450.0, 250.0)
        items = PhysicsTools.create_radiation_shielding_diagram(
            center=pos,
            preset_code=data.get("preset_code", "COBALT_60_GAMMA_LEAD_SHIELDING"),
            isotope_name=data.get("isotope_name", "Cobalt-60 (⁶⁰Co, γ = 1.25 MeV)"),
            radiation_type=data.get("radiation_type", "gamma"),
            absorber_material=data.get("absorber_material", "Chì (Lead - Pb)"),
            thickness_cm=data.get("thickness_cm", 2.50),
            i0_cps=data.get("i0_cps", 1000.0),
            i_trans_cps=data.get("i_trans_cps", 194.5),
            transmission_pct=data.get("transmission_pct", 19.45),
            hvl_cm=data.get("hvl_cm", 1.058),
            tvl_cm=data.get("tvl_cm", 3.515),
            linear_mu_cm_inv=data.get("linear_mu_cm_inv", 0.655),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_maxwell_boltzmann(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Maxwell-Boltzmann Speed Distribution simulator."""
        if self._maxwell_boltzmann_item is None:
            from src.ui.interactive_maxwell_boltzmann import InteractiveMaxwellBoltzmannItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._maxwell_boltzmann_item = InteractiveMaxwellBoltzmannItem(pos=spawn_pos)
            self._maxwell_boltzmann_item.stamp_requested.connect(self._on_stamp_maxwell_boltzmann)
            self.addItem(self._maxwell_boltzmann_item)
        self._maxwell_boltzmann_item.setVisible(visible)
        if visible and pos is not None:
            self._maxwell_boltzmann_item.setPos(pos[0], pos[1])
        return self._maxwell_boltzmann_item if visible else None

    def toggle_maxwell_boltzmann(self) -> bool:
        """Toggles visibility of the Maxwell-Boltzmann simulator."""
        if self._maxwell_boltzmann_item is None or not self._maxwell_boltzmann_item.isVisible():
            self.show_maxwell_boltzmann(True)
            return True
        else:
            self.show_maxwell_boltzmann(False)
            return False

    def is_maxwell_boltzmann_visible(self) -> bool:
        """Returns True if the Maxwell-Boltzmann simulator is currently visible."""
        return self._maxwell_boltzmann_item is not None and self._maxwell_boltzmann_item.isVisible()

    def get_maxwell_boltzmann(self) -> Optional[QGraphicsItem]:
        """Returns the Maxwell-Boltzmann item if visible."""
        if self._maxwell_boltzmann_item is not None and self._maxwell_boltzmann_item.isVisible():
            return self._maxwell_boltzmann_item
        return None

    def get_maxwell_boltzmann_item(self) -> Optional[QGraphicsItem]:
        """Returns the Maxwell-Boltzmann item instance regardless of visibility."""
        return self._maxwell_boltzmann_item

    def _on_stamp_maxwell_boltzmann(self, data: dict) -> None:
        """Stamp vector diagram of Maxwell-Boltzmann Distribution onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._maxwell_boltzmann_item.pos().x() + 340.0,
            self._maxwell_boltzmann_item.pos().y() + 190.0,
        ) if self._maxwell_boltzmann_item else (450.0, 250.0)
        items = PhysicsTools.create_maxwell_boltzmann_diagram(
            center=pos,
            preset_code=data.get("preset_code", "HELIUM_GAS_ROOM_TEMP_300K"),
            gas_name=data.get("gas_name", "Helium (He)"),
            formula=data.get("formula", "⁴He"),
            temperature_K=data.get("temperature_K", 300.0),
            pressure_atm=data.get("pressure_atm", 1.0),
            molar_mass_kg_mol=data.get("molar_mass_kg_mol", 0.0040026),
            vp_m_s=data.get("vp_m_s", 1115.8),
            v_avg_m_s=data.get("v_avg_m_s", 1259.0),
            v_rms_m_s=data.get("v_rms_m_s", 1366.6),
            mean_free_path_nm=data.get("mean_free_path_nm", 197.5),
            collision_freq_GHz=data.get("collision_freq_GHz", 6.37),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_molecular_orbital(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Molecular Orbital & Hybridization simulator."""
        if self._molecular_orbital_item is None:
            from src.ui.interactive_molecular_orbital import InteractiveMolecularOrbitalItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._molecular_orbital_item = InteractiveMolecularOrbitalItem(pos=spawn_pos)
            self._molecular_orbital_item.stamp_requested.connect(self._on_stamp_molecular_orbital)
            self.addItem(self._molecular_orbital_item)
        self._molecular_orbital_item.setVisible(visible)
        if visible and pos is not None:
            self._molecular_orbital_item.setPos(pos[0], pos[1])
        return self._molecular_orbital_item if visible else None

    def toggle_molecular_orbital(self) -> bool:
        """Toggles visibility of the Molecular Orbital simulator."""
        if self._molecular_orbital_item is None or not self._molecular_orbital_item.isVisible():
            self.show_molecular_orbital(True)
            return True
        else:
            self.show_molecular_orbital(False)
            return False

    def is_molecular_orbital_visible(self) -> bool:
        """Returns True if the Molecular Orbital simulator is currently visible."""
        return self._molecular_orbital_item is not None and self._molecular_orbital_item.isVisible()

    def get_molecular_orbital(self) -> Optional[QGraphicsItem]:
        """Returns the Molecular Orbital item if visible."""
        if self._molecular_orbital_item is not None and self._molecular_orbital_item.isVisible():
            return self._molecular_orbital_item
        return None

    def get_molecular_orbital_item(self) -> Optional[QGraphicsItem]:
        """Returns the Molecular Orbital item instance regardless of visibility."""
        return self._molecular_orbital_item

    def _on_stamp_molecular_orbital(self, data: dict) -> None:
        """Stamp vector diagram of Molecular Orbital onto the lesson board."""
        from src.ui.stem_tools import ChemistryTools
        pos = (
            self._molecular_orbital_item.pos().x() + 340.0,
            self._molecular_orbital_item.pos().y() + 190.0,
        ) if self._molecular_orbital_item else (450.0, 250.0)
        items = ChemistryTools.create_molecular_orbital_diagram(
            center=pos,
            preset_code=data.get("preset_code", "HYDROGEN_MOLECULE_H2_SIGMA_BOND"),
            molecule_name=data.get("molecule_name", "Hydrogen Molecule (H₂)"),
            orbital_type=data.get("orbital_type", "mo_sigma"),
            distance_angstrom=data.get("distance_angstrom", 0.74),
            overlap_integral_S=data.get("overlap_integral_S", 0.753),
            bond_order=data.get("bond_order", 1.0),
            is_antibonding=data.get("is_antibonding", False),
            hybridization_str=data.get("hybridization_str", "1s + 1s (Không lai hóa)"),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_rlc_resonance(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive RLC AC Resonance & Phasor simulator."""
        if self._rlc_resonance_item is None:
            from src.ui.interactive_rlc_resonance import InteractiveRLCResonanceItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._rlc_resonance_item = InteractiveRLCResonanceItem(pos=spawn_pos)
            self._rlc_resonance_item.stamp_requested.connect(self._on_stamp_rlc_resonance)
            self.addItem(self._rlc_resonance_item)
        self._rlc_resonance_item.setVisible(visible)
        if visible and pos is not None:
            self._rlc_resonance_item.setPos(pos[0], pos[1])
        return self._rlc_resonance_item if visible else None

    def toggle_rlc_resonance(self) -> bool:
        """Toggles visibility of the RLC Resonance simulator."""
        if self._rlc_resonance_item is None or not self._rlc_resonance_item.isVisible():
            self.show_rlc_resonance(True)
            return True
        else:
            self.show_rlc_resonance(False)
            return False

    def is_rlc_resonance_visible(self) -> bool:
        """Returns True if the RLC Resonance simulator is currently visible."""
        return self._rlc_resonance_item is not None and self._rlc_resonance_item.isVisible()

    def get_rlc_resonance(self) -> Optional[QGraphicsItem]:
        """Returns the RLC Resonance item if visible."""
        if self._rlc_resonance_item is not None and self._rlc_resonance_item.isVisible():
            return self._rlc_resonance_item
        return None

    def get_rlc_resonance_item(self) -> Optional[QGraphicsItem]:
        """Returns the RLC Resonance item instance regardless of visibility."""
        return self._rlc_resonance_item

    def _on_stamp_rlc_resonance(self, data: dict) -> None:
        """Stamp vector diagram of RLC Resonance onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._rlc_resonance_item.pos().x() + 340.0,
            self._rlc_resonance_item.pos().y() + 190.0,
        ) if self._rlc_resonance_item else (450.0, 250.0)
        items = PhysicsTools.create_rlc_resonance_diagram(
            center=pos,
            preset_code=data.get("preset_code", "AM_RADIO_TUNER_540KHZ_HIGH_Q"),
            circuit_name=data.get("circuit_name", "AM Receiver Tuner (540 kHz)"),
            f0_hz=data.get("f0_hz", 540000.0),
            freq_hz=data.get("freq_hz", 540000.0),
            r_ohms=data.get("r_ohms", 10.0),
            l_henries=data.get("l_henries", 220.0e-6),
            c_farads=data.get("c_farads", 395.7e-12),
            z_ohms=data.get("z_ohms", 10.0),
            i0_amps=data.get("i0_amps", 0.10),
            phi_deg=data.get("phi_deg", 0.0),
            quality_factor_Q=data.get("quality_factor_Q", 74.6),
            bandwidth_hz=data.get("bandwidth_hz", 7238.0),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_stress_strain(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Stress-Strain Curve & Tensile Test simulator."""
        if self._stress_strain_item is None:
            from src.ui.interactive_stress_strain import InteractiveStressStrainItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._stress_strain_item = InteractiveStressStrainItem(pos=spawn_pos)
            self._stress_strain_item.stamp_requested.connect(self._on_stamp_stress_strain)
            self.addItem(self._stress_strain_item)
        self._stress_strain_item.setVisible(visible)
        if visible and pos is not None:
            self._stress_strain_item.setPos(pos[0], pos[1])
        return self._stress_strain_item if visible else None

    def toggle_stress_strain(self) -> bool:
        """Toggles visibility of the Stress-Strain simulator."""
        if self._stress_strain_item is None or not self._stress_strain_item.isVisible():
            self.show_stress_strain(True)
            return True
        else:
            self.show_stress_strain(False)
            return False

    def is_stress_strain_visible(self) -> bool:
        """Returns True if the Stress-Strain simulator is currently visible."""
        return self._stress_strain_item is not None and self._stress_strain_item.isVisible()

    def get_stress_strain(self) -> Optional[QGraphicsItem]:
        """Returns the Stress-Strain item if visible."""
        if self._stress_strain_item is not None and self._stress_strain_item.isVisible():
            return self._stress_strain_item
        return None

    def get_stress_strain_item(self) -> Optional[QGraphicsItem]:
        """Returns the Stress-Strain item instance regardless of visibility."""
        return self._stress_strain_item

    def _on_stamp_stress_strain(self, data: dict) -> None:
        """Stamp vector diagram of Stress-Strain Tensile Test onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._stress_strain_item.pos().x() + 340.0,
            self._stress_strain_item.pos().y() + 190.0,
        ) if self._stress_strain_item else (450.0, 250.0)
        items = PhysicsTools.create_stress_strain_diagram(
            center=pos,
            preset_code=data.get("preset_code", "STRUCTURAL_STEEL_A36_DUCTILE"),
            material_name=data.get("material_name", "Thép Kết Cấu A36 (Structural Carbon Steel)"),
            youngs_modulus_GPa=data.get("youngs_modulus_GPa", 200.0),
            yield_strength_MPa=data.get("yield_strength_MPa", 250.0),
            uts_strength_MPa=data.get("uts_strength_MPa", 400.0),
            current_strain_pct=data.get("current_strain_pct", 0.125),
            current_stress_MPa=data.get("current_stress_MPa", 250.0),
            tensile_force_kN=data.get("tensile_force_kN", 19.63),
            elongation_mm=data.get("elongation_mm", 0.062),
            is_fractured=data.get("is_fractured", False),
            regime=data.get("regime", "Đàn Hồi Tuyến Tính (Hooke)"),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_kepler_orbits(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Kepler Orbits & Astrodynamics simulator."""
        if self._kepler_orbits_item is None:
            from src.ui.interactive_kepler_orbits import InteractiveKeplerOrbitsItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._kepler_orbits_item = InteractiveKeplerOrbitsItem(pos=spawn_pos)
            self._kepler_orbits_item.stamp_requested.connect(self._on_stamp_kepler_orbits)
            self.addItem(self._kepler_orbits_item)
        self._kepler_orbits_item.setVisible(visible)
        if visible and pos is not None:
            self._kepler_orbits_item.setPos(pos[0], pos[1])
        return self._kepler_orbits_item if visible else None

    def toggle_kepler_orbits(self) -> bool:
        """Toggles visibility of the Kepler Orbits simulator."""
        if self._kepler_orbits_item is None or not self._kepler_orbits_item.isVisible():
            self.show_kepler_orbits(True)
            return True
        else:
            self.show_kepler_orbits(False)
            return False

    def is_kepler_orbits_visible(self) -> bool:
        """Returns True if the Kepler Orbits simulator is currently visible."""
        return self._kepler_orbits_item is not None and self._kepler_orbits_item.isVisible()

    def get_kepler_orbits(self) -> Optional[QGraphicsItem]:
        """Returns the Kepler Orbits item if visible."""
        if self._kepler_orbits_item is not None and self._kepler_orbits_item.isVisible():
            return self._kepler_orbits_item
        return None

    def get_kepler_orbits_item(self) -> Optional[QGraphicsItem]:
        """Returns the Kepler Orbits item instance regardless of visibility."""
        return self._kepler_orbits_item

    def _on_stamp_kepler_orbits(self, data: dict) -> None:
        """Stamp vector diagram of Kepler Orbits onto the lesson board."""
        from src.ui.stem_tools import PhysicsTools
        pos = (
            self._kepler_orbits_item.pos().x() + 340.0,
            self._kepler_orbits_item.pos().y() + 190.0,
        ) if self._kepler_orbits_item else (450.0, 250.0)
        items = PhysicsTools.create_kepler_orbits_diagram(
            center=pos,
            preset_code=data.get("preset_code", "HALLEY_COMET_EXTREME_ECCENTRICITY"),
            title_vi=data.get("title_vi", "Sao Chổi Halley (Độ Lệch Tâm Cực Lớn e = 0.967)"),
            body_name=data.get("body_name", "Sao Chổi Halley (1P/Halley)"),
            central_body=data.get("central_body", "Mặt Trời (Sun)"),
            semi_major_axis_km=data.get("semi_major_axis_km", 2.668e9),
            eccentricity=data.get("eccentricity", 0.967),
            semi_minor_axis_km=data.get("semi_minor_axis_km", 6.798e8),
            periapsis_km=data.get("periapsis_km", 8.766e7),
            apoapsis_km=data.get("apoapsis_km", 5.248e9),
            orbital_period_s=data.get("orbital_period_s", 2.376e9),
            periapsis_velocity_km_s=data.get("periapsis_velocity_km_s", 54.55),
            apoapsis_velocity_km_s=data.get("apoapsis_velocity_km_s", 0.91),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

    def show_logic_circuits(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Shows, creates or hides the Interactive Logic Circuits & Digital Computing simulator."""
        if self._logic_circuits_item is None:
            from src.ui.interactive_logic_circuits import InteractiveLogicCircuitsItem
            spawn_pos = pos if pos is not None else (120.0, 45.0)
            self._logic_circuits_item = InteractiveLogicCircuitsItem(pos=spawn_pos)
            self._logic_circuits_item.stamp_requested.connect(self._on_stamp_logic_circuits)
            self.addItem(self._logic_circuits_item)
        self._logic_circuits_item.setVisible(visible)
        if visible and pos is not None:
            self._logic_circuits_item.setPos(pos[0], pos[1])
        return self._logic_circuits_item if visible else None

    def toggle_logic_circuits(self) -> bool:
        """Toggles visibility of the Logic Circuits simulator."""
        if self._logic_circuits_item is None or not self._logic_circuits_item.isVisible():
            self.show_logic_circuits(True)
            return True
        else:
            self.show_logic_circuits(False)
            return False

    def is_logic_circuits_visible(self) -> bool:
        """Returns True if the Logic Circuits simulator is currently visible."""
        return self._logic_circuits_item is not None and self._logic_circuits_item.isVisible()

    def get_logic_circuits(self) -> Optional[QGraphicsItem]:
        """Returns the Logic Circuits item if visible."""
        if self._logic_circuits_item is not None and self._logic_circuits_item.isVisible():
            return self._logic_circuits_item
        return None

    def get_logic_circuits_item(self) -> Optional[QGraphicsItem]:
        """Returns the Logic Circuits item instance regardless of visibility."""
        return self._logic_circuits_item

    def _on_stamp_logic_circuits(self, data: dict) -> None:
        """Stamp vector diagram of Logic Circuits onto the lesson board."""
        from src.ui.stem_tools import InformaticsTools
        pos = (
            self._logic_circuits_item.pos().x() + 340.0,
            self._logic_circuits_item.pos().y() + 190.0,
        ) if self._logic_circuits_item else (450.0, 250.0)
        items = InformaticsTools.create_logic_circuits_diagram(
            center=pos,
            preset_code=data.get("preset_code", "HALF_ADDER_CIRCUIT"),
            title_vi=data.get("title_vi", "Mạch Nửa Cộng (Half Adder - XOR + AND)"),
            circuit_name=data.get("circuit_name", "Binary Half Adder"),
            boolean_formula=data.get("boolean_formula", "Sum = A ⊕ B  |  C_out = A · B"),
            inputs=data.get("inputs", (1, 0)),
            outputs=data.get("outputs", (1, 0)),
            input_names=data.get("input_names", ["A", "B"]),
            output_names=data.get("output_names", ["Sum", "C_out"]),
            description=data.get("description", ""),
        )
        for itm in items:
            self.addItem(itm)
        self.update()

















    def show_dynamic_programming(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Informatics Dynamic Programming (DP) Simulator."""
        if self._dynamic_programming_item is None:
            from src.ui.interactive_dynamic_programming import InteractiveDynamicProgrammingItem
            spawn_pos = pos if pos is not None else (290.0, 110.0)
            self._dynamic_programming_item = InteractiveDynamicProgrammingItem(pos=spawn_pos)
            self.addItem(self._dynamic_programming_item)

        self._dynamic_programming_item.setVisible(visible)
        if visible and pos is not None:
            self._dynamic_programming_item.setPos(pos[0], pos[1])
        self.update()
        return self._dynamic_programming_item if visible else None

    def toggle_dynamic_programming(self) -> bool:
        """Toggle interactive dynamic programming simulator visibility; returns new state."""
        if self._dynamic_programming_item is None or not self._dynamic_programming_item.isVisible():
            self.show_dynamic_programming(True)
            return True
        else:
            self.show_dynamic_programming(False)
            return False

    def is_dynamic_programming_visible(self) -> bool:
        """Return True if interactive dynamic programming simulator is active on board."""
        return self._dynamic_programming_item is not None and self._dynamic_programming_item.isVisible()

    def get_dynamic_programming(self) -> Optional[QGraphicsItem]:
        """Return interactive dynamic programming instance if active and visible."""
        if self._dynamic_programming_item is not None and self._dynamic_programming_item.isVisible():
            return self._dynamic_programming_item
        return None

    def get_dynamic_programming_item(self) -> Optional[QGraphicsItem]:
        """Return interactive dynamic programming instance if created."""
        return self._dynamic_programming_item

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
            getattr(self, "_transformer_attention_item", None),
            getattr(self, "_special_relativity_item", None),
            getattr(self, "_quantum_tunneling_item", None),
            getattr(self, "_coriolis_earth_item", None),
            getattr(self, "_acoustic_sonar_item", None),
            getattr(self, "_cloud_chamber_item", None),
            getattr(self, "_orbital_mechanics_item", None),
            getattr(self, "_heat_transfer_item", None),
            getattr(self, "_quantum_diffraction_item", None),
            getattr(self, "_laval_nozzle_item", None),
            getattr(self, "_ramachandran_plot_item", None),
            getattr(self, "_incline_tool_item", None),
            getattr(self, "_optics_refraction_item", None),
            getattr(self, "_standing_wave_item", None),
            getattr(self, "_vsepr_geometry_item", None),
            getattr(self, "_dna_replication_item", None),
            getattr(self, "_kepler_orbit_item", None),
            getattr(self, "_graph_pathfinding_item", None),
            getattr(self, "_electric_field_item", None),
            getattr(self, "_lorentz_force_item", None),
            getattr(self, "_trophic_web_item", None),
            getattr(self, "_gas_laws_item", None),
            getattr(self, "_titration_item", None),
            getattr(self, "_harmonic_oscillator_item", None),
            getattr(self, "_radioactive_decay_item", None),
            getattr(self, "_chemical_kinetics_item", None),
            getattr(self, "_cryptography_item", None),
            getattr(self, "_gravitational_orbits_item", None),
            getattr(self, "_spacetime_relativity_item", None),
            getattr(self, "_young_interference_item", None),
            getattr(self, "_rlc_circuit_item", None),
            getattr(self, "_thin_lens_item", None),
            getattr(self, "_bragg_diffraction_item", None),
            getattr(self, "_buffer_system_item", None),
            getattr(self, "_stern_gerlach_item", None),
            getattr(self, "_enzyme_kinetics_item", None),
            getattr(self, "_poiseuille_flow_item", None),
            getattr(self, "_cosmic_expansion_item", None),
            getattr(self, "_seismic_waves_item", None),
            getattr(self, "_mass_spectrometer_item", None),
            getattr(self, "_spectrophotometry_item", None),
            getattr(self, "_hall_effect_item", None),
            getattr(self, "_radiation_shielding_item", None),
            getattr(self, "_maxwell_boltzmann_item", None),
            getattr(self, "_molecular_orbital_item", None),
            getattr(self, "_rlc_resonance_item", None),
            getattr(self, "_stress_strain_item", None),
            getattr(self, "_kepler_orbits_item", None),
            getattr(self, "_logic_circuits_item", None),
            getattr(self, "_photoelectric_effect_item", None),
            getattr(self, "_punnett_genetics_item", None),
            getattr(self, "_reaction_kinetics_item", None),
            getattr(self, "_calculus_taylor_item", None),
            getattr(self, "_michelson_interferometer_item", None),
            getattr(self, "_huffman_coding_item", None),
            getattr(self, "_action_potential_item", None),
            getattr(self, "_cyclotron_item", None),
            getattr(self, "_aho_corasick_item", None),
            getattr(self, "_phase_diagram_item", None),
            getattr(self, "_em_wave_item", None),
            getattr(self, "_hardy_weinberg_item", None),
            getattr(self, "_fft_item", None),
            getattr(self, "_fabry_perot_item", None),
            getattr(self, "_hplc_item", None),
            getattr(self, "_photoelectric_item", None),
            getattr(self, "_pcr_item", None),
            getattr(self, "_thin_lens_item", None),
            getattr(self, "_carnot_engine_item", None),
            getattr(self, "_bohr_atom_item", None),
            getattr(self, "_automata_item", None),
            getattr(self, "_sorting_item", None),
            getattr(self, "_punnett_genetics_item", None),
            getattr(self, "_organic_chemistry_item", None),
            getattr(self, "_fluid_venturi_item", None),
            getattr(self, "_binary_tree_item", None),
            getattr(self, "_electrochemistry_item", None),
            getattr(self, "_circular_motion_item", None),
            getattr(self, "_ph_titration_item", None),
            getattr(self, "_electromagnetic_induction_item", None),
            getattr(self, "_dynamic_programming_item", None),
            getattr(self, "_projectile_motion_item", None),
            getattr(self, "_photosynthesis_item", None),
            getattr(self, "_gcms_item", None),
            getattr(self, "_cnn_visualizer_item", None),
            getattr(self, "_airfoil_item", None),
            getattr(self, "_periodic_table_item", None),
            getattr(self, "_cellular_respiration_item", None),
            getattr(self, "_protein_folding_item", None),
            getattr(self, "_mlp_backprop_item", None),
            getattr(self, "_max_flow_item", None),
            getattr(self, "_oxyz_geometry_item", None),
            getattr(self, "_semiconductor_diode_item", None),
            getattr(self, "_nuclear_reactions_item", None),
            getattr(self, "_doppler_effect_item", None),
            getattr(self, "_red_black_tree_item", None),
            getattr(self, "_millikan_oil_drop_item", None),
            getattr(self, "_quantum_well_item", None),
            getattr(self, "_hohmann_transfer_item", None),
            getattr(self, "_photoelectric_effect_item", None),
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

    def get_recognized_items_for_export(self, page_index: int = 0) -> list[dict]:
        """
        Serialize all active OCR text/LaTeX overlays and stamped annotations to
        a list of dicts for .airteach package export (recognized/items.json).
        """
        import uuid
        items_data = []
        for itm in self._overlay_items:
            if itm.scene() is not self:
                continue
            brect = itm.sceneBoundingRect()
            bbox = [
                float(brect.left()),
                float(brect.top()),
                float(brect.right()),
                float(brect.bottom()),
            ]
            data = itm.data(1)
            if isinstance(data, dict):
                items_data.append({
                    "item_id": str(uuid.uuid4())[:8],
                    "page_index": page_index,
                    "text": str(data.get("text", "")),
                    "latex": str(data.get("latex", "")),
                    "confidence": float(data.get("conf", 0.95)),
                    "bounding_box": bbox,
                    "is_formula": bool(data.get("is_formula", False)),
                })
            elif isinstance(itm, QGraphicsTextItem):
                items_data.append({
                    "item_id": str(uuid.uuid4())[:8],
                    "page_index": page_index,
                    "text": itm.toPlainText(),
                    "latex": "",
                    "confidence": 0.95,
                    "bounding_box": bbox,
                    "is_formula": False,
                })
            elif isinstance(itm, QGraphicsPixmapItem):
                latex_val = getattr(itm, "_latex_str", "")
                items_data.append({
                    "item_id": str(uuid.uuid4())[:8],
                    "page_index": page_index,
                    "text": "",
                    "latex": latex_val,
                    "confidence": 0.95,
                    "bounding_box": bbox,
                    "is_formula": bool(latex_val),
                })
        return items_data

    def snapshot_as_pixmap(self) -> QPixmap:
        """Render the entire board to a QPixmap (for export / thumbnail)."""
        bg_col = self.backgroundBrush().color()
        if not bg_col.isValid() or bg_col == Qt.GlobalColor.transparent:
            bg_col = QColor("#f8f9fa")
        pixmap = QPixmap(BOARD_WIDTH, BOARD_HEIGHT)
        pixmap.fill(bg_col)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.render(painter)
        painter.end()
        return pixmap

    def snapshot_as_image(
        self,
        target_width: int = 1920,
        target_height: int = 1080,
        scale: float = 1.0,
        transparent_bg: bool = False,
        include_overlays: bool = False,
    ) -> QImage:
        """Render the lesson board to a high-resolution QImage."""
        from src.ui.canvas_exporter import CanvasExporter
        return CanvasExporter.render_scene_to_image(
            scene=self,
            width=target_width,
            height=target_height,
            scale=scale,
            transparent_bg=transparent_bg,
            include_overlays=include_overlays,
        )

    def export_to_file(
        self,
        output_path: Union[str, Path],
        scale: float = 1.0,
        transparent_bg: bool = False,
        include_overlays: bool = False,
        quality: int = 95,
    ) -> bool:
        """Directly export board canvas to an image file (PNG, JPG) using atomic writing."""
        from pathlib import Path
        from src.ui.canvas_exporter import CanvasExporter
        qimage = self.snapshot_as_image(
            scale=scale,
            transparent_bg=transparent_bg,
            include_overlays=include_overlays,
        )
        return CanvasExporter.save_image_atomic(qimage, Path(output_path), quality=quality)
