"""
AirTeach AI - Interactive Chemistry Molecular 3D VSEPR Geometry Inspector & Orbital Visualizer
=============================================================================================
Provides an interactive 3D chemical molecular geometry simulation:
  - Valence Shell Electron Pair Repulsion (VSEPR) models:
      1. AX2: Tuyến tính (Linear, 180°, lai hóa sp, vd: CO2, BeCl2)
      2. AX3: Tam giác phẳng (Trigonal Planar, 120°, lai hóa sp2, vd: BF3, SO3)
      3. AX2E: Gấp khúc (Bent, ~119°, lai hóa sp2, vd: SO2)
      4. AX4: Tứ diện đều (Tetrahedral, 109.5°, lai hóa sp3, vd: CH4, CCl4)
      5. AX3E: Chóp tam giác (Trigonal Pyramidal, ~107.3°, lai hóa sp3, vd: NH3)
      6. AX2E2: Gấp khúc kiểu nước (Bent, ~104.5°, lai hóa sp3, vd: H2O)
      7. AX5: Lưỡng tháp tam giác (Trigonal Bipyramidal, 90°/120°, lai hóa sp3d, vd: PCl5)
      8. AX6: Bát diện đều (Octahedral, 90°, lai hóa sp3d2, vd: SF6)
  - Full 3D rotation (Pitch & Yaw) via mouse drag or touchless air pinch gesture.
  - Auto-spin animation mode via QTimer.
  - Lone pair electron lobes (Cặp electron tự do E) with repulsion visualization.
  - Depth-sorted 3D sphere shading (Painter's algorithm with QRadialGradient).
  - Stamping permanent vector molecular diagrams onto LessonBoard.
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
    QRadialGradient,
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


@dataclass
class Atom3D:
    x: float
    y: float
    z: float
    symbol: str
    radius: float
    color: QColor
    border_color: QColor
    text_color: QColor


@dataclass
class LonePair3D:
    x: float
    y: float
    z: float
    label: str = "··"


@dataclass
class VseprModel:
    type_code: str          # e.g. "AX4"
    name_vi: str            # e.g. "Tứ diện đều (Tetrahedral)"
    hybridization: str      # e.g. "sp³"
    bond_angle_str: str     # e.g. "109.5°"
    example_formula: str    # e.g. "CH₄"
    example_name: str       # e.g. "Metan"
    central_atom: Atom3D
    ligands: list[Atom3D]
    lone_pairs: list[LonePair3D]
    bonds: list[tuple[int, int]]  # pairs of atom indices (0 = central, 1..N = ligands)


def _build_vsepr_presets(scale: float = 85.0) -> dict[str, VseprModel]:
    """Constructs accurate 3D coordinates for all 8 standard VSEPR configurations."""
    R = scale
    c_white = QColor("#ffffff")
    c_slate_dark = QColor("#1e293b")
    c_gray_border = QColor("#64748b")
    c_dark_text = QColor("#0f172a")

    presets: dict[str, VseprModel] = {}

    # 1. AX2: Linear (CO2)
    c_c = Atom3D(0.0, 0.0, 0.0, "C", 19.0, QColor("#334155"), QColor("#0f172a"), c_white)
    o1 = Atom3D(-R, 0.0, 0.0, "O", 17.0, QColor("#ef4444"), QColor("#991b1b"), c_white)
    o2 = Atom3D(R, 0.0, 0.0, "O", 17.0, QColor("#ef4444"), QColor("#991b1b"), c_white)
    presets["AX2"] = VseprModel(
        type_code="AX₂",
        name_vi="Tuyến tính (Linear)",
        hybridization="sp",
        bond_angle_str="180°",
        example_formula="CO₂",
        example_name="Cacbon dioxit",
        central_atom=c_c,
        ligands=[o1, o2],
        lone_pairs=[],
        bonds=[(0, 1), (0, 2)],
    )

    # 2. AX3: Trigonal Planar (BF3)
    c_b = Atom3D(0.0, 0.0, 0.0, "B", 18.0, QColor("#f59e0b"), QColor("#b45309"), c_white)
    f1 = Atom3D(0.0, -R, 0.0, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    f2 = Atom3D(R * math.cos(math.radians(210)), -R * math.sin(math.radians(210)), 0.0, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    f3 = Atom3D(R * math.cos(math.radians(330)), -R * math.sin(math.radians(330)), 0.0, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    presets["AX3"] = VseprModel(
        type_code="AX₃",
        name_vi="Tam giác phẳng (Trigonal Planar)",
        hybridization="sp²",
        bond_angle_str="120°",
        example_formula="BF₃",
        example_name="Bo triflorua",
        central_atom=c_b,
        ligands=[f1, f2, f3],
        lone_pairs=[],
        bonds=[(0, 1), (0, 2), (0, 3)],
    )

    # 3. AX2E: Bent (SO2)
    c_s = Atom3D(0.0, 0.0, 0.0, "S", 20.0, QColor("#eab308"), QColor("#a16207"), c_white)
    rad_119 = math.radians(119.0 * 0.5)
    so1 = Atom3D(-R * math.sin(rad_119), R * math.cos(rad_119), 0.0, "O", 17.0, QColor("#ef4444"), QColor("#991b1b"), c_white)
    so2 = Atom3D(R * math.sin(rad_119), R * math.cos(rad_119), 0.0, "O", 17.0, QColor("#ef4444"), QColor("#991b1b"), c_white)
    lp_so = LonePair3D(0.0, -R * 0.85, 0.0)
    presets["AX2E"] = VseprModel(
        type_code="AX₂E",
        name_vi="Gấp khúc (Bent)",
        hybridization="sp²",
        bond_angle_str="≈ 119°",
        example_formula="SO₂",
        example_name="Lưu huỳnh dioxit",
        central_atom=c_s,
        ligands=[so1, so2],
        lone_pairs=[lp_so],
        bonds=[(0, 1), (0, 2)],
    )

    # 4. AX4: Tetrahedral (CH4)
    c_ch4 = Atom3D(0.0, 0.0, 0.0, "C", 19.0, QColor("#334155"), QColor("#0f172a"), c_white)
    h1 = Atom3D(0.0, -R, 0.0, "H", 13.0, c_white, c_gray_border, c_dark_text)
    h2 = Atom3D(R * (math.sqrt(8.0) / 3.0), R / 3.0, 0.0, "H", 13.0, c_white, c_gray_border, c_dark_text)
    h3 = Atom3D(-R * (math.sqrt(2.0) / 3.0), R / 3.0, R * (math.sqrt(6.0) / 3.0), "H", 13.0, c_white, c_gray_border, c_dark_text)
    h4 = Atom3D(-R * (math.sqrt(2.0) / 3.0), R / 3.0, -R * (math.sqrt(6.0) / 3.0), "H", 13.0, c_white, c_gray_border, c_dark_text)
    presets["AX4"] = VseprModel(
        type_code="AX₄",
        name_vi="Tứ diện đều (Tetrahedral)",
        hybridization="sp³",
        bond_angle_str="109.5°",
        example_formula="CH₄",
        example_name="Metan",
        central_atom=c_ch4,
        ligands=[h1, h2, h3, h4],
        lone_pairs=[],
        bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
    )

    # 5. AX3E: Trigonal Pyramidal (NH3)
    c_n = Atom3D(0.0, 0.0, 0.0, "N", 19.0, QColor("#3b82f6"), QColor("#1d4ed8"), c_white)
    rad_pyr = math.radians(107.3)
    # 3 Hydrogens at base (y > 0)
    nh1 = Atom3D(R * 0.94 * math.cos(0.0), R * 0.35, R * 0.94 * math.sin(0.0), "H", 13.0, c_white, c_gray_border, c_dark_text)
    nh2 = Atom3D(R * 0.94 * math.cos(math.radians(120)), R * 0.35, R * 0.94 * math.sin(math.radians(120)), "H", 13.0, c_white, c_gray_border, c_dark_text)
    nh3 = Atom3D(R * 0.94 * math.cos(math.radians(240)), R * 0.35, R * 0.94 * math.sin(math.radians(240)), "H", 13.0, c_white, c_gray_border, c_dark_text)
    lp_n = LonePair3D(0.0, -R * 0.85, 0.0)
    presets["AX3E"] = VseprModel(
        type_code="AX₃E",
        name_vi="Chóp tam giác (Trigonal Pyramidal)",
        hybridization="sp³",
        bond_angle_str="≈ 107.3°",
        example_formula="NH₃",
        example_name="Amoniac",
        central_atom=c_n,
        ligands=[nh1, nh2, nh3],
        lone_pairs=[lp_n],
        bonds=[(0, 1), (0, 2), (0, 3)],
    )

    # 6. AX2E2: Bent (H2O)
    c_o = Atom3D(0.0, 0.0, 0.0, "O", 19.0, QColor("#ef4444"), QColor("#991b1b"), c_white)
    rad_104 = math.radians(104.5 * 0.5)
    w_h1 = Atom3D(-R * math.sin(rad_104), R * math.cos(rad_104), 0.0, "H", 13.0, c_white, c_gray_border, c_dark_text)
    w_h2 = Atom3D(R * math.sin(rad_104), R * math.cos(rad_104), 0.0, "H", 13.0, c_white, c_gray_border, c_dark_text)
    lp_w1 = LonePair3D(-R * 0.7 * math.sin(math.radians(55)), -R * 0.7 * math.cos(math.radians(55)), R * 0.4)
    lp_w2 = LonePair3D(R * 0.7 * math.sin(math.radians(55)), -R * 0.7 * math.cos(math.radians(55)), -R * 0.4)
    presets["AX2E2"] = VseprModel(
        type_code="AX₂E₂",
        name_vi="Gấp khúc (Bent)",
        hybridization="sp³",
        bond_angle_str="≈ 104.5°",
        example_formula="H₂O",
        example_name="Nước",
        central_atom=c_o,
        ligands=[w_h1, w_h2],
        lone_pairs=[lp_w1, lp_w2],
        bonds=[(0, 1), (0, 2)],
    )

    # 7. AX5: Trigonal Bipyramidal (PCl5)
    c_p = Atom3D(0.0, 0.0, 0.0, "P", 20.0, QColor("#f97316"), QColor("#c2410c"), c_white)
    # Axial (up & down)
    ax1 = Atom3D(0.0, -R, 0.0, "Cl", 17.0, QColor("#10b981"), QColor("#047857"), c_white)
    ax2 = Atom3D(0.0, R, 0.0, "Cl", 17.0, QColor("#10b981"), QColor("#047857"), c_white)
    # Equatorial (X-Z plane)
    eq1 = Atom3D(R, 0.0, 0.0, "Cl", 17.0, QColor("#10b981"), QColor("#047857"), c_white)
    eq2 = Atom3D(R * math.cos(math.radians(120)), 0.0, R * math.sin(math.radians(120)), "Cl", 17.0, QColor("#10b981"), QColor("#047857"), c_white)
    eq3 = Atom3D(R * math.cos(math.radians(240)), 0.0, R * math.sin(math.radians(240)), "Cl", 17.0, QColor("#10b981"), QColor("#047857"), c_white)
    presets["AX5"] = VseprModel(
        type_code="AX₅",
        name_vi="Lưỡng tháp tam giác (Trigonal Bipyramidal)",
        hybridization="sp³d",
        bond_angle_str="90° & 120°",
        example_formula="PCl₅",
        example_name="Photpho pentaclorua",
        central_atom=c_p,
        ligands=[ax1, ax2, eq1, eq2, eq3],
        lone_pairs=[],
        bonds=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    )

    # 8. AX6: Octahedral (SF6)
    c_sf6 = Atom3D(0.0, 0.0, 0.0, "S", 20.0, QColor("#eab308"), QColor("#a16207"), c_white)
    oct1 = Atom3D(0.0, -R, 0.0, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    oct2 = Atom3D(0.0, R, 0.0, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    oct3 = Atom3D(-R, 0.0, 0.0, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    oct4 = Atom3D(R, 0.0, 0.0, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    oct5 = Atom3D(0.0, 0.0, -R, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    oct6 = Atom3D(0.0, 0.0, R, "F", 16.0, QColor("#10b981"), QColor("#047857"), c_white)
    presets["AX6"] = VseprModel(
        type_code="AX₆",
        name_vi="Bát diện đều (Octahedral)",
        hybridization="sp³d²",
        bond_angle_str="90°",
        example_formula="SF₆",
        example_name="Lưu huỳnh hexaflorua",
        central_atom=c_sf6,
        ligands=[oct1, oct2, oct3, oct4, oct5, oct6],
        lone_pairs=[],
        bonds=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
    )

    return presets


class InteractiveVseprGeometryItem(QGraphicsObject):
    """
    Interactive 3D VSEPR molecular geometry simulator with real-time 3D rotation,
    hybridization orbital display, lone pair lobes, and LessonBoard vector stamping.
    """

    model_changed = Signal(str)           # (type_code, e.g. "AX4")
    rotation_changed = Signal(float, float) # (pitch, yaw)

    def __init__(
        self,
        pos: tuple[float, float] = (500.0, 330.0),
        width: float = 640.0,
        height: float = 420.0,
        model_key: str = "AX4",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        # 3D Rotation Angles (in degrees)
        self._pitch = 18.0
        self._yaw = -25.0
        self._is_spinning = False

        # Auto-spin timer
        self._spin_timer = QTimer()
        self._spin_timer.setInterval(30)  # ~33 FPS
        self._spin_timer.timeout.connect(self._on_spin_tick)

        # Load models
        self._presets = _build_vsepr_presets(scale=85.0)
        self._current_key = model_key if model_key in self._presets else "AX4"

        # Center point for 3D projection (local item coordinates)
        self._proj_cx = self._width * 0.44
        self._proj_cy = self._height * 0.54

        # Dragging state
        self._active_drag: Optional[str] = None  # "ROTATION", "BODY"
        self._last_mouse_pos = QPointF()
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Control button bounding rects (local coords)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_spin = QRectF(self._width - 245.0, 10.0, 90.0, 26.0)

        # 8 Model selector buttons across top row
        self._keys_order = ["AX2", "AX3", "AX2E", "AX4", "AX3E", "AX2E2", "AX5", "AX6"]
        self._rect_btn_models: list[QRectF] = []
        btn_w = 46.0
        start_x = 18.0
        for i in range(len(self._keys_order)):
            self._rect_btn_models.append(QRectF(start_x + i * (btn_w + 4.0), 10.0, btn_w, 26.0))

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin for 3D atoms."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── 3D Projection Mathematics ───────────────────────────────────────────

    def _rotate_3d(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """
        Applies Pitch (X-axis) and Yaw (Y-axis) Euler rotation:
          1. Yaw around Y-axis by angle theta_y
          2. Pitch around X-axis by angle theta_x
        """
        rad_y = math.radians(self._yaw)
        rad_x = math.radians(self._pitch)

        # 1. Yaw rotation (around Y)
        x1 = x * math.cos(rad_y) + z * math.sin(rad_y)
        y1 = y
        z1 = -x * math.sin(rad_y) + z * math.cos(rad_y)

        # 2. Pitch rotation (around X)
        x2 = x1
        y2 = y1 * math.cos(rad_x) - z1 * math.sin(rad_x)
        z2 = y1 * math.sin(rad_x) + z1 * math.cos(rad_x)

        return x2, y2, z2

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """
        Projects 3D point (x, y, z) into 2D item coordinates with pseudo-perspective:
        Returns (screen_x, screen_y, depth_z).
        """
        rx, ry, rz = self._rotate_3d(x, y, z)
        # Perspective distance
        d_cam = 450.0
        scale_p = d_cam / (d_cam + rz)
        sx = self._proj_cx + rx * scale_p
        sy = self._proj_cy + ry * scale_p
        return sx, sy, rz

    # ─── Model Getters & Mutators ────────────────────────────────────────────

    def current_model(self) -> VseprModel:
        """Returns the active VSEPR molecular model configuration."""
        return self._presets[self._current_key]

    def current_key(self) -> str:
        """Returns active model key string (e.g. 'AX4')."""
        return self._current_key

    def set_model(self, key: str) -> None:
        """Selects a VSEPR configuration by key."""
        if key in self._presets:
            self._current_key = key
            self.model_changed.emit(key)
            self.update()

    def set_rotation(self, pitch: float, yaw: float) -> None:
        """Sets 3D rotation angles."""
        self._pitch = ((pitch + 180.0) % 360.0) - 180.0
        self._yaw = ((yaw + 180.0) % 360.0) - 180.0
        self.rotation_changed.emit(self._pitch, self._yaw)
        self.update()

    def toggle_spin(self) -> bool:
        """Toggles smooth 3D auto-spin animation."""
        self._is_spinning = not self._is_spinning
        if self._is_spinning:
            self._spin_timer.start()
        else:
            self._spin_timer.stop()
        self.update()
        return self._is_spinning

    def is_spinning(self) -> bool:
        """Returns True if 3D auto-spin is active."""
        return self._is_spinning

    def _on_spin_tick(self) -> None:
        """Updates yaw angle during auto-spin."""
        self.set_rotation(self._pitch, self._yaw + 1.2)

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders 3D molecular geometry with depth sorting and CPK shading."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._height

        # 1. Container Frosted Glass Card
        card_rect = QRectF(0.0, 0.0, w, h)
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Render 3D Molecular Model (Depth Sorted)
        self._draw_molecule_3d(painter)

        # 3. VSEPR Theory HUD Card (Bottom Right)
        self._draw_vsepr_hud(painter)

        # 4. Control Buttons (Model selectors, spin, stamp, close)
        self._draw_buttons(painter)

        painter.restore()

    def _draw_molecule_3d(self, painter: QPainter) -> None:
        """Renders 3D bonds, lone pairs, and atoms using Painter's Depth Sorting Algorithm."""
        model = self.current_model()
        all_atoms = [model.central_atom] + model.ligands

        # 1. Project all atoms to screen space
        proj_atoms: list[tuple[float, float, float, Atom3D]] = []
        for atm in all_atoms:
            sx, sy, sz = self.project_point(atm.x, atm.y, atm.z)
            proj_atoms.append((sx, sy, sz, atm))

        # 2. Project all lone pairs
        proj_lps: list[tuple[float, float, float, LonePair3D]] = []
        for lp in model.lone_pairs:
            lx, ly, lz = self.project_point(lp.x, lp.y, lp.z)
            proj_lps.append((lx, ly, lz, lp))

        # 3. Collect all render primitives with average Z-depth for sorting
        render_queue: list[tuple[float, str, object]] = []

        # Add bonds
        for idx1, idx2 in model.bonds:
            if idx1 < len(proj_atoms) and idx2 < len(proj_atoms):
                p1 = proj_atoms[idx1]
                p2 = proj_atoms[idx2]
                avg_z = (p1[2] + p2[2]) * 0.5
                render_queue.append((avg_z, "BOND", (p1, p2)))

        # Add lone pair lobes
        for lp_data in proj_lps:
            # Connect central atom to lone pair lobe
            c_data = proj_atoms[0]
            avg_z = (c_data[2] + lp_data[2]) * 0.5 - 5.0
            render_queue.append((avg_z, "LONE_PAIR", (c_data, lp_data)))

        # Add atoms
        for atm_data in proj_atoms:
            render_queue.append((atm_data[2], "ATOM", atm_data))

        # 4. Sort ascending by depth Z (draw furthest negative/deepest Z first)
        render_queue.sort(key=lambda item: item[0], reverse=True)

        # 5. Render queue in depth order
        for _, prim_type, prim_data in render_queue:
            if prim_type == "BOND":
                p1, p2 = prim_data
                self._draw_bond(painter, p1, p2)
            elif prim_type == "LONE_PAIR":
                c_data, lp_data = prim_data
                self._draw_lone_pair_lobe(painter, c_data, lp_data)
            elif prim_type == "ATOM":
                self._draw_atom(painter, prim_data)

    def _draw_bond(
        self,
        painter: QPainter,
        p1: tuple[float, float, float, Atom3D],
        p2: tuple[float, float, float, Atom3D],
    ) -> None:
        """Draws a 3D cylindrical chemical bond connecting two atoms."""
        pt1 = QPointF(p1[0], p1[1])
        pt2 = QPointF(p2[0], p2[1])

        # Outer shaded bond
        pen_bond_outer = QPen(QColor("#475569"), 7.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bond_outer)
        painter.drawLine(pt1, pt2)

        # Inner highlight core
        pen_bond_core = QPen(QColor("#94a3b8"), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bond_core)
        painter.drawLine(pt1, pt2)

    def _draw_lone_pair_lobe(
        self,
        painter: QPainter,
        c_data: tuple[float, float, float, Atom3D],
        lp_data: tuple[float, float, float, LonePair3D],
    ) -> None:
        """Draws a translucent orbital electron pair lobe with '··' dots."""
        cx, cy, _, _ = c_data
        lx, ly, _, lp = lp_data

        # Draw connecting translucent teardrop lobe
        lobe_path = QPainterPath()
        lobe_path.moveTo(cx, cy)
        dx = lx - cx
        dy = ly - cy
        length = math.hypot(dx, dy)
        if length > 1e-2:
            ux, uy = dx / length, dy / length
            px, py = -uy * 14.0, ux * 14.0
            lobe_path.cubicTo(
                cx + dx * 0.4 + px, cy + dy * 0.4 + py,
                lx + px * 0.8, ly + py * 0.8,
                lx, ly,
            )
            lobe_path.cubicTo(
                lx - px * 0.8, ly - py * 0.8,
                cx + dx * 0.4 - px, cy + dy * 0.4 - py,
                cx, cy,
            )

        painter.setPen(QPen(QColor(168, 85, 247, 140), 1.4, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(168, 85, 247, 45)))
        painter.drawPath(lobe_path)

        # Draw electron dots '··'
        f_dots = QFont("Segoe UI", 12)
        f_dots.setBold(True)
        painter.setFont(f_dots)
        painter.setPen(QPen(QColor("#e9d5ff")))
        painter.drawText(QRectF(lx - 12.0, ly - 12.0, 24.0, 24.0), Qt.AlignmentFlag.AlignCenter, lp.label)

    def _draw_atom(
        self,
        painter: QPainter,
        atm_data: tuple[float, float, float, Atom3D],
    ) -> None:
        """Draws a 3D sphere with radial gradient specular highlight."""
        sx, sy, sz, atm = atm_data
        # Depth perspective scaling for radius
        d_cam = 450.0
        scale_r = d_cam / (d_cam + sz)
        r = atm.radius * scale_r

        # Radial gradient for 3D sphere lighting (specular top-left)
        grad = QRadialGradient(sx - r * 0.35, sy - r * 0.35, r * 1.35)
        # Highlight color
        grad.setColorAt(0.0, QColor(255, 255, 255, 220))
        grad.setColorAt(0.25, atm.color)
        grad.setColorAt(1.0, atm.border_color)

        painter.setPen(QPen(atm.border_color, 1.4))
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(sx, sy), r, r)

        # Atomic Symbol text
        f_sym = QFont("Segoe UI", max(7, int(r * 0.85)))
        f_sym.setBold(True)
        painter.setFont(f_sym)
        painter.setPen(QPen(atm.text_color))
        painter.drawText(
            QRectF(sx - r, sy - r, r * 2.0, r * 2.0),
            Qt.AlignmentFlag.AlignCenter,
            atm.symbol,
        )

    def _draw_vsepr_hud(self, painter: QPainter) -> None:
        """Renders the VSEPR Theory information card."""
        hud_w = 230.0
        hud_h = 135.0
        hud_x = self._width - hud_w - 18.0
        hud_y = self._height - hud_h - 18.0
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        model = self.current_model()

        # Title: Type code + Name
        f_title = QFont("Segoe UI", 9)
        f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 8.0, hud_w - 20.0, 18.0), f"{model.type_code} — {model.name_vi}")

        f_vals = QFont("Segoe UI", 8)
        painter.setFont(f_vals)
        painter.setPen(QPen(QColor("#cbd5e1")))

        painter.drawText(QRectF(hud_x + 10.0, hud_y + 28.0, hud_w - 20.0, 16.0), f"Lai hóa orbital: {model.hybridization}")
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 46.0, hud_w - 20.0, 16.0), f"Góc liên kết: {model.bond_angle_str}")

        num_bonds = len(model.ligands)
        num_lps = len(model.lone_pairs)
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 64.0, hud_w - 20.0, 16.0), f"Số liên kết: {num_bonds} | Cặp e tự do: {num_lps}")

        # Example molecule badge
        badge_rect = QRectF(hud_x + 10.0, hud_y + 88.0, hud_w - 20.0, 36.0)
        painter.setPen(QPen(QColor("#10b981"), 1.2))
        painter.setBrush(QBrush(QColor(16, 185, 129, 35)))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        f_badge = QFont("Segoe UI", 8)
        f_badge.setBold(True)
        painter.setFont(f_badge)
        painter.setPen(QPen(QColor("#a7f3d0")))
        painter.drawText(
            badge_rect,
            Qt.AlignmentFlag.AlignCenter,
            f"Ví dụ: {model.example_formula} ({model.example_name})",
        )

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders VSEPR model selector buttons and action controls."""
        f_btn = QFont("Segoe UI", 8)
        f_btn.setBold(True)
        painter.setFont(f_btn)

        # Close button [✕]
        painter.setPen(QPen(QColor("#ef4444"), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68, 40)))
        painter.drawRoundedRect(self._rect_btn_close, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ef4444")))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # Stamp button [📌 Đóng dấu]
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor(56, 189, 248, 40)))
        painter.drawRoundedRect(self._rect_btn_stamp, 6.0, 6.0)
        painter.setPen(QPen(QColor("#e0f2fe")))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # Spin button [🔄 Tự xoay]
        spin_txt = "⏸ Dừng" if self._is_spinning else "🔄 Tự xoay"
        spin_col = QColor("#f59e0b") if self._is_spinning else QColor("#818cf8")
        painter.setPen(QPen(spin_col, 1.2))
        painter.setBrush(QBrush(QColor(spin_col.red(), spin_col.green(), spin_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_spin, 6.0, 6.0)
        painter.setPen(QPen(spin_col))
        painter.drawText(self._rect_btn_spin, Qt.AlignmentFlag.AlignCenter, spin_txt)

        # 8 Model selector buttons across top row
        for i, key in enumerate(self._keys_order):
            rect = self._rect_btn_models[i]
            is_active = (key == self._current_key)
            if is_active:
                painter.setPen(QPen(QColor("#38bdf8"), 1.4))
                painter.setBrush(QBrush(QColor(56, 189, 248, 120)))
                painter.setPen(QPen(QColor("#ffffff")))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
                painter.setPen(QPen(QColor("#94a3b8")))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            model_obj = self._presets[key]
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, model_obj.type_code)

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles 3D rotation dragging or button clicks."""
        pos = event.pos()

        # Check button clicks
        if self._rect_btn_close.contains(pos):
            self.close()
            event.accept()
            return

        if self._rect_btn_spin.contains(pos):
            self.toggle_spin()
            event.accept()
            return

        # Check model buttons
        for i, rect in enumerate(self._rect_btn_models):
            if rect.contains(pos):
                self.set_model(self._keys_order[i])
                event.accept()
                return

        # Check stamp button
        if self._rect_btn_stamp.contains(pos):
            scene = self.scene()
            if scene is not None:
                board = getattr(scene, "board", None)
                if board is None and hasattr(scene, "views") and scene.views():
                    view = scene.views()[0]
                    board = view if hasattr(view, "addItem") else getattr(view, "board", None)
                if board is not None:
                    self.stamp_to_canvas(board)
            event.accept()
            return

        # Center area -> 3D Rotation drag
        self._active_drag = "ROTATION"
        self._last_mouse_pos = event.pos()
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Rotates the 3D molecule according to mouse delta."""
        if self._active_drag == "ROTATION":
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            # Yaw rotates with X movement, Pitch rotates with Y movement
            new_yaw = self._yaw + delta.x() * 0.75
            new_pitch = self._pitch - delta.y() * 0.75
            self.set_rotation(new_pitch, new_yaw)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Releases rotation drag state."""
        self._active_drag = None
        super().mouseReleaseEvent(event)

    # ─── Touchless Air Gesture Integration ───────────────────────────────────

    def set_air_hover_pos(self, scene_pos: Optional[QPointF]) -> None:
        """Sets touchless air pointer hover position."""
        if scene_pos is not None:
            self._air_hover_pos = self.mapFromScene(scene_pos)
        else:
            self._air_hover_pos = None
        self.update()

    def handle_air_drag(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch drag rotating the 3D molecule."""
        pos = self.mapFromScene(scene_pos)

        if self._active_drag is None:
            self._active_drag = "ROTATION"
            self._last_mouse_pos = pos
            return True

        delta = pos - self._last_mouse_pos
        self._last_mouse_pos = pos
        new_yaw = self._yaw + delta.x() * 0.85
        new_pitch = self._pitch - delta.y() * 0.85
        self.set_rotation(new_pitch, new_yaw)
        return True

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Touchless air tap targeting model buttons and controls."""
        pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(pos):
            self.close()
            return True

        if self._rect_btn_spin.contains(pos):
            self.toggle_spin()
            return True

        for i, rect in enumerate(self._rect_btn_models):
            if rect.contains(pos):
                self.set_model(self._keys_order[i])
                return True

        if self._rect_btn_stamp.contains(pos):
            target_board = board
            if target_board is None and self.scene() is not None:
                scene = self.scene()
                target_board = getattr(scene, "board", None)
            if target_board is not None:
                self.stamp_to_canvas(target_board)
                return True

        return False

    def close(self) -> None:
        """Stops spin timer and removes this interactive item from its scene."""
        self._spin_timer.stop()
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

    # ─── Stamping onto LessonBoard ───────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """
        Permanently stamps crisp vector 3D molecular geometry onto LessonBoard
        via ChemistryTools.create_molecular_geometry.
        """
        from src.ui.stem_tools import ChemistryTools

        model = self.current_model()
        origin = self.pos()
        center_scene = (origin.x() + self._proj_cx, origin.y() + self._proj_cy)

        # Map type to ChemistryTools geometry keyword
        geom_map = {
            "AX2": "linear",
            "AX3": "trigonal planar",
            "AX2E": "bent",
            "AX4": "tetrahedral",
            "AX3E": "trigonal pyramidal",
            "AX2E2": "bent",
            "AX5": "trigonal planar",
            "AX6": "octahedral",
        }
        geom_key = geom_map.get(self._current_key, "tetrahedral")
        title_str = f"Mô hình VSEPR: {model.type_code} — {model.example_formula} ({model.name_vi}, lai hóa {model.hybridization})"

        stamped = ChemistryTools.create_molecular_geometry(
            center=center_scene,
            geometry_type=geom_key,
            scale=80.0,
            title=title_str,
            show_bonds=True,
            show_labels=True,
        )

        for itm in stamped:
            board.addItem(itm)

        return stamped
