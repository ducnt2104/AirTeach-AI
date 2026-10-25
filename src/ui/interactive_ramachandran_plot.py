"""
AirTeach AI - Interactive Structural Biochemistry: Ramachandran Plot & Protein Folding
======================================================================================
Authentic structural biochemistry, peptide backbone torsion angles & steric clash physics:
- Ramachandran Dihedral Angles (G. N. Ramachandran, C. Ramakrishnan & V. Sasisekharan, 1963):
    - ϕ (phi): Rotation around the N - C_alpha bond.
        Defined by atoms C'_{i-1} - N_i - C_alpha_i - C'_i.
    - ψ (psi): Rotation around the C_alpha - C' bond.
        Defined by atoms N_i - C_alpha_i - C'_i - N_{i+1}.
    - ω (omega): Peptide bond C' - N, planar resonance (trans: ω ≈ 180°, cis: ω ≈ 0°).
- Hard-Sphere Steric Clashing (van der Waals Contact Distance):
    Atoms cannot approach closer than their contact distance:
      d(A, B) >= r_vdw(A) + r_vdw(B) - Δ
      r_vdw: H ≈ 1.20 Å, C ≈ 1.70 Å, N ≈ 1.55 Å, O ≈ 1.52 Å, C_beta ≈ 1.80 Å.
    Any dihedral pair (ϕ, ψ) that produces atomic overlap d < d_limit is energetically forbidden.
- Secondary Structure Basins on (ϕ, ψ) Map:
    1. Right-handed alpha-helix (α_R): ϕ ≈ -57°, ψ ≈ -47° (i -> i+4 H-bonds).
    2. Antiparallel beta-sheet (β_A): ϕ ≈ -139°, ψ ≈ +135° (interstrand H-bonds).
    3. Parallel beta-sheet (β_P): ϕ ≈ -119°, ψ ≈ +113°.
    4. Collagen triple-helix / Polyproline II (P_II): ϕ ≈ -75°, ψ ≈ +145°.
    5. Left-handed alpha-helix (α_L): ϕ ≈ +57°, ψ ≈ +47° (predominantly Glycine).
- Residue Specificity:
    - General (Ala, Val, Leu, etc.): Standard core, allowed, and generously allowed zones.
    - Glycine (Gly, G): No C_beta sidechain, fourfold symmetric allowed region (~61% of map).
    - Proline (Pro, P): Cyclic pyrrolidine ring locks ϕ ≈ -65° ± 15°.
- 5 Curriculum Presets:
    1. ALPHA_HELIX: Classic α_R helix (ϕ = -57°, ψ = -47°), optimum H-bonding.
    2. ANTIPARALLEL_BETA: Extended β-strand (ϕ = -139°, ψ = +135°).
    3. COLLAGEN_PII: Polyproline II helix (ϕ = -75°, ψ = +145°), collagen motif.
    4. GLYCINE_FLEXIBLE: Non-chiral Glycine demonstrating four-quadrant conformational freedom.
    5. STERIC_CLASH_ECLIPSED: Fully eclipsed disallowed conformation (ϕ = 0°, ψ = 0°), severe atomic overlap.
- Touchless webcam air gestures (air-tap, slider dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
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
class RamachandranPreset:
    """Preset configuration for educational protein conformation experiments."""
    key: str
    name: str
    desc: str
    phi_deg: float
    psi_deg: float
    residue_type: str        # 'general', 'glycine', 'proline'
    h_bond_expected: bool
    is_favorable: bool


PRESETS: List[RamachandranPreset] = [
    LavalPreset if False else RamachandranPreset(
        key="alpha_helix",
        name="Xoắn alpha (α-Helix)",
        desc="Cấu trúc xoắn alpha thuận (ϕ = -57°, ψ = -47°), liên kết hydro i -> i+4 tối ưu, năng lượng cực tiểu",
        phi_deg=-57.0,
        psi_deg=-47.0,
        residue_type="general",
        h_bond_expected=True,
        is_favorable=True,
    ),
    RamachandranPreset(
        key="beta_sheet",
        name="Phiến gấp nếp beta (β-Sheet)",
        desc="Mạch polypeptide duỗi thẳng đối song song (ϕ = -139°, ψ = +135°), tạo phiến gấp nếp bền vững",
        phi_deg=-139.0,
        psi_deg=135.0,
        residue_type="general",
        h_bond_expected=True,
        is_favorable=True,
    ),
    RamachandranPreset(
        key="collagen_pii",
        name="Xoắn Collagen (Polyproline II)",
        desc="Xoắn ba sợi Collagen / Polyproline II (ϕ = -75°, ψ = +145°), cấu trúc giàu Proline và Glycine",
        phi_deg=-75.0,
        psi_deg=145.0,
        residue_type="general",
        h_bond_expected=False,
        is_favorable=True,
    ),
    RamachandranPreset(
        key="glycine_flexible",
        name="Glycine linh hoạt (Gly, G)",
        desc="Glycine không có nhóm thế Cβ (chỉ có H), không bị cản trở không gian, mở rộng cả 4 góc phần tư",
        phi_deg=60.0,
        psi_deg=40.0,
        residue_type="glycine",
        h_bond_expected=False,
        is_favorable=True,
    ),
    RamachandranPreset(
        key="steric_clash",
        name="Va chạm không gian (Steric Clash)",
        desc="Cấu hình che khuất (ϕ = 0°, ψ = 0°), oxy cacbonyl và hydro amit va chạm kịch liệt, vùng bị cấm",
        phi_deg=0.0,
        psi_deg=0.0,
        residue_type="general",
        h_bond_expected=False,
        is_favorable=False,
    ),
]


class InteractiveRamachandranPlotItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Structural Biochemistry, Ramachandran Plot,
    peptide backbone dihedral torsion angles (phi, psi), and steric clashing simulation.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 920.0
    HEIGHT: float = 600.0

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 0
        self._is_paused: bool = False
        self._sim_time: float = 0.0

        # Dihedral Torsion Angles (degrees)
        p = PRESETS[0]
        self._phi_deg: float = p.phi_deg
        self._psi_deg: float = p.psi_deg
        self._omega_deg: float = 180.0     # Trans-peptide bond
        self._residue_type: str = p.residue_type

        # UI interaction tracking
        self._dragging_probe: bool = False
        self._dragging_slider: Optional[str] = None

    @property
    def current_preset(self) -> RamachandranPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active Ramachandran conformation preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._phi_deg = p.phi_deg
            self._psi_deg = p.psi_deg
            self._residue_type = p.residue_type
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_phi(self, val_deg: float) -> None:
        """Update phi dihedral angle (degrees in [-180, +180])."""
        if self._residue_type == "proline":
            self._phi_deg = max(-80.0, min(val_deg, -50.0))
        else:
            self._phi_deg = max(-180.0, min(val_deg, 180.0))
        self.property_changed.emit("phi_deg", self._phi_deg)
        self.update()

    def set_psi(self, val_deg: float) -> None:
        """Update psi dihedral angle (degrees in [-180, +180])."""
        self._psi_deg = max(-180.0, min(val_deg, 180.0))
        self.property_changed.emit("psi_deg", self._psi_deg)
        self.update()

    def set_residue_type(self, res_type: str) -> None:
        """Set amino acid residue type: 'general', 'glycine', 'proline'."""
        if res_type in ("general", "glycine", "proline"):
            self._residue_type = res_type
            if res_type == "proline":
                # Constrain phi to proline ring window
                self._phi_deg = max(-80.0, min(self._phi_deg, -50.0))
            self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def step_simulation(self, dt: float) -> None:
        """Advance time for dynamic atomic thermal vibrations in dipeptide model."""
        if self._is_paused or dt <= 0.0:
            return
        self._sim_time += dt
        self.update()

    # -------------------------------------------------------------------------
    # Steric Clashing & Conformation Evaluation
    # -------------------------------------------------------------------------

    def evaluate_conformation(self) -> dict[str, Any]:
        """
        Evaluate Ramachandran region, steric collisions, and secondary structure matching:
        - Favorable (Core): Low energy minima (Alpha-helix, Beta-sheet, Collagen)
        - Allowed: Periphery of favorable basins
        - Generously Allowed: Outer boundary with slight van der Waals compression
        - Disallowed (Steric Clash): Severe atomic overlap
        """
        phi = self._phi_deg
        psi = self._psi_deg

        # Calculate distances to standard conformational centers
        d_alpha_r = math.sqrt((phi - (-57.0))**2 + (psi - (-47.0))**2)
        d_beta = math.sqrt((phi - (-135.0))**2 + (psi - 135.0)**2)
        d_pii = math.sqrt((phi - (-75.0))**2 + (psi - 145.0)**2)
        d_alpha_l = math.sqrt((phi - 57.0)**2 + (psi - 47.0)**2)

        # Steric clash calculation: overlap between O_i and H_{i+1} or C_{i-1} and N_{i+1}
        # In eclipsed conformation (phi ≈ 0, psi ≈ 0), severe clash occurs
        d_eclipsed = math.sqrt(phi**2 + psi**2)
        d_clash_n_c = math.sqrt((phi - 0.0)**2 + (psi - 180.0)**2)

        is_clash = False
        clash_severity = 0.0
        if d_eclipsed < 45.0:
            is_clash = True
            clash_severity = 1.0 - (d_eclipsed / 45.0)
        elif d_clash_n_c < 30.0:
            is_clash = True
            clash_severity = 1.0 - (d_clash_n_c / 30.0)

        # Region classification
        if self._residue_type == "glycine":
            # Glycine has 4-quadrant symmetric allowed basins
            if d_alpha_r < 35.0 or d_alpha_l < 35.0 or d_beta < 45.0 or math.sqrt((phi - 135.0)**2 + (psi - (-135.0))**2) < 45.0:
                region = "core"
                structure_name = "Vùng lõi Glycine (Cả 2 bán cầu)"
            elif not is_clash:
                region = "allowed"
                structure_name = "Vùng cho phép Glycine"
            else:
                region = "disallowed"
                structure_name = "Va chạm không gian (Disallowed)"

        elif self._residue_type == "proline":
            if not (-85.0 <= phi <= -45.0):
                region = "disallowed"
                structure_name = "Bị khóa bởi vòng Pyrrolidine (Proline)"
                is_clash = True
                clash_severity = 0.95
            elif d_alpha_r < 30.0:
                region = "core"
                structure_name = "Xoắn alpha Proline"
            elif d_pii < 35.0:
                region = "core"
                structure_name = "Xoắn Polyproline II (PII)"
            else:
                region = "allowed"
                structure_name = "Vùng cho phép Proline"

        else:
            # General amino acid (Ala, Val, Leu, etc.)
            if d_alpha_r < 25.0:
                region = "core"
                structure_name = "Xoắn alpha thuận (α_R)"
            elif d_beta < 30.0:
                region = "core"
                structure_name = "Phiến gấp nếp beta (β-sheet)"
            elif d_pii < 25.0:
                region = "core"
                structure_name = "Xoắn Collagen (PII)"
            elif d_alpha_l < 18.0:
                region = "generously_allowed"
                structure_name = "Xoắn alpha nghịch (α_L)"
            elif (d_alpha_r < 45.0 or d_beta < 55.0 or d_pii < 45.0) and not is_clash:
                region = "allowed"
                structure_name = "Vùng cho phép mở rộng"
            else:
                region = "disallowed"
                structure_name = "Vùng cấm (Steric Clash)"
                is_clash = True
                clash_severity = max(clash_severity, 0.70)

        # Hydrogen bond formation check (optimum for alpha-helix)
        h_bond = (d_alpha_r < 20.0 or d_beta < 25.0)

        return {
            "region": region,
            "structure_name": structure_name,
            "is_clash": is_clash,
            "clash_severity": clash_severity,
            "h_bond": h_bond,
            "d_alpha_r": d_alpha_r,
            "d_beta": d_beta,
        }

    # -------------------------------------------------------------------------
    # Layout Geometry
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _plot_rect(self) -> QRectF:
        """Bounding rect of the 2D Ramachandran (phi, psi) contour map."""
        return QRectF(25.0, 115.0, 440.0, 360.0)

    def _peptide_rect(self) -> QRectF:
        """Bounding rect of the 3D-projected dipeptide atomic model."""
        return QRectF(480.0, 115.0, 415.0, 360.0)

    def _controls_rect(self) -> QRectF:
        """Bounding rect of bottom controls."""
        return QRectF(25.0, 490.0, 870.0, 95.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, RamachandranPreset]]:
        rects = []
        btn_w = 166.0
        btn_h = 32.0
        start_x = 25.0
        y = 65.0
        spacing = 10.0
        for i, preset in enumerate(PRESETS):
            x = start_x + i * (btn_w + spacing)
            rects.append((QRectF(x, y, btn_w, btn_h), preset))
        return rects

    # -------------------------------------------------------------------------
    # Painting & Rendering
    # -------------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. Background
        self._paint_background(painter)

        # 2. Header
        self._paint_header(painter)

        # 3. Preset Buttons
        self._paint_preset_buttons(painter)

        # 4. 2D Ramachandran Plot Canvas
        eval_data = self.evaluate_conformation()
        self._paint_ramachandran_plot(painter, eval_data)

        # 5. Dipeptide 3D Backbone & Steric Clashing Model
        self._paint_dipeptide_model(painter, eval_data)

        # 6. Bottom Controls & Sliders
        self._paint_controls(painter)

        painter.restore()

    def _paint_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        bg_grad = QLinearGradient(0, 0, 0, self.HEIGHT)
        bg_grad.setColorAt(0.0, QColor(10, 15, 29, 250))
        bg_grad.setColorAt(1.0, QColor(15, 23, 42, 252))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(59, 130, 246, 80), 1.5))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        painter.setPen(QColor(241, 245, 249))
        font_title = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font_title)
        painter.drawText(28, 38, "HÓA SINH CẤU TRÚC: GIẢN ĐỒ RAMACHANDRAN & GẤP CUỘN PROTEIN")

        painter.setPen(QColor(148, 163, 184))
        font_sub = QFont("Segoe UI", 9)
        painter.setFont(font_sub)
        painter.drawText(28, 54, "Góc nhị diện (ϕ, ψ), va chạm không gian van der Waals & cấu trúc bậc hai protein")

        # Status badge (Live / Paused)
        badge_rect = QRectF(self.WIDTH - 150.0, 22.0, 122.0, 26.0)
        if self._is_paused:
            painter.setBrush(QColor(239, 68, 68, 40))
            painter.setPen(QPen(QColor(239, 68, 68), 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QColor(252, 165, 165))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "⏸ TẠM DỪNG")
        else:
            painter.setBrush(QColor(59, 130, 246, 40))
            painter.setPen(QPen(QColor(59, 130, 246), 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QColor(191, 219, 254))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "🔬 ĐANG MÔ PHỎNG")

    def _paint_preset_buttons(self, painter: QPainter) -> None:
        font_btn = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(font_btn)
        for i, (btn_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == self._preset_index)
            if is_active:
                painter.setBrush(QColor(37, 99, 235, 220))
                painter.setPen(QPen(QColor(191, 219, 254), 1.2))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))
            painter.drawRoundedRect(btn_rect, 6.0, 6.0)

            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(btn_rect, Qt.AlignCenter, preset.name)

    def _paint_ramachandran_plot(self, painter: QPainter, eval_data: dict[str, Any]) -> None:
        """Render 2D Ramachandran (phi, psi) contour map with energy basins and interactive probe."""
        p_rect = self._plot_rect()

        # Canvas frame
        painter.setBrush(QColor(10, 15, 26))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(p_rect, 8.0, 8.0)

        # Plot margins: [-180, +180] in both phi (x) and psi (y)
        mx0 = p_rect.left() + 35.0
        mx1 = p_rect.right() - 15.0
        my0 = p_rect.top() + 25.0
        my1 = p_rect.bottom() - 25.0
        pw = mx1 - mx0
        ph = my1 - my0

        cx = mx0 + pw * 0.5
        cy = my0 + ph * 0.5

        # 1. Grid & Axes
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        # Center axes (phi = 0, psi = 0)
        painter.drawLine(QPointF(cx, my0), QPointF(cx, my1))
        painter.drawLine(QPointF(mx0, cy), QPointF(mx1, cy))

        # Ticks and grid lines (-90, +90)
        painter.setPen(QPen(QColor(30, 41, 59), 1.0, Qt.DotLine))
        painter.drawLine(QPointF(cx - pw * 0.25, my0), QPointF(cx - pw * 0.25, my1))
        painter.drawLine(QPointF(cx + pw * 0.25, my0), QPointF(cx + pw * 0.25, my1))
        painter.drawLine(QPointF(mx0, cy - ph * 0.25), QPointF(mx1, cy - ph * 0.25))
        painter.drawLine(QPointF(mx0, cy + ph * 0.25), QPointF(mx1, cy + ph * 0.25))

        # 2. Draw Favorable & Allowed Energy Basins
        painter.save()
        painter.setClipRect(QRectF(mx0, my0, pw, ph))

        def deg_to_pt(phi_d: float, psi_d: float) -> QPointF:
            px = cx + (phi_d / 180.0) * (pw * 0.5)
            # Psi increases upwards, so inverted in screen y
            py = cy - (psi_d / 180.0) * (ph * 0.5)
            return QPointF(px, py)

        # Alpha-R Helix core region (phi in [-90, -40], psi in [-70, -20])
        a_r_pt = deg_to_pt(-57.0, -47.0)
        painter.setBrush(QColor(16, 185, 129, 60))
        painter.setPen(QPen(QColor(16, 185, 129, 140), 1.2))
        painter.drawEllipse(a_r_pt, pw * (35.0 / 360.0), ph * (35.0 / 360.0))
        # Core center
        painter.setBrush(QColor(16, 185, 129, 140))
        painter.drawEllipse(a_r_pt, pw * (18.0 / 360.0), ph * (18.0 / 360.0))

        # Beta-sheet core region (phi in [-160, -90], psi in [90, 160])
        beta_pt = deg_to_pt(-135.0, 135.0)
        painter.setBrush(QColor(59, 130, 246, 60))
        painter.setPen(QPen(QColor(59, 130, 246, 140), 1.2))
        painter.drawEllipse(beta_pt, pw * (45.0 / 360.0), ph * (45.0 / 360.0))
        # Core center
        painter.setBrush(QColor(59, 130, 246, 140))
        painter.drawEllipse(beta_pt, pw * (25.0 / 360.0), ph * (25.0 / 360.0))

        # Polyproline II (PII) / Collagen region (phi ≈ -75, psi ≈ +145)
        pii_pt = deg_to_pt(-75.0, 145.0)
        painter.setBrush(QColor(168, 85, 247, 50))
        painter.setPen(QPen(QColor(168, 85, 247, 120), 1.0))
        painter.drawEllipse(pii_pt, pw * (28.0 / 360.0), ph * (28.0 / 360.0))

        # Alpha-L Helix region (phi in [40, 80], psi in [30, 70])
        a_l_pt = deg_to_pt(57.0, 47.0)
        if self._residue_type == "glycine":
            painter.setBrush(QColor(234, 179, 8, 70))
            painter.setPen(QPen(QColor(234, 179, 8, 140), 1.2))
            painter.drawEllipse(a_l_pt, pw * (35.0 / 360.0), ph * (35.0 / 360.0))
        else:
            painter.setBrush(QColor(234, 179, 8, 30))
            painter.setPen(QPen(QColor(234, 179, 8, 70), 1.0, Qt.DashLine))
            painter.drawEllipse(a_l_pt, pw * (20.0 / 360.0), ph * (20.0 / 360.0))

        # Region labels
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(167, 243, 208))
        painter.drawText(QRectF(a_r_pt.x() - 30, a_r_pt.y() - 8, 60, 16), Qt.AlignCenter, "α_R")
        painter.setPen(QColor(191, 219, 254))
        painter.drawText(QRectF(beta_pt.x() - 30, beta_pt.y() - 8, 60, 16), Qt.AlignCenter, "β")
        painter.setPen(QColor(233, 213, 255))
        painter.drawText(QRectF(pii_pt.x() - 30, pii_pt.y() - 8, 60, 16), Qt.AlignCenter, "P_II")
        painter.setPen(QColor(254, 240, 138))
        painter.drawText(QRectF(a_l_pt.x() - 30, a_l_pt.y() - 8, 60, 16), Qt.AlignCenter, "α_L")

        # 3. Interactive Probe Marker (Current (phi, psi))
        cur_pt = deg_to_pt(self._phi_deg, self._psi_deg)
        probe_color = (
            QColor(239, 68, 68) if eval_data["is_clash"]
            else (QColor(16, 185, 129) if eval_data["region"] == "core" else QColor(56, 189, 248))
        )

        # Pulse halo
        pulse = math.sin(self._sim_time * 10.0) * 2.0
        painter.setBrush(QColor(probe_color.red(), probe_color.green(), probe_color.blue(), 60))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cur_pt, 12.0 + pulse, 12.0 + pulse)

        # Main marker
        painter.setBrush(probe_color)
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(cur_pt, 6.0, 6.0)

        painter.restore()

        # Axis Labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(QRectF(mx0, my1 + 4, pw, 16), Qt.AlignCenter, "Góc xoay ϕ (Phi) [độ]")
        painter.save()
        painter.translate(mx0 - 22, cy)
        painter.rotate(-90)
        painter.drawText(QRectF(-60, -8, 120, 16), Qt.AlignCenter, "Góc xoay ψ (Psi) [độ]")
        painter.restore()

        # Ticks values (-180, 0, +180)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(100, 116, 139))
        painter.drawText(int(mx0 - 12), int(my1 + 14), "-180°")
        painter.drawText(int(cx - 6), int(my1 + 14), "0°")
        painter.drawText(int(mx1 - 12), int(my1 + 14), "+180°")
        painter.drawText(int(mx0 - 28), int(my0 + 6), "+180°")
        painter.drawText(int(mx0 - 18), int(cy + 4), "0°")
        painter.drawText(int(mx0 - 28), int(my1 + 4), "-180°")

    def _paint_dipeptide_model(self, painter: QPainter, eval_data: dict[str, Any]) -> None:
        """Render 3D-projected dipeptide backbone model with real-time steric clash detection."""
        m_rect = self._peptide_rect()

        # Canvas frame
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(m_rect, 8.0, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(59, 130, 246))
        painter.drawText(int(m_rect.left() + 12), int(m_rect.top() + 20), "MÔ HÌNH KHÔNG GIAN DIPEPTIDE")

        # Status badge for conformation
        stat_rect = QRectF(m_rect.right() - 210, m_rect.top() + 8, 195, 22)
        if eval_data["is_clash"]:
            painter.setBrush(QColor(239, 68, 68, 40))
            painter.setPen(QPen(QColor(239, 68, 68), 1.0))
            painter.drawRoundedRect(stat_rect, 4.0, 4.0)
            painter.setPen(QColor(252, 165, 165))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.drawText(stat_rect, Qt.AlignCenter, "❌ VA CHẠM KHÔNG GIAN")
        else:
            painter.setBrush(QColor(16, 185, 129, 40))
            painter.setPen(QPen(QColor(16, 185, 129), 1.0))
            painter.drawRoundedRect(stat_rect, 4.0, 4.0)
            painter.setPen(QColor(110, 231, 183))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.drawText(stat_rect, Qt.AlignCenter, f"✓ {eval_data['structure_name']}")

        # 3D projection of dipeptide backbone atoms
        # Atoms: C'_{i-1} -> N_i -> C_alpha_i -> C'_i -> N_{i+1} -> C_alpha_{i+1}
        # Torsion angles phi rotates around N_i - C_alpha_i, psi rotates around C_alpha_i - C'_i
        phi_rad = math.radians(self._phi_deg)
        psi_rad = math.radians(self._psi_deg)
        vib = math.sin(self._sim_time * 8.0) * 1.5

        cx = m_rect.left() + m_rect.width() * 0.48
        cy = m_rect.top() + m_rect.height() * 0.50

        # Backbone nodes (2D projected coordinates)
        # N_i
        n_x = cx - 85.0
        n_y = cy + 15.0
        # C_alpha_i (central pivot)
        ca_x = cx - 20.0
        ca_y = cy - 25.0
        # C'_i
        cp_x = cx + 55.0
        cp_y = cy + 15.0

        # Carbonyl Oxygen O_i (rotates with psi)
        o_dist = 52.0
        o_ang = -math.pi * 0.5 + psi_rad * 0.75
        o_x = cp_x + o_dist * math.cos(o_ang)
        o_y = cp_y + o_dist * math.sin(o_ang)

        # Amide Hydrogen H_i (rotates with phi)
        h_dist = 42.0
        h_ang = math.pi * 0.65 - phi_rad * 0.75
        h_x = n_x + h_dist * math.cos(h_ang)
        h_y = n_y + h_dist * math.sin(h_ang)

        # Sidechain C_beta
        cb_x = ca_x + 10.0 + vib
        cb_y = ca_y - 55.0

        # Next Nitrogen N_{i+1} (rotates with psi)
        n_next_dist = 70.0
        n_next_ang = 0.35 + psi_rad * 0.65
        n_next_x = cp_x + n_next_dist * math.cos(n_next_ang)
        n_next_y = cp_y + n_next_dist * math.sin(n_next_ang)

        # Prev C' (C'_{i-1})
        cp_prev_dist = 65.0
        cp_prev_ang = math.pi * 0.95 + phi_rad * 0.65
        cp_prev_x = n_x + cp_prev_dist * math.cos(cp_prev_ang)
        cp_prev_y = n_y + cp_prev_dist * math.sin(cp_prev_ang)

        # 1. Draw Bonds
        painter.setPen(QPen(QColor(148, 163, 184), 4.5))
        painter.drawLine(QPointF(cp_prev_x, cp_prev_y), QPointF(n_x, n_y))
        painter.drawLine(QPointF(n_x, n_y), QPointF(ca_x, ca_y))
        painter.drawLine(QPointF(ca_x, ca_y), QPointF(cp_x, cp_y))
        painter.drawLine(QPointF(cp_x, cp_y), QPointF(n_next_x, n_next_y))
        # Carbonyl double bond C'=O
        painter.setPen(QPen(QColor(239, 68, 68), 3.5))
        painter.drawLine(QPointF(cp_x, cp_y), QPointF(o_x, o_y))
        # N-H bond
        painter.setPen(QPen(QColor(226, 232, 240), 2.5))
        painter.drawLine(QPointF(n_x, n_y), QPointF(h_x, h_y))
        # Sidechain C_alpha - C_beta
        if self._residue_type != "glycine":
            painter.setPen(QPen(QColor(234, 179, 8), 3.5))
            painter.drawLine(QPointF(ca_x, ca_y), QPointF(cb_x, cb_y))

        # 2. Hydrogen bond dashed line (if active)
        if eval_data["h_bond"]:
            painter.setPen(QPen(QColor(16, 185, 129), 2.0, Qt.DashLine))
            painter.drawLine(QPointF(o_x, o_y), QPointF(n_next_x, n_next_y))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(110, 231, 183))
            painter.drawText(int((o_x + n_next_x) * 0.5 - 25), int((o_y + n_next_y) * 0.5 - 6), "H-Bond")

        # 3. Steric Clash Warning Rings (Pulsing Red)
        if eval_data["is_clash"]:
            clash_dist = math.sqrt((o_x - h_x)**2 + (o_y - h_y)**2)
            clash_mid_x = (o_x + h_x) * 0.5
            clash_mid_y = (o_y + h_y) * 0.5

            pulse_r = 16.0 + math.sin(self._sim_time * 15.0) * 4.0
            painter.setBrush(QColor(239, 68, 68, 120))
            painter.setPen(QPen(QColor(254, 202, 202), 2.0))
            painter.drawEllipse(QPointF(clash_mid_x, clash_mid_y), pulse_r, pulse_r)

            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(QRectF(clash_mid_x - 50, clash_mid_y - 10, 100, 20), Qt.AlignCenter, "CLASH!")

        # 4. Draw Atomic Spheres
        atoms = [
            (cp_prev_x, cp_prev_y, 14.0, QColor(71, 85, 105), "C'"),
            (n_x, n_y, 16.0, QColor(59, 130, 246), "N"),
            (ca_x, ca_y, 18.0, QColor(100, 116, 139), "Cα"),
            (cp_x, cp_y, 15.0, QColor(71, 85, 105), "C'"),
            (o_x, o_y, 15.0, QColor(239, 68, 68), "O"),
            (h_x, h_y, 11.0, QColor(241, 245, 249), "H"),
            (n_next_x, n_next_y, 16.0, QColor(59, 130, 246), "N"),
        ]
        if self._residue_type != "glycine":
            atoms.append((cb_x, cb_y, 16.0, QColor(234, 179, 8), "Cβ"))

        for ax, ay, ar, col, sym in atoms:
            grad = QRadialGradient(ax - ar * 0.3, ay - ar * 0.3, ar)
            grad.setColorAt(0.0, QColor(255, 255, 255, 200))
            grad.setColorAt(0.3, col)
            grad.setColorAt(1.0, QColor(col.red() // 2, col.green() // 2, col.blue() // 2))
            painter.setBrush(QBrush(grad))
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
            painter.drawEllipse(QPointF(ax, ay), ar, ar)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRectF(ax - ar, ay - ar, ar * 2, ar * 2), Qt.AlignCenter, sym)

        # Dihedral Angles Telemetry Box
        info_rect = QRectF(m_rect.left() + 15, m_rect.bottom() - 65, m_rect.width() - 30, 50)
        painter.setBrush(QColor(10, 15, 26, 200))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(info_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(int(info_rect.left() + 10), int(info_rect.top() + 18), f"ϕ (N-Cα): {self._phi_deg:.1f}°")
        painter.drawText(int(info_rect.left() + 140), int(info_rect.top() + 18), f"ψ (Cα-C'): {self._psi_deg:.1f}°")
        painter.drawText(int(info_rect.left() + 270), int(info_rect.top() + 18), f"ω (Peptide): {self._omega_deg:.1f}°")

        painter.drawText(int(info_rect.left() + 10), int(info_rect.top() + 38), f"Axit amin: {self._residue_type.upper()}")
        painter.drawText(int(info_rect.left() + 140), int(info_rect.top() + 38), f"Vùng: {eval_data['region'].upper()}")
        painter.setPen(QColor(16, 185, 129) if eval_data['h_bond'] else QColor(148, 163, 184))
        painter.drawText(int(info_rect.left() + 270), int(info_rect.top() + 38), f"H-Bond: {'CÓ' if eval_data['h_bond'] else 'KHÔNG'}")

    def _paint_controls(self, painter: QPainter) -> None:
        """Render interactive bottom sliders for phi and psi dihedral angles."""
        ctrl_rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(ctrl_rect, 8.0, 8.0)

        # 2 Wide Sliders (Phi & Psi)
        col_w = (ctrl_rect.width() - 50.0) / 2.0
        sliders = [
            ("phi", "Góc nhị diện ϕ (N - Cα)", f"{self._phi_deg:.1f}°", self._phi_deg, -180.0, 180.0),
            ("psi", "Góc nhị diện ψ (Cα - C')", f"{self._psi_deg:.1f}°", self._psi_deg, -180.0, 180.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * (col_w + 10.0)
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(59, 130, 246))
            painter.drawText(QRectF(cx, cy - 12, col_w - 20, 14), Qt.AlignRight, val_text)

            # Track
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx, track_y, track_w, 6.0)
            painter.setBrush(QColor(30, 41, 59))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(track_rect, 3.0, 3.0)

            # Active fill from center (0 deg)
            norm = (val - min_v) / (max_v - min_v)
            fill_rect = QRectF(cx, track_y, track_w * norm, 6.0)
            painter.setBrush(QColor(37, 99, 235))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(37, 99, 235), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Bottom residue selector pills (General / Glycine / Proline)
        pill_y = ctrl_rect.top() + 62.0
        pills = [("general", "Chuẩn (Ala/Leu/Val)"), ("glycine", "Glycine (Gly)"), ("proline", "Proline (Pro)")]
        for idx, (res_key, res_lbl) in enumerate(pills):
            p_rect = QRectF(ctrl_rect.left() + 20 + idx * 160, pill_y, 145, 24)
            is_sel = (self._residue_type == res_key)
            painter.setBrush(QColor(37, 99, 235, 200) if is_sel else QColor(30, 41, 59))
            painter.setPen(QPen(QColor(191, 219, 254) if is_sel else QColor(71, 85, 105), 1.0))
            painter.drawRoundedRect(p_rect, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(255, 255, 255) if is_sel else QColor(203, 213, 225))
            painter.drawText(p_rect, Qt.AlignCenter, res_lbl)

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Status badge click (pause / resume)
        badge_rect = QRectF(self.WIDTH - 150.0, 22.0, 122.0, 26.0)
        if badge_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Check 2D Ramachandran Plot drag
        p_rect = self._plot_rect()
        mx0 = p_rect.left() + 35.0
        mx1 = p_rect.right() - 15.0
        my0 = p_rect.top() + 25.0
        my1 = p_rect.bottom() - 25.0
        plot_bounds = QRectF(mx0, my0, mx1 - mx0, my1 - my0)

        if plot_bounds.contains(pos):
            self._dragging_probe = True
            self._update_angles_from_plot(pos, plot_bounds)
            event.accept()
            return

        # Check Residue type pills
        ctrl_rect = self._controls_rect()
        pill_y = ctrl_rect.top() + 62.0
        pills = ["general", "glycine", "proline"]
        for idx, res_key in enumerate(pills):
            p_rect = QRectF(ctrl_rect.left() + 20 + idx * 160, pill_y, 145, 24)
            if p_rect.contains(pos):
                self.set_residue_type(res_key)
                event.accept()
                return

        # Check Sliders
        col_w = (ctrl_rect.width() - 50.0) / 2.0
        for i, key in enumerate(["phi", "psi"]):
            cx = ctrl_rect.left() + 20.0 + i * (col_w + 10.0)
            cy = ctrl_rect.top() + 18.0
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx - 5.0, track_y - 8.0, track_w + 10.0, 22.0)
            if track_rect.contains(pos):
                self._dragging_slider = key
                self._update_slider_from_pos(key, pos.x(), cx, track_w)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_probe:
            p_rect = self._plot_rect()
            mx0 = p_rect.left() + 35.0
            mx1 = p_rect.right() - 15.0
            my0 = p_rect.top() + 25.0
            my1 = p_rect.bottom() - 25.0
            plot_bounds = QRectF(mx0, my0, mx1 - mx0, my1 - my0)
            self._update_angles_from_plot(pos, plot_bounds)
            event.accept()
            return

        if self._dragging_slider:
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 50.0) / 2.0
            slider_keys = ["phi", "psi"]
            if self._dragging_slider in slider_keys:
                idx = slider_keys.index(self._dragging_slider)
                cx = ctrl_rect.left() + 20.0 + idx * (col_w + 10.0)
                track_w = col_w - 25.0
                self._update_slider_from_pos(self._dragging_slider, pos.x(), cx, track_w)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_probe = False
        self._dragging_slider = None
        super().mouseReleaseEvent(event)

    def _update_angles_from_plot(self, pos: QPointF, bounds: QRectF) -> None:
        norm_x = max(0.0, min(1.0, (pos.x() - bounds.left()) / bounds.width()))
        # Inverted for psi (top is +180, bottom is -180)
        norm_y = max(0.0, min(1.0, (pos.y() - bounds.top()) / bounds.height()))
        new_phi = -180.0 + norm_x * 360.0
        new_psi = 180.0 - norm_y * 360.0
        self.set_phi(new_phi)
        self.set_psi(new_psi)

    def _update_slider_from_pos(self, key: str, px: float, cx: float, track_w: float) -> None:
        norm = max(0.0, min(1.0, (px - cx) / track_w))
        val = -180.0 + norm * 360.0
        if key == "phi":
            self.set_phi(val)
        elif key == "psi":
            self.set_psi(val)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveRamachandranPlotItem",
            "preset_key": self.current_preset.key,
            "phi_deg": self._phi_deg,
            "psi_deg": self._psi_deg,
            "omega_deg": self._omega_deg,
            "residue_type": self._residue_type,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveRamachandranPlotItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "alpha_helix")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "phi_deg" in data:
            item.set_phi(float(data["phi_deg"]))
        if "psi_deg" in data:
            item.set_psi(float(data["psi_deg"]))
        if "residue_type" in data:
            item.set_residue_type(data["residue_type"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
