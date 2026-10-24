"""
AirTeach AI — Interactive Quantum Chemistry & Molecular Orbital Theory:
Atomic Orbitals, Hybridization (sp, sp2, sp3) & Molecular Orbital LCAO Simulator.
Milestone 150 (Task I.137).

Provides real-time interactive simulation of:
1. Linear Combination of Atomic Orbitals (LCAO-MO):
   psi_bonding = (phi_A + phi_B) / sqrt(2 * (1 + S))
   psi_antibonding = (phi_A - phi_B) / sqrt(2 * (1 - S))
2. Overlap integral for 1s-1s: S(rho) = exp(-rho) * (1 + rho + rho^2 / 3) where rho = R / a_0.
3. Hybridization geometry: sp (180° linear), sp2 (120° trigonal planar), sp3 (109.47° tetrahedral).
4. Bond Order: BO = (N_bonding - N_antibonding) / 2.
5. Molecular Orbital Energy Level splitting diagrams with electron spin pairings.
6. 5 Authentic Curriculum Presets: Hydrogen H2 sigma bond, Carbon sp3 in Methane,
   Ethylene sp2 with pi bonding, Helium dimer He2 instability, and Paramagnetic Oxygen O2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QLinearGradient,
    QRadialGradient,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsObject,
    QGraphicsItem,
)

# Fundamental Physical Constants
BOHR_RADIUS_ANGSTROM: float = 0.529177210903  # Bohr radius a_0 in Angstroms


@dataclass(frozen=True)
class OrbitalPreset:
    """Represents a curriculum scenario in quantum chemistry."""
    code: str
    title_vi: str
    molecule_name: str
    orbital_type: str             # "mo_sigma", "mo_pi", "hybrid_sp", "hybrid_sp2", "hybrid_sp3"
    default_distance_angstrom: float
    n_bonding_electrons: int
    n_antibonding_electrons: int
    bond_order: float
    hybridization_str: str
    accent_color: QColor
    description: str


ORBITAL_PRESETS: dict[str, OrbitalPreset] = {
    "HYDROGEN_MOLECULE_H2_SIGMA_BOND": OrbitalPreset(
        code="HYDROGEN_MOLECULE_H2_SIGMA_BOND",
        title_vi="Phân tử Hydro H₂: Liên kết Sigma 1s-1s (Bonding σ₁ₛ)",
        molecule_name="Hydrogen Molecule (H₂)",
        orbital_type="mo_sigma",
        default_distance_angstrom=0.74, # 74 pm equilibrium bond length
        n_bonding_electrons=2,
        n_antibonding_electrons=0,
        bond_order=1.0,
        hybridization_str="1s + 1s (Không lai hóa)",
        accent_color=QColor(0, 210, 255), # Cyan
        description="Xen phủ trục đồng pha (+/+) của hai obitan 1s tạo liên kết đơn bền vững, mật độ điện tử tích tụ giữa hai hạt nhân.",
    ),
    "CARBON_SP3_METHANE_CH4_TETRAHEDRAL": OrbitalPreset(
        code="CARBON_SP3_METHANE_CH4_TETRAHEDRAL",
        title_vi="Carbon Lai Hóa sp³ Tứ Diện trong Metan (CH₄ Tetrahedral)",
        molecule_name="Methane (CH₄, C sp³)",
        orbital_type="hybrid_sp3",
        default_distance_angstrom=1.09, # C-H bond distance 1.09 Å
        n_bonding_electrons=8,
        n_antibonding_electrons=0,
        bond_order=1.0,
        hybridization_str="sp³ (25% s, 75% p • 109.47°)",
        accent_color=QColor(76, 201, 240), # Vibrant sky
        description="Sự tổ hợp 1 obitan 2s và 3 obitan 2p hướng về 4 đỉnh hình tứ diện đều với góc liên kết 109.5°.",
    ),
    "ETHYLENE_SP2_PI_BONDING_C2H4": OrbitalPreset(
        code="ETHYLENE_SP2_PI_BONDING_C2H4",
        title_vi="Etylen C₂H₄: Lai Hóa sp² & Liên Kết Pi (sp² + 2p_z π)",
        molecule_name="Ethylene (C₂H₄, C sp²)",
        orbital_type="hybrid_sp2",
        default_distance_angstrom=1.34, # C=C double bond length
        n_bonding_electrons=4,
        n_antibonding_electrons=0,
        bond_order=2.0,
        hybridization_str="sp² (33% s, 67% p • 120°) + π",
        accent_color=QColor(247, 37, 133), # Magenta
        description="Mặt phẳng tam giác 120° từ khung liên kết sigma sp² kết hợp xen phủ bên obitan 2p_z tạo liên kết đôi C=C.",
    ),
    "HELIUM_DIMER_HE2_UNSTABLE": OrbitalPreset(
        code="HELIUM_DIMER_HE2_UNSTABLE",
        title_vi="He₂ Không Tồn Tại: Phản Liên Kết σ* Khử Liên Kết (BO = 0)",
        molecule_name="Helium Dimer (He₂)",
        orbital_type="mo_sigma",
        default_distance_angstrom=2.50, # Van der Waals distance
        n_bonding_electrons=2,
        n_antibonding_electrons=2,
        bond_order=0.0,
        hybridization_str="σ₁ₛ² σ*₁ₛ² (Bậc liên kết = 0)",
        accent_color=QColor(239, 71, 111), # Red coral
        description="Cả hai obitan liên kết σ và phản liên kết σ* đều bị lấp đầy bởi 4 electron, triệt tiêu hoàn toàn lực liên kết.",
    ),
    "PARAMAGNETIC_OXYGEN_O2_DIRADICAL": OrbitalPreset(
        code="PARAMAGNETIC_OXYGEN_O2_DIRADICAL",
        title_vi="Oxy Thuận Từ O₂: 2 Electron Độc Thân Obitan π* (Paramagnetic)",
        molecule_name="Diatomic Oxygen (O₂)",
        orbital_type="mo_pi",
        default_distance_angstrom=1.21, # O=O bond length
        n_bonding_electrons=8,
        n_antibonding_electrons=4,
        bond_order=2.0,
        hybridization_str="π* degenerate diradical",
        accent_color=QColor(255, 183, 3), # Gold amber
        description="Thuyết MO giải thích xuất sắc từ tính của oxy lỏng: 2 electron độc thân ở hai obitan phản liên kết π* phân cực thuận từ.",
    ),
}


class InteractiveMolecularOrbitalItem(QGraphicsObject):
    """
    High-fidelity interactive QGraphicsObject simulating Quantum Molecular Orbitals (LCAO-MO),
    Overlap Integrals S(R), Hybridization Geometry (sp, sp2, sp3), and Energy Level Diagrams.
    """

    # Signals
    orbital_computed = Signal(dict)
    stamp_requested = Signal(dict)

    CARD_W: float = 680.0
    CARD_H: float = 410.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "HYDROGEN_MOLECULE_H2_SIGMA_BOND"
        self._preset: OrbitalPreset = ORBITAL_PRESETS[self._active_preset_code]

        self._distance_angstrom: float = self._preset.default_distance_angstrom
        self._is_antibonding_view: bool = False
        self._rotation_angle_deg: float = 0.0

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None

        # Timer for smooth 3D orbital lobe rotation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(20)

    # ------------------------------------------------------------------
    # Physics Properties & Analytical Quantum Chemistry Equations
    # ------------------------------------------------------------------

    @property
    def distance_bohr_rho(self) -> float:
        """Internuclear distance in units of Bohr radius: rho = R / a_0."""
        return max(0.2, self._distance_angstrom) / BOHR_RADIUS_ANGSTROM

    @property
    def overlap_integral_S(self) -> float:
        """
        Analytical Overlap Integral S(rho) for two 1s orbitals:
        S(rho) = exp(-rho) * (1 + rho + rho^2 / 3)
        Strictly ranges in (0, 1].
        """
        rho = self.distance_bohr_rho
        s_val = math.exp(-rho) * (1.0 + rho + (rho * rho) / 3.0)
        return max(0.0, min(1.0, s_val))

    @property
    def bonding_energy_eV(self) -> float:
        """
        Bond stabilization energy delta E in eV:
        Delta E_bonding ≈ (beta - alpha * S) / (1 + S)
        Normalized relative to equilibrium depth ~4.52 eV for H2.
        """
        s = self.overlap_integral_S
        r = self._distance_angstrom
        # Morse potential shape
        r_eq = 0.74
        de = 4.52  # Dissociation energy in eV
        a = 1.94   # Parameter in 1/Angstrom
        morse = de * (math.exp(-2.0 * a * (r - r_eq)) - 2.0 * math.exp(-a * (r - r_eq)))
        return morse

    @property
    def bond_order(self) -> float:
        """Bond Order: BO = (N_bonding - N_antibonding) / 2."""
        return float(self._preset.n_bonding_electrons - self._preset.n_antibonding_electrons) / 2.0

    # ------------------------------------------------------------------
    # State Mutations & Actions
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Switches curriculum scenario."""
        if preset_code in ORBITAL_PRESETS:
            self._active_preset_code = preset_code
            self._preset = ORBITAL_PRESETS[preset_code]
            self._distance_angstrom = self._preset.default_distance_angstrom
            self._is_antibonding_view = False
            self._emit_computed_values()
            self.update()

    def set_distance_angstrom(self, dist_a: float) -> None:
        """Sets internuclear distance R in Angstroms within [0.4, 4.0] Å."""
        self._distance_angstrom = max(0.4, min(4.0, round(dist_a, 2)))
        self._emit_computed_values()
        self.update()

    def toggle_bonding_mode(self) -> None:
        """Toggles between Bonding (sigma/pi) and Antibonding (sigma*/pi*) view."""
        self._is_antibonding_view = not self._is_antibonding_view
        self._emit_computed_values()
        self.update()

    def reset_defaults(self) -> None:
        """Restores default values of current preset."""
        self.set_preset(self._active_preset_code)

    def _emit_computed_values(self) -> None:
        """Emits telemetry dictionary."""
        data = {
            "preset_code": self._active_preset_code,
            "molecule_name": self._preset.molecule_name,
            "orbital_type": self._preset.orbital_type,
            "distance_angstrom": self._distance_angstrom,
            "overlap_integral_S": self.overlap_integral_S,
            "bond_order": self.bond_order,
            "is_antibonding": self._is_antibonding_view,
            "bonding_energy_eV": self.bonding_energy_eV,
        }
        self.orbital_computed.emit(data)

    def _on_sim_tick(self) -> None:
        """Animation tick for subtle orbital phase pulsing and 3D rotation."""
        self._rotation_angle_deg = (self._rotation_angle_deg + 0.6) % 360.0
        self.update()

    # ------------------------------------------------------------------
    # UI Layout & Buttons
    # ------------------------------------------------------------------

    def _layout_buttons(self) -> None:
        """Arranges interactive control buttons in bottom panel."""
        ctrl_x = 385.0
        row1_y = 248.0
        btn_w_half = 72.0
        btn_h = 24.0

        self._buttons["preset_prev"] = QRectF(ctrl_x, row1_y, 45.0, btn_h)
        self._buttons["preset_next"] = QRectF(ctrl_x + 52.0, row1_y, 45.0, btn_h)
        self._buttons["mode_toggle"] = QRectF(ctrl_x + 104.0, row1_y, 172.0, btn_h)

        row2_y = 278.0
        self._buttons["r_dec"] = QRectF(ctrl_x, row2_y, btn_w_half, btn_h)
        self._buttons["r_inc"] = QRectF(ctrl_x + 78.0, row2_y, btn_w_half, btn_h)
        self._buttons["r_eq"] = QRectF(ctrl_x + 158.0, row2_y, 118.0, btn_h)

        row3_y = 308.0
        self._buttons["set_sp"] = QRectF(ctrl_x, row3_y, 62.0, btn_h)
        self._buttons["set_sp2"] = QRectF(ctrl_x + 68.0, row3_y, 62.0, btn_h)
        self._buttons["set_sp3"] = QRectF(ctrl_x + 136.0, row3_y, 62.0, btn_h)
        self._buttons["set_h2"] = QRectF(ctrl_x + 204.0, row3_y, 72.0, btn_h)

        row4_y = 342.0
        self._buttons["reset"] = QRectF(ctrl_x, row4_y, 100.0, 26.0)
        self._buttons["stamp"] = QRectF(ctrl_x + 108.0, row4_y, 168.0, 26.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # ------------------------------------------------------------------
    # Rendering Pipeline
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option=None, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card
        self._paint_card_background(painter)

        # 2. Header and Title Banner
        self._paint_header(painter)

        # 3. Left Side: Quantum Wavefunction Iso-Surface Visualization
        self._paint_orbital_wavefunction(painter)

        # 4. Right Side: MO Energy Level Diagram & Telemetry
        self._paint_energy_level_diagram(painter)

        # 5. Buttons
        self._paint_buttons(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        rect = self.boundingRect()
        bg_grad = QLinearGradient(0.0, 0.0, rect.width(), rect.height())
        bg_grad.setColorAt(0.0, QColor(14, 18, 28, 245))
        bg_grad.setColorAt(0.6, QColor(18, 24, 38, 245))
        bg_grad.setColorAt(1.0, QColor(12, 16, 26, 250))

        painter.setPen(QPen(QColor(45, 62, 90, 220), 1.5))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        font_title = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(20, 28, "⚛ CƠ HỌC LƯỢNG TỬ: OBITAN NGUYÊN TỬ & PHÂN TỬ (MO-LCAO)")

        font_sub = QFont("Segoe UI", 8, QFont.Weight.Normal)
        painter.setFont(font_sub)
        painter.setPen(QColor(160, 180, 210))
        is_anti = self._is_antibonding_view
        mode_txt = "Phản liên kết (Antibonding *)" if is_anti else "Liên kết (Bonding)"
        painter.drawText(20, 44, f"Kịch bản: {self._preset.title_vi} | Chế độ: {mode_txt}")

        # Badge (Top Right)
        badge_rect = QRectF(self.CARD_W - 200.0, 14.0, 180.0, 22.0)
        accent = self._preset.accent_color
        painter.setPen(QPen(accent, 1.2))
        painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 40)))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        badge_txt = f"{self._preset.hybridization_str[:18]} • BO={self.bond_order:.1f}"
        font_badge = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_badge)
        painter.setPen(accent)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_txt)

    def _paint_orbital_wavefunction(self, painter: QPainter) -> None:
        """
        Renders the 3D-styled quantum electron cloud wavefunction iso-surfaces:
        - Atomic nuclei
        - Positive phase lobes (+): Cyan / Azure gradient
        - Negative phase lobes (-): Crimson / Ruby gradient
        - Nodal planes in antibonding mode
        """
        box = QRectF(20.0, 56.0, 345.0, 180.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.2))
        painter.setBrush(QBrush(QColor(8, 12, 20, 220)))
        painter.drawRoundedRect(box, 8.0, 8.0)

        # Center of orbital rendering
        cx = (box.left() + box.right()) * 0.5
        cy = (box.top() + box.bottom()) * 0.5

        # Visual separation based on internuclear distance
        r_px = max(25.0, min(110.0, self._distance_angstrom * 45.0))
        nuc_A = QPointF(cx - r_px * 0.5, cy)
        nuc_B = QPointF(cx + r_px * 0.5, cy)

        is_anti = self._is_antibonding_view
        is_hybrid = "hybrid" in self._preset.orbital_type

        if is_hybrid:
            # Render Hybrid Orbitals (sp, sp2, or sp3) around single central nucleus
            self._paint_hybrid_lobes(painter, QPointF(cx, cy))
        else:
            # Render Diatomic Molecular Orbital (H2, He2, O2)
            self._paint_diatomic_mo(painter, nuc_A, nuc_B, is_anti)

        # Nuclei dots
        if not is_hybrid:
            for nuc, name in [(nuc_A, "A"), (nuc_B, "B")]:
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.setBrush(QBrush(QColor(255, 80, 80)))
                painter.drawEllipse(nuc, 4.0, 4.0)

                font_n = QFont("Segoe UI", 7, QFont.Weight.Bold)
                painter.setFont(font_n)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(int(nuc.x() - 4.0), int(nuc.y() + 15.0), name)

        # Distance indicator arrow between nuclei
        if not is_hybrid:
            painter.setPen(QPen(QColor(255, 210, 80), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(nuc_A.x(), nuc_A.y() - 35.0, nuc_B.x(), nuc_B.y() - 35.0)

            font_d = QFont("Segoe UI", 7, QFont.Weight.Bold)
            painter.setFont(font_d)
            painter.setPen(QColor(255, 210, 80))
            painter.drawText(
                int(cx - 30.0),
                int(cy - 40.0),
                f"R = {self._distance_angstrom:.2f} Å (S = {self.overlap_integral_S:.3f})",
            )

        # Telemetry panel below orbital box (y = 245 to 395)
        self._paint_telemetry_table(painter)

    def _paint_diatomic_mo(self, painter: QPainter, nuc_A: QPointF, nuc_B: QPointF, is_anti: bool) -> None:
        """Renders bonding (symmetric single lobe) or antibonding (opposed lobes with nodal line)."""
        mid_x = (nuc_A.x() + nuc_B.x()) * 0.5
        cy = nuc_A.y()

        if not is_anti:
            # Bonding Orbital: Large constructive overlap between nuclei
            # Draw combined ovate contour
            lobe_w = (nuc_B.x() - nuc_A.x()) + 60.0
            lobe_h = 55.0
            lobe_rect = QRectF(mid_x - lobe_w * 0.5, cy - lobe_h * 0.5, lobe_w, lobe_h)

            grad = QRadialGradient(QPointF(mid_x, cy), lobe_w * 0.5)
            grad.setColorAt(0.0, QColor(0, 230, 255, 220)) # High density core
            grad.setColorAt(0.7, QColor(0, 160, 220, 120))
            grad.setColorAt(1.0, QColor(0, 120, 180, 0))

            painter.setPen(QPen(QColor(0, 230, 255), 1.4))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(lobe_rect)

            # Positive phase label (+)
            font_ph = QFont("Segoe UI", 12, QFont.Weight.Bold)
            painter.setFont(font_ph)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(mid_x - 6.0), int(cy + 4.0), "+")
        else:
            # Antibonding Orbital: Destructive interference, nodal line at mid_x
            # Left lobe (+)
            lobe_w = 42.0
            lobe_h = 48.0
            left_rect = QRectF(nuc_A.x() - lobe_w * 0.7, cy - lobe_h * 0.5, lobe_w, lobe_h)
            grad_l = QRadialGradient(left_rect.center(), lobe_w * 0.5)
            grad_l.setColorAt(0.0, QColor(0, 230, 255, 200))
            grad_l.setColorAt(1.0, QColor(0, 120, 180, 0))
            painter.setPen(QPen(QColor(0, 230, 255), 1.2))
            painter.setBrush(QBrush(grad_l))
            painter.drawEllipse(left_rect)

            # Right lobe (-)
            right_rect = QRectF(nuc_B.x() - lobe_w * 0.3, cy - lobe_h * 0.5, lobe_w, lobe_h)
            grad_r = QRadialGradient(right_rect.center(), lobe_w * 0.5)
            grad_r.setColorAt(0.0, QColor(255, 60, 80, 200))
            grad_r.setColorAt(1.0, QColor(180, 20, 40, 0))
            painter.setPen(QPen(QColor(255, 80, 80), 1.2))
            painter.setBrush(QBrush(grad_r))
            painter.drawEllipse(right_rect)

            # Nodal line
            painter.setPen(QPen(QColor(255, 255, 255), 1.4, Qt.PenStyle.DashLine))
            painter.drawLine(mid_x, cy - 35.0, mid_x, cy + 35.0)

            # Phase labels
            font_ph = QFont("Segoe UI", 10, QFont.Weight.Bold)
            painter.setFont(font_ph)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(left_rect.center().x() - 5.0), int(cy + 4.0), "+")
            painter.drawText(int(right_rect.center().x() - 5.0), int(cy + 4.0), "−")

            # Nodal plane caption
            font_node = QFont("Segoe UI", 6, QFont.Weight.Bold)
            painter.setFont(font_node)
            painter.setPen(QColor(255, 200, 200))
            painter.drawText(int(mid_x - 32.0), int(cy + 48.0), "Mặt nút (Node ψ=0)")

    def _paint_hybrid_lobes(self, painter: QPainter, center: QPointF) -> None:
        """Renders directional hybrid lobes (sp, sp2, or sp3)."""
        # Central atom nucleus
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.setBrush(QBrush(QColor(240, 180, 50)))
        painter.drawEllipse(center, 5.0, 5.0)

        orb_type = self._preset.orbital_type
        angles_deg = []
        if orb_type == "hybrid_sp":
            angles_deg = [0.0, 180.0]
        elif orb_type == "hybrid_sp2":
            angles_deg = [90.0, 210.0, 330.0]
        else: # sp3 tetrahedral projection
            angles_deg = [90.0, 210.0, 330.0, 270.0]

        lobe_len = 50.0
        lobe_w = 26.0

        for ang in angles_deg:
            rad = math.radians(ang + self._rotation_angle_deg * 0.2)
            dx = math.cos(rad)
            dy = math.sin(rad)

            tip = QPointF(center.x() + dx * lobe_len, center.y() + dy * lobe_len)
            mid = QPointF(center.x() + dx * (lobe_len * 0.6), center.y() + dy * (lobe_len * 0.6))

            # Draw directional ovate lobe
            painter.setPen(QPen(self._preset.accent_color, 1.2))
            grad = QRadialGradient(mid, lobe_len * 0.5)
            grad.setColorAt(0.0, QColor(0, 210, 255, 180))
            grad.setColorAt(1.0, QColor(0, 80, 160, 20))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(mid, lobe_w * 0.5, lobe_len * 0.4)

    def _paint_telemetry_table(self, painter: QPainter) -> None:
        """Renders exact quantum parameters table."""
        t_box = QRectF(20.0, 245.0, 345.0, 150.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.0))
        painter.setBrush(QBrush(QColor(10, 15, 25, 220)))
        painter.drawRoundedRect(t_box, 8.0, 8.0)

        font_h = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_h)
        painter.setPen(QColor(240, 200, 100))
        painter.drawText(int(t_box.left() + 12.0), int(t_box.top() + 18.0), "⚛ THÔNG SỐ HÓA LƯỢNG TỬ & OBITAN PHÂN TỬ")

        font_k = QFont("Segoe UI", 7, QFont.Weight.Normal)
        font_v = QFont("Consolas", 8, QFont.Weight.Bold)

        is_anti = self._is_antibonding_view
        lines = [
            ("Khoảng cách liên hạt nhân R:", f"{self._distance_angstrom:.2f} Å ({self._distance_angstrom*100:.0f} pm)", QColor("#38bdf8")),
            ("Tích phân xen phủ S(R):", f"{self.overlap_integral_S:.4f}", QColor("#4ade80")),
            ("Bậc liên kết (Bond Order):", f"{self.bond_order:.1f}", QColor("#fbbf24")),
            ("Năng lượng ổn định ΔE:", f"{self.bonding_energy_eV:+.2f} eV", QColor("#f43f5e") if is_anti else QColor("#2dd4bf")),
            ("Trạng thái quan sát:", "Phản liên kết σ* (Không bền)" if is_anti else "Liên kết σ (Bền vững)", QColor("#f87171") if is_anti else QColor("#38bdf8")),
        ]

        curr_y = t_box.top() + 38.0
        for lbl, val_str, val_col in lines:
            painter.setFont(font_k)
            painter.setPen(QColor(160, 180, 205))
            painter.drawText(int(t_box.left() + 12.0), int(curr_y), lbl)

            painter.setFont(font_v)
            painter.setPen(val_col)
            painter.drawText(int(t_box.left() + 195.0), int(curr_y), val_str)
            curr_y += 22.0

    def _paint_energy_level_diagram(self, painter: QPainter) -> None:
        """Renders Molecular Orbital Energy Level Splitting Diagram."""
        plot_rect = QRectF(385.0, 56.0, 275.0, 180.0)
        painter.setPen(QPen(QColor(40, 56, 82), 1.0))
        painter.setBrush(QBrush(QColor(12, 17, 28, 220)))
        painter.drawRoundedRect(plot_rect, 8.0, 8.0)

        font_hdr = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_hdr)
        painter.setPen(QColor(240, 200, 100))
        painter.drawText(int(plot_rect.left() + 12.0), int(plot_rect.top() + 18.0), "📊 GIẢN ĐỒ NĂNG LƯỢNG MO (ENERGY SPLITTING)")

        # Diagram coordinates
        y_mid = plot_rect.top() + 90.0
        x_ao_A = plot_rect.left() + 35.0
        x_ao_B = plot_rect.right() - 65.0
        x_mo = (plot_rect.left() + plot_rect.right()) * 0.5 - 20.0
        bar_w = 32.0

        # Atomic Orbitals (AO) Level (1s_A and 1s_B)
        painter.setPen(QPen(QColor(160, 180, 210), 2.0))
        painter.drawLine(x_ao_A, y_mid, x_ao_A + bar_w, y_mid)
        painter.drawLine(x_ao_B, y_mid, x_ao_B + bar_w, y_mid)

        # Splitting based on overlap S
        s = self.overlap_integral_S
        split_dy = 20.0 + s * 30.0

        y_bonding = y_mid + split_dy     # Lower energy
        y_antibonding = y_mid - split_dy # Higher energy

        # MO Levels (Bonding and Antibonding)
        # Bonding level (Cyan)
        painter.setPen(QPen(QColor(0, 220, 255), 2.4))
        painter.drawLine(x_mo, y_bonding, x_mo + bar_w, y_bonding)

        # Antibonding level (Red/Coral)
        painter.setPen(QPen(QColor(255, 80, 80), 2.4))
        painter.drawLine(x_mo, y_antibonding, x_mo + bar_w, y_antibonding)

        # Dashed correlation lines
        painter.setPen(QPen(QColor(70, 95, 130), 1.0, Qt.PenStyle.DashLine))
        # AO_A -> MO bonding & antibonding
        painter.drawLine(x_ao_A + bar_w, y_mid, x_mo, y_bonding)
        painter.drawLine(x_ao_A + bar_w, y_mid, x_mo, y_antibonding)
        # AO_B -> MO bonding & antibonding
        painter.drawLine(x_ao_B, y_mid, x_mo + bar_w, y_bonding)
        painter.drawLine(x_ao_B, y_mid, x_mo + bar_w, y_antibonding)

        # Labels for energy levels
        font_l = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font_l)
        painter.setPen(QColor(160, 180, 210))
        painter.drawText(int(x_ao_A - 5.0), int(y_mid - 6.0), "1s (A)")
        painter.drawText(int(x_ao_B - 5.0), int(y_mid - 6.0), "1s (B)")

        painter.setPen(QColor(0, 220, 255))
        painter.drawText(int(x_mo + bar_w + 6.0), int(y_bonding + 4.0), "σ (Liên kết)")

        painter.setPen(QColor(255, 80, 80))
        painter.drawText(int(x_mo + bar_w + 6.0), int(y_antibonding + 4.0), "σ* (Phản liên kết)")

        # Electron spin arrows in bonding level (H2 has 2 electrons in sigma)
        n_b = self._preset.n_bonding_electrons
        if n_b >= 2:
            font_arr = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font_arr)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(x_mo + 6.0), int(y_bonding - 4.0), "↿⇂")

    def _paint_buttons(self, painter: QPainter) -> None:
        button_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "mode_toggle": "Đổi σ ↔ σ* (Phản LK)" if not self._is_antibonding_view else "Về σ (Liên kết)",
            "r_dec": "R: −0.1 Å",
            "r_inc": "R: +0.1 Å",
            "r_eq": "R cân bằng",
            "set_sp": "Lai sp",
            "set_sp2": "Lai sp²",
            "set_sp3": "Lai sp³",
            "set_h2": "MO H₂",
            "reset": "🔄 Mặc định",
            "stamp": "📌 Đóng dấu bài học",
        }

        font_btn = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font_btn)

        for btn_key, rect in self._buttons.items():
            is_hover = (self._hovered_btn == btn_key)
            if btn_key == "stamp":
                border_col = QColor(0, 230, 180) if is_hover else QColor(0, 180, 140)
                fill_col = QColor(0, 140, 100, 180) if is_hover else QColor(0, 100, 75, 150)
                txt_col = QColor(255, 255, 255)
            elif btn_key == "reset":
                border_col = QColor(220, 100, 100) if is_hover else QColor(160, 70, 70)
                fill_col = QColor(140, 50, 50, 160) if is_hover else QColor(90, 35, 35, 140)
                txt_col = QColor(255, 230, 230)
            elif btn_key == "mode_toggle":
                border_col = QColor(240, 180, 50) if is_hover else QColor(190, 140, 30)
                fill_col = QColor(160, 110, 20, 160) if is_hover else QColor(100, 70, 15, 140)
                txt_col = QColor(255, 250, 220)
            else:
                border_col = QColor(80, 140, 220) if is_hover else QColor(45, 75, 120)
                fill_col = QColor(35, 60, 95, 180) if is_hover else QColor(22, 38, 62, 160)
                txt_col = QColor(225, 238, 255)

            painter.setPen(QPen(border_col, 1.0))
            painter.setBrush(QBrush(fill_col))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setPen(txt_col)
            label = button_labels.get(btn_key, btn_key)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # ------------------------------------------------------------------
    # Mouse Events
    # ------------------------------------------------------------------

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_hover = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if old_hover != self._hovered_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._handle_button_click(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _handle_button_click(self, btn_key: str) -> None:
        preset_keys = list(ORBITAL_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key == "preset_prev":
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "preset_next":
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "mode_toggle":
            self.toggle_bonding_mode()
        elif btn_key == "r_dec":
            self.set_distance_angstrom(self._distance_angstrom - 0.10)
        elif btn_key == "r_inc":
            self.set_distance_angstrom(self._distance_angstrom + 0.10)
        elif btn_key == "r_eq":
            self.set_distance_angstrom(self._preset.default_distance_angstrom)
        elif btn_key == "set_sp":
            self.set_preset("ETHYLENE_SP2_PI_BONDING_C2H4") # Contains sp2/pi
        elif btn_key == "set_sp2":
            self.set_preset("ETHYLENE_SP2_PI_BONDING_C2H4")
        elif btn_key == "set_sp3":
            self.set_preset("CARBON_SP3_METHANE_CH4_TETRAHEDRAL")
        elif btn_key == "set_h2":
            self.set_preset("HYDROGEN_MOLECULE_H2_SIGMA_BOND")
        elif btn_key == "reset":
            self.reset_defaults()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal for LessonBoard permanent vector rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "molecule_name": self._preset.molecule_name,
            "orbital_type": self._preset.orbital_type,
            "distance_angstrom": self._distance_angstrom,
            "overlap_integral_S": self.overlap_integral_S,
            "bond_order": self.bond_order,
            "is_antibonding": self._is_antibonding_view,
            "hybridization_str": self._preset.hybridization_str,
        }
        self.stamp_requested.emit(data)
