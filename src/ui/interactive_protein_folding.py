"""
AirTeach AI — Interactive Protein Folding, Secondary Structures & Ramachandran Dihedral Angle Explorer
Authentic polypeptide stereochemistry, Ramachandran phi-psi steric allowed energy basins,
Pauling-Corey hydrogen bonding networks, Gibbs free energy denaturation thermodynamics,
and touchless air gesture integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
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
    QStyleOptionGraphicsItem,
    QWidget,
)


@dataclass(frozen=True)
class ProteinPreset:
    code: str
    name_vi: str
    target_phi_deg: float
    target_psi_deg: float
    default_temp_c: float
    default_ph: float
    structure_type: str  # "alpha_helix", "beta_sheet", "collagen", "random_coil", "amyloid_fibril"
    residues: tuple[str, ...]
    description: str


PRESETS: dict[str, ProteinPreset] = {
    "ALPHA_HELIX_KERATIN_MYOGLOBIN": ProteinPreset(
        code="ALPHA_HELIX_KERATIN_MYOGLOBIN",
        name_vi="Xoắn Alpha (α-Helix Keratin/Myoglobin)",
        target_phi_deg=-57.0,
        target_psi_deg=-47.0,
        default_temp_c=37.0,
        default_ph=7.4,
        structure_type="alpha_helix",
        residues=("Ala", "Leu", "Glu", "Met", "Lys", "Ala", "Phe", "Gln", "Val", "Ile", "Arg", "Ser"),
        description="Xoắn phải điển hình với 3.6 gốc/vòng, bước xoắn 5.4Å. Liên kết hydro nội chuỗi i -> i+4 giữa C=O và N-H.",
    ),
    "BETA_SHEET_SILK_FIBROIN": ProteinPreset(
        code="BETA_SHEET_SILK_FIBROIN",
        name_vi="Phiến Beta đối song (Antiparallel β-Sheet)",
        target_phi_deg=-139.0,
        target_psi_deg=135.0,
        default_temp_c=25.0,
        default_ph=7.0,
        structure_type="beta_sheet",
        residues=("Gly", "Ala", "Gly", "Ser", "Gly", "Ala", "Gly", "Ala", "Gly", "Ser", "Val", "Ala"),
        description="Phiến gấp nếp phẳng kéo dài tơ tằm Fibroin. Liên kết hydro thẳng hàng 180° giữa các sợi peptide đối song.",
    ),
    "COLLAGEN_TRIPLE_HELIX": ProteinPreset(
        code="COLLAGEN_TRIPLE_HELIX",
        name_vi="Xoắn ba Collagen (Polyproline II / Gly-Pro-Hyp)",
        target_phi_deg=-78.0,
        target_psi_deg=149.0,
        default_temp_c=30.0,
        default_ph=7.2,
        structure_type="collagen",
        residues=("Gly", "Pro", "Hyp", "Gly", "Pro", "Hyp", "Gly", "Pro", "Hyp", "Gly", "Pro", "Ala"),
        description="Cấu hình Polyproline II xoắn trái tạo thành siêu xoắn 3 sợi vặn phải. Glycine nhỏ ở tâm trục ngăn va chạm steric.",
    ),
    "THERMAL_DENATURATION_UNFOLDED": ProteinPreset(
        code="THERMAL_DENATURATION_UNFOLDED",
        name_vi="Biến tính nhiệt / Cuộn ngẫu nhiên (Random Coil)",
        target_phi_deg=-90.0,
        target_psi_deg=60.0,
        default_temp_c=85.0,
        default_ph=7.0,
        structure_type="random_coil",
        residues=("Asp", "Glu", "Arg", "Lys", "His", "Pro", "Gly", "Asn", "Gln", "Thr", "Ser", "Tyr"),
        description="Nhiệt độ cao (>75°C) phá vỡ toàn bộ liên kết hydro yếu; chuỗi chuyển động hỗn loạn không cấu trúc bậc hai.",
    ),
    "PRION_MISFOLDING_AMYLOID": ProteinPreset(
        code="PRION_MISFOLDING_AMYLOID",
        name_vi="Gấp nếp sai Prion / Thể sợi Amyloid (Cross-β)",
        target_phi_deg=-125.0,
        target_psi_deg=120.0,
        default_temp_c=42.0,
        default_ph=5.0,
        structure_type="amyloid_fibril",
        residues=("Asn", "Gln", "Tyr", "Met", "His", "Leu", "Phe", "Trp", "Val", "Ile", "Ala", "Gly"),
        description="PrP-C biến đổi cấu hình từ α-helix sang giàu β-sheet không hòa tan tạo sợi Amyloid gây bệnh thoái hóa thần kinh.",
    ),
}


class InteractiveProteinFoldingItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Protein Secondary Structure folding,
    Ramachandran plot phi-psi dihedral conformation basins, and thermal/pH denaturation.
    """

    WIDTH = 660.0
    HEIGHT = 400.0

    stamp_requested = Signal()
    conformation_changed = Signal(float, float)  # phi, psi

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (160.0, 70.0),
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
        )

        # Simulation State
        self._active_preset_code: str = "ALPHA_HELIX_KERATIN_MYOGLOBIN"
        self._preset: ProteinPreset = PRESETS[self._active_preset_code]

        # Dihedral angles (degrees)
        self._phi_deg: float = self._preset.target_phi_deg
        self._psi_deg: float = self._preset.target_psi_deg

        # Environment conditions
        self._temp_c: float = self._preset.default_temp_c
        self._ph: float = self._preset.default_ph

        # 3D View rotation angles (degrees)
        self._rot_y: float = 25.0
        self._anim_phase: float = 0.0

        # Animation timer (30 FPS)
        self._is_animating: bool = True
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Sets up interactive button rects and Ramachandran plot canvas coordinates."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0

        # 5 Preset buttons
        preset_names = [
            ("ALPHA_HELIX_KERATIN_MYOGLOBIN", "α-Helix"),
            ("BETA_SHEET_SILK_FIBROIN", "β-Sheet"),
            ("COLLAGEN_TRIPLE_HELIX", "Collagen"),
            ("THERMAL_DENATURATION_UNFOLDED", "Biến tính"),
            ("PRION_MISFOLDING_AMYLOID", "Prion Amyloid"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 74.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 4.0

        # Control toggles
        r_play = QRectF(cur_x + 6.0, btn_y, 36.0, btn_h)
        self._btn_rects["toggle_anim"] = r_play

        r_t_down = QRectF(r_play.right() + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["temp_down"] = r_t_down

        r_t_up = QRectF(r_t_down.right() + 3.0, btn_y, 28.0, btn_h)
        self._btn_rects["temp_up"] = r_t_up

        r_ph_down = QRectF(r_t_up.right() + 4.0, btn_y, 32.0, btn_h)
        self._btn_rects["ph_down"] = r_ph_down

        r_ph_up = QRectF(r_ph_down.right() + 3.0, btn_y, 32.0, btn_h)
        self._btn_rects["ph_up"] = r_ph_up

        r_stamp = QRectF(r_ph_up.right() + 6.0, btn_y, 48.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        # Ramachandran plot canvas rect on the left
        self._rama_rect = QRectF(16.0, 78.0, 200.0, 200.0)
        self._btn_rects["ramachandran_canvas"] = self._rama_rect

        # 3D Ribbon viewport rect in the center
        self._ribbon_rect = QRectF(228.0, 78.0, 240.0, 200.0)

        # Diagnostics ledger on the right
        self._diag_rect = QRectF(480.0, 78.0, 166.0, 200.0)

    def cleanup(self) -> None:
        """Stops animation timer to prevent memory leaks."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    # -------------------------------------------------------------------------
    # Biophysical & Stereochemical Analytical Engine
    # -------------------------------------------------------------------------
    @property
    def phi_deg(self) -> float:
        return self._phi_deg

    @property
    def psi_deg(self) -> float:
        return self._psi_deg

    @property
    def temp_c(self) -> float:
        return self._temp_c

    @property
    def ph(self) -> float:
        return self._ph

    @property
    def active_preset(self) -> ProteinPreset:
        return self._preset

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def check_ramachandran_region(self, phi: float, psi: float) -> tuple[str, bool]:
        """
        Evaluates stereochemical Steric Allowed region in Ramachandran space.
        Returns (region_label, is_sterically_favored).
        """
        # Right-handed alpha-helix (alpha_R) core: phi in [-95, -35], psi in [-70, -15]
        if -95.0 <= phi <= -35.0 and -70.0 <= psi <= -15.0:
            return ("α-Helix (Thuận lợi)", True)
        if -160.0 <= phi <= -20.0 and -90.0 <= psi <= 20.0:
            return ("α-Helix (Cho phép)", True)

        # Beta-sheet core: phi in [-165, -90], psi in [105, 175] or [-180, -165]
        if -165.0 <= phi <= -90.0 and (105.0 <= psi <= 175.0 or -180.0 <= psi <= -165.0):
            return ("Phiến β (Thuận lợi)", True)

        # Collagen / Polyproline II: phi in [-90, -50], psi in [120, 175]
        if -90.0 <= phi <= -50.0 and 120.0 <= psi <= 175.0:
            return ("Collagen / PPII (Thuận lợi)", True)

        # Beta-sheet allowed region
        if -180.0 <= phi <= -50.0 and (80.0 <= psi <= 180.0 or -180.0 <= psi <= -150.0):
            return ("Phiến β (Cho phép)", True)

        # Left-handed alpha-helix (alpha_L): phi in [35, 95], psi in [15, 75]
        if 35.0 <= phi <= 95.0 and 15.0 <= psi <= 75.0:
            return ("α-Helix trái (Glycine)", True)

        return ("Vùng cấm Steric (Xung đột)", False)

    @property
    def folding_free_energy_kj(self) -> float:
        """
        Calculates Gibbs free energy of folding Delta G_fold (kJ/mol).
        Delta G = Delta H - T * Delta S + Delta G_electrostatic(pH) + Delta G_steric.
        Negative indicates thermodynamically stable folded state.
        """
        t_kelvin = self._temp_c + 273.15

        # Base enthalpy of H-bonding & hydrophobic collapse
        delta_h = -90.0  # kJ/mol (exothermic formation of H-bonds)
        delta_s = -0.27  # kJ/(mol*K) (conformational entropy loss upon folding, Tm ~ 60°C)

        # Conformational steric penalty if outside favored basins
        _, is_favored = self.check_ramachandran_region(self._phi_deg, self._psi_deg)
        steric_penalty = 0.0 if is_favored else 45.0

        # pH electrostatic penalty away from physiological isoelectric optimum ~7.4
        ph_penalty = 4.5 * ((self._ph - 7.4) ** 2)

        delta_g = (delta_h - t_kelvin * delta_s) + steric_penalty + ph_penalty
        return delta_g

    @property
    def fraction_folded(self) -> float:
        """
        Calculates fraction of folded protein f_N using two-state Boltzmann equilibrium:
        f_N = 1 / (1 + exp(Delta G / (R * T))).
        """
        r_gas = 0.008314  # kJ/(mol*K)
        t_kelvin = self._temp_c + 273.15
        d_g = self.folding_free_energy_kj

        exponent = max(-30.0, min(30.0, d_g / (r_gas * t_kelvin)))
        return 1.0 / (1.0 + math.exp(exponent))

    @property
    def hydrogen_bond_count(self) -> int:
        """Calculates active stabilizing hydrogen bonds in 12-residue segment."""
        max_bonds = 9 if self._preset.structure_type == "alpha_helix" else 8
        if self._preset.structure_type == "random_coil":
            max_bonds = 2
        return int(round(self.fraction_folded * max_bonds))

    # -------------------------------------------------------------------------
    # State Setters & Preset Switching
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined protein folding preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._phi_deg = self._preset.target_phi_deg
        self._psi_deg = self._preset.target_psi_deg
        self._temp_c = self._preset.default_temp_c
        self._ph = self._preset.default_ph
        self.conformation_changed.emit(self._phi_deg, self._psi_deg)
        self.update()

    def set_dihedral_angles(self, phi: float, psi: float) -> None:
        """Sets (phi, psi) dihedral angles with clamping to [-180, 180]."""
        self._phi_deg = max(-180.0, min(180.0, float(phi)))
        self._psi_deg = max(-180.0, min(180.0, float(psi)))
        self.conformation_changed.emit(self._phi_deg, self._psi_deg)
        self.update()

    def set_temperature_c(self, temp: float) -> None:
        """Sets temperature in Celsius clamped between 0 and 100°C."""
        self._temp_c = max(0.0, min(100.0, float(temp)))
        self.update()

    def set_ph(self, ph_val: float) -> None:
        """Sets pH clamped between 1.0 and 14.0."""
        self._ph = max(1.0, min(14.0, float(ph_val)))
        self.update()

    def toggle_animation(self) -> bool:
        """Toggles 3D rotation and thermal fluctuation animation."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def _on_animation_tick(self) -> None:
        """Updates continuous 3D rotation and thermal vibration phase."""
        if not self._is_animating:
            return
        self._rot_y = (self._rot_y + 1.2) % 360.0
        self._anim_phase += 0.08
        self.update()

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active Protein Folding & Ramachandran diagram to LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import BiologyTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = BiologyTools.create_protein_folding_diagram(
            center=center,
            preset_code=self._active_preset_code,
            phi_deg=self._phi_deg,
            psi_deg=self._psi_deg,
            temp_c=self._temp_c,
            ph=self._ph,
            fraction_folded=self.fraction_folded,
        )
        for itm in items:
            scene.addItem(itm)
        self.stamp_requested.emit()

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Ramachandran Plot Canvas direct drag/pinch
        if self._rama_rect.contains(local_pos):
            # Map canvas (x, y) to phi in [-180, 180] and psi in [-180, 180]
            frac_x = (local_pos.x() - self._rama_rect.left()) / self._rama_rect.width()
            frac_y = (self._rama_rect.bottom() - local_pos.y()) / self._rama_rect.height()
            new_phi = -180.0 + frac_x * 360.0
            new_psi = -180.0 + frac_y * 360.0
            self.set_dihedral_angles(new_phi, new_psi)
            return True

        # 3. Temperature steppers
        r_td = self._btn_rects.get("temp_down")
        if r_td and r_td.contains(local_pos):
            self.set_temperature_c(self._temp_c - 5.0)
            return True

        r_tu = self._btn_rects.get("temp_up")
        if r_tu and r_tu.contains(local_pos):
            self.set_temperature_c(self._temp_c + 5.0)
            return True

        # 4. pH steppers
        r_pd = self._btn_rects.get("ph_down")
        if r_pd and r_pd.contains(local_pos):
            self.set_ph(self._ph - 0.5)
            return True

        r_pu = self._btn_rects.get("ph_up")
        if r_pu and r_pu.contains(local_pos):
            self.set_ph(self._ph + 0.5)
            return True

        # 5. Play/Pause toggle
        r_play = self._btn_rects.get("toggle_anim")
        if r_play and r_play.contains(local_pos):
            self.toggle_animation()
            return True

        # 6. Stamp to LessonBoard
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
            return True

        return True

    # -------------------------------------------------------------------------
    # Rendering & Painting
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if painter is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header Banner
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 36.0), 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 26.0, self.WIDTH, 10.0))

        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(14.0, 0.0, 480.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "🧬 CẤU TRÚC BẬC HAI PROTEIN & ĐỒ THỊ RAMACHANDRAN (PAULING & COREY)",
        )

        # Preset Badge
        p_badge = QRectF(self.WIDTH - 156.0, 7.0, 142.0, 22.0)
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.setPen(QPen(QColor("#818cf8"), 1.0))
        painter.drawRoundedRect(p_badge, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#c7d2fe"))
        painter.drawText(p_badge, Qt.AlignmentFlag.AlignCenter, self._preset.name_vi[:22])

        # 3. Preset Buttons & Controls
        preset_items = [
            ("ALPHA_HELIX_KERATIN_MYOGLOBIN", "α-Helix"),
            ("BETA_SHEET_SILK_FIBROIN", "β-Sheet"),
            ("COLLAGEN_TRIPLE_HELIX", "Collagen"),
            ("THERMAL_DENATURATION_UNFOLDED", "Biến tính"),
            ("PRION_MISFOLDING_AMYLOID", "Prion Amyloid"),
        ]
        for p_code, p_lbl in preset_items:
            r = self._btn_rects[f"preset_{p_code}"]
            is_active = (p_code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#3b82f6") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#60a5fa") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, p_lbl)

        # Toggle Animation
        r_play = self._btn_rects["toggle_anim"]
        painter.setBrush(QBrush(QColor("#065f46") if self._is_animating else QColor("#334155")))
        painter.setPen(QPen(QColor("#10b981") if self._is_animating else QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_play, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, "▶ Chạy" if self._is_animating else "⏸ Dừng")

        # Temperature steppers
        r_td = self._btn_rects["temp_down"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(r_td, 3.0, 3.0)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_td, Qt.AlignmentFlag.AlignCenter, "T-")

        r_tu = self._btn_rects["temp_up"]
        painter.drawRoundedRect(r_tu, 3.0, 3.0)
        painter.setPen(QColor("#ef4444"))
        painter.drawText(r_tu, Qt.AlignmentFlag.AlignCenter, "T+")

        # pH steppers
        r_pd = self._btn_rects["ph_down"]
        painter.drawRoundedRect(r_pd, 3.0, 3.0)
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(r_pd, Qt.AlignmentFlag.AlignCenter, "pH-")

        r_pu = self._btn_rects["ph_up"]
        painter.drawRoundedRect(r_pu, 3.0, 3.0)
        painter.setPen(QColor("#a855f7"))
        painter.drawText(r_pu, Qt.AlignmentFlag.AlignCenter, "pH+")

        # Stamp Button
        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#831843")))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # ---------------------------------------------------------------------
        # 4. Left Panel: Interactive Ramachandran Plot Canvas
        # ---------------------------------------------------------------------
        rx, ry, rw, rh = self._rama_rect.x(), self._rama_rect.y(), self._rama_rect.width(), self._rama_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._rama_rect, 6.0, 6.0)

        # Draw grid lines & axes (origin at center)
        cx = rx + rw / 2.0
        cy = ry + rh / 2.0

        painter.setPen(QPen(QColor("#334155"), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rx, cy), QPointF(rx + rw, cy))  # psi = 0
        painter.drawLine(QPointF(cx, ry), QPointF(cx, ry + rh))  # phi = 0

        # Draw sterically allowed basin zones
        # 1. Beta-sheet zone (Top Left): phi in [-180, -45], psi in [60, 180]
        beta_x = rx
        beta_y = ry
        beta_w = (135.0 / 360.0) * rw
        beta_h = (120.0 / 360.0) * rh
        painter.setBrush(QBrush(QColor(16, 185, 129, 65)))  # Emerald
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(QRectF(beta_x, beta_y, beta_w, beta_h), 4.0, 4.0)

        # 2. Right-handed alpha-helix (alpha_R) zone (Bottom Left): phi in [-120, -35], psi in [-75, 10]
        alpha_x = rx + ((180.0 - 120.0) / 360.0) * rw
        alpha_y = ry + ((180.0 - 10.0) / 360.0) * rh
        alpha_w = (85.0 / 360.0) * rw
        alpha_h = (85.0 / 360.0) * rh
        painter.setBrush(QBrush(QColor(59, 130, 246, 75)))  # Blue
        painter.setPen(QPen(QColor("#3b82f6"), 1.0))
        painter.drawRoundedRect(QRectF(alpha_x, alpha_y, alpha_w, alpha_h), 4.0, 4.0)

        # 3. Left-handed alpha-helix (alpha_L) zone (Top Right): phi in [35, 95], psi in [15, 75]
        al_x = rx + ((180.0 + 35.0) / 360.0) * rw
        al_y = ry + ((180.0 - 75.0) / 360.0) * rh
        al_w = (60.0 / 360.0) * rw
        al_h = (60.0 / 360.0) * rh
        painter.setBrush(QBrush(QColor(168, 85, 247, 50)))  # Purple
        painter.setPen(QPen(QColor("#a855f7"), 1.0))
        painter.drawRoundedRect(QRectF(al_x, al_y, al_w, al_h), 4.0, 4.0)

        # Zone labels
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#6ee7b7"))
        painter.drawText(QPointF(beta_x + 6.0, beta_y + 14.0), "Phiến β")

        painter.setPen(QColor("#93c5fd"))
        painter.drawText(QPointF(alpha_x + 6.0, alpha_y + 14.0), "Xoắn α_R")

        painter.setPen(QColor("#d8b4fe"))
        painter.drawText(QPointF(al_x + 4.0, al_y + 12.0), "α_L")

        # Current (phi, psi) probe marker
        px = rx + ((self._phi_deg + 180.0) / 360.0) * rw
        py = ry + ((180.0 - self._psi_deg) / 360.0) * rh

        # Glowing halo
        painter.setBrush(QBrush(QColor(244, 63, 94, 60)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(px, py), 9.0, 9.0)

        painter.setBrush(QBrush(QColor("#f43f5e")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(QPointF(px, py), 4.5, 4.5)

        # Axis Titles & Readout
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(rx + 8.0, ry + rh - 6.0), f"φ = {self._phi_deg:+.1f}°")
        painter.drawText(QPointF(rx + rw - 65.0, ry + 14.0), f"ψ = {self._psi_deg:+.1f}°")

        # ---------------------------------------------------------------------
        # 5. Center Panel: 3D Vector Polypeptide Ribbon Simulation
        # ---------------------------------------------------------------------
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._ribbon_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(self._ribbon_rect.x() + 8.0, self._ribbon_rect.y() + 16.0), "MÔ HÌNH VECTƠ DÂY CUỘN 3D")

        # Compute 3D projected coordinates for 12 residues
        cx_rib = self._ribbon_rect.center().x()
        cy_rib = self._ribbon_rect.center().y() + 6.0

        n_res = len(self._preset.residues)
        coords_2d: list[QPointF] = []
        rad_y = math.radians(self._rot_y)

        # Structure-dependent geometric projection
        st = self._preset.structure_type
        fn = self.fraction_folded

        for i in range(n_res):
            frac = i / (n_res - 1)
            # Add thermal vibration jitter proportional to (1 - fn)
            jitter = (1.0 - fn) * 12.0 * math.sin(self._anim_phase + i * 1.5)

            if st == "alpha_helix":
                # Cylindrical helix: radius R=28, pitch dz=12
                angle = i * 1.745 + (self._phi_deg + 57.0) * 0.02
                x3d = 28.0 * math.cos(angle)
                y3d = (i - (n_res / 2.0)) * 13.0 + jitter
                z3d = 28.0 * math.sin(angle)
            elif st == "beta_sheet":
                # Pleated zigzag ribbon
                x3d = (i - (n_res / 2.0)) * 16.0
                y3d = (-1.0 if i % 2 == 0 else 1.0) * 22.0 + jitter
                z3d = (i % 2) * 10.0
            elif st == "collagen":
                # Narrow, elongated polyproline helix
                angle = i * 2.094  # 3 residues per turn
                x3d = 16.0 * math.cos(angle)
                y3d = (i - (n_res / 2.0)) * 14.5 + jitter
                z3d = 16.0 * math.sin(angle)
            else:
                # Random coil or misfolded amyloid
                angle = i * 0.8 + self._anim_phase * 0.5
                r = 25.0 + 15.0 * math.sin(i * 1.2)
                x3d = r * math.cos(angle)
                y3d = (i - (n_res / 2.0)) * 11.0 + jitter * 1.5
                z3d = r * math.sin(angle)

            # 3D Rotation about Y axis
            rot_x = x3d * math.cos(rad_y) + z3d * math.sin(rad_y)
            rot_z = -x3d * math.sin(rad_y) + z3d * math.cos(rad_y)
            rot_y_coord = y3d

            # Perspective projection
            scale = 220.0 / (220.0 + rot_z)
            coords_2d.append(QPointF(cx_rib + rot_x * scale, cy_rib + rot_y_coord * scale))

        # Render Hydrogen Bond dashed lines (between i and i+4 in alpha-helix)
        if fn > 0.3:
            painter.setPen(QPen(QColor(251, 191, 36, int(fn * 200)), 1.2, Qt.PenStyle.DashLine))
            step_hb = 4 if st == "alpha_helix" else 2
            for i in range(len(coords_2d) - step_hb):
                painter.drawLine(coords_2d[i], coords_2d[i + step_hb])

        # Render Main-chain Polypeptide Backbone Ribbon
        ribbon_path = QPainterPath()
        if coords_2d:
            ribbon_path.moveTo(coords_2d[0])
            for pt in coords_2d[1:]:
                ribbon_path.lineTo(pt)

        painter.setPen(QPen(QColor("#38bdf8"), 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(ribbon_path)

        # Render Residue Amino Acid Nodes & Badges
        for i, pt in enumerate(coords_2d):
            r_name = self._preset.residues[i]
            # Node color based on polarity/charge
            node_col = QColor("#3b82f6") if i % 2 == 0 else QColor("#10b981")
            painter.setBrush(QBrush(node_col))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawEllipse(pt, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#e2e8f0"))
            painter.drawText(QPointF(pt.x() + 7.0, pt.y() + 3.0), f"{r_name}{i+1}")

        # ---------------------------------------------------------------------
        # 6. Right Panel: Diagnostic Bioenergetics & Folding Ledger
        # ---------------------------------------------------------------------
        dx, dy, dw, dh = self._diag_rect.x(), self._diag_rect.y(), self._diag_rect.width(), self._diag_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._diag_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(dx + 8.0, dy + 16.0), "NĂNG LƯỢNG HỌC & ĐỘNG HỌC")

        reg_name, is_fav = self.check_ramachandran_region(self._phi_deg, self._psi_deg)
        d_g = self.folding_free_energy_kj
        h_bonds = self.hydrogen_bond_count

        lines = [
            (f"• Góc nhị diện φ: {self._phi_deg:+.1f}°", QColor("#38bdf8")),
            (f"• Góc nhị diện ψ: {self._psi_deg:+.1f}°", QColor("#818cf8")),
            (f"• Nhiệt độ T: {self._temp_c:.1f}°C", QColor("#ef4444") if self._temp_c > 65.0 else QColor("#38bdf8")),
            (f"• Độ pH môi trường: {self._ph:.1f}", QColor("#10b981") if 6.8 <= self._ph <= 7.8 else QColor("#fbbf24")),
            (f"• Năng lượng tự do ΔG: {d_g:.1f} kJ/mol", QColor("#10b981") if d_g < 0 else QColor("#ef4444")),
            (f"• Tỷ lệ cuộn gập f_N: {fn * 100:.1f}%", QColor("#34d399") if fn > 0.7 else QColor("#f43f5e")),
            (f"• LK Hydro nội phân tử: {h_bonds} lk", QColor("#fbbf24")),
            (f"• Phân loại: {reg_name[:20]}", QColor("#a78bfa")),
        ]

        ly = dy + 32.0
        for l_txt, l_col in lines:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(l_col)
            painter.drawText(QPointF(dx + 8.0, ly), l_txt)
            ly += 17.0

        # Progress bar of folded state f_N
        bar_r = QRectF(dx + 8.0, dy + dh - 26.0, dw - 16.0, 14.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(bar_r, 3.0, 3.0)

        fill_w = max(0.0, (dw - 16.0) * fn)
        fill_r = QRectF(dx + 8.0, dy + dh - 26.0, fill_w, 14.0)
        painter.setBrush(QBrush(QColor("#10b981") if fn > 0.5 else QColor("#f43f5e")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(fill_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(bar_r, Qt.AlignmentFlag.AlignCenter, f"Ổn định bậc 2: {fn * 100:.0f}%")

        # ---------------------------------------------------------------------
        # 7. Bottom Panel: Educational Principles & Anfinsen Thermodynamic Card
        # ---------------------------------------------------------------------
        bot_y = self._rama_rect.bottom() + 10.0
        bot_h = self.HEIGHT - bot_y - 10.0
        bot_rect = QRectF(14.0, bot_y, self.WIDTH - 28.0, bot_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(bot_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(22.0, bot_y + 16.0), "NGUYÊN LÝ STERICH RAMACHANDRAN & ĐỊNH ĐỀ ANFINSEN (1963):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        b_desc = (
            f"• Liên kết peptide phẳng trans (ω ≈ 180°). Cấu hình chuỗi được xác định bởi 2 góc nhị diện phi (N-Cα) và psi (Cα-C).\n"
            f"• Đồ thị Ramachandran xác định các vùng năng lượng steric cực tiểu không va chạm nguyên tử (Pauli exclusion).\n"
            f"• Đặc tính mẫu: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, bot_y + 22.0, bot_rect.width() - 20.0, 58.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            b_desc,
        )
