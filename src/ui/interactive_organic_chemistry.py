"""
AirTeach AI - Interactive Chemistry Organic Hydrocarbon & Functional Group Isomer Simulator
============================================================================================
Provides an authentic interactive 2D organic chemistry and structural isomerism simulator:
  - Isomerism categories:
      1. Constitutional Isomers (Đồng phân cấu tạo):
           - Functional group isomers (Đồng phân nhóm chức): Ancol vs Ete (C₂H₆O), Axit vs Este (C₄H₈O₂), Andehit vs Xeton (C₃H₆O).
           - Position isomers (Đồng phân vị trí): But-1-en vs But-2-en (C₄H₈).
      2. Stereoisomers (Đồng phân lập thể):
           - Geometric isomers (Đồng phân hình học cis / trans quanh liên kết đôi C=C): cis-But-2-en vs trans-But-2-en.
  - Zero Mock Data:
      - Exact atomic masses (C: 12.011, H: 1.008, O: 15.999, N: 14.007, Cl: 35.453).
      - Degree of unsaturation (Độ bất bão hòa) formula: k = (2C + 2 - H + N - X) / 2.
      - Valence validation engine (C: 4, H: 1, O: 2, N: 3, Halogen: 1).
      - Physical constants: boiling point (°C), melting point (°C), dipole moment (Debye), intermolecular forces (H-bonding).
  - Visualization:
      - Lewis structural formula with explicit valence bonds and lone pair dots (··).
      - Skeletal formula (Khung cacbon gấp khúc).
      - Interactive comparison card highlighting why physical and chemical properties diverge.
      - Vector diagram stamping onto LessonBoard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Callable

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

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Standard atomic weights (g/mol) from IUPAC
ATOMIC_WEIGHTS: dict[str, float] = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "Cl": 35.453,
    "Br": 79.904,
}

# Standard valences
STANDARD_VALENCES: dict[str, int] = {
    "C": 4,
    "H": 1,
    "O": 2,
    "N": 3,
    "Cl": 1,
    "Br": 1,
}

# CPK Atomic color palette
CPK_COLORS: dict[str, tuple[QColor, QColor]] = {
    "C": (QColor(51, 65, 85), QColor(248, 250, 252)),     # Slate-700 / White text
    "H": (QColor(226, 232, 240), QColor(15, 23, 42)),    # Slate-200 / Dark text
    "O": (QColor(239, 68, 68), QColor(255, 255, 255)),    # Red-500 / White text
    "N": (QColor(59, 130, 246), QColor(255, 255, 255)),   # Blue-500 / White text
    "Cl": (QColor(16, 185, 129), QColor(255, 255, 255)),  # Emerald-500 / White text
    "Br": (QColor(180, 83, 9), QColor(255, 255, 255)),    # Amber-700 / White text
}


@dataclass
class AtomNode:
    """Represents a 2D atom node in the organic molecule."""
    id: int
    symbol: str
    x: float
    y: float
    valence: int = 4
    formal_charge: int = 0
    is_dragged: bool = False

    def __post_init__(self):
        if self.symbol in STANDARD_VALENCES:
            self.valence = STANDARD_VALENCES[self.symbol]


@dataclass
class ChemicalBond:
    """Represents a covalent bond between two atoms."""
    atom1_id: int
    atom2_id: int
    order: int = 1  # 1: single, 2: double, 3: triple
    is_hydrogen_bond: bool = False


@dataclass
class MoleculeStructure:
    """Complete organic molecule structural and physical description."""
    name_vi: str
    iupac_name: str
    formula: str
    class_vi: str
    boiling_point_c: float
    melting_point_c: float
    dipole_moment_debye: float
    intermolecular_forces: str
    physical_state_vi: str
    chemical_reactions_vi: str
    functional_groups_vi: list[str]
    atoms: list[AtomNode]
    bonds: list[ChemicalBond]
    skeletal_path_segments: list[tuple[int, int]] = field(default_factory=list)

    @property
    def molecular_mass(self) -> float:
        """Exact molecular weight in g/mol."""
        return sum(ATOMIC_WEIGHTS.get(a.symbol, 0.0) for a in self.atoms)

    @property
    def degree_of_unsaturation(self) -> float:
        """Degree of unsaturation k = (2C + 2 - H + N - X) / 2."""
        c = sum(1 for a in self.atoms if a.symbol == "C")
        h = sum(1 for a in self.atoms if a.symbol == "H")
        n = sum(1 for a in self.atoms if a.symbol == "N")
        x = sum(1 for a in self.atoms if a.symbol in ("Cl", "Br"))
        return (2 * c + 2 - h + n - x) / 2.0

    def get_atom(self, atom_id: int) -> Optional[AtomNode]:
        for a in self.atoms:
            if a.id == atom_id:
                return a
        return None

    def validate_valences(self) -> dict[int, tuple[int, int, bool]]:
        """
        Validate valences for each atom.
        Returns: {atom_id: (actual_bonds, target_valence, is_valid)}
        """
        bond_counts: dict[int, int] = {a.id: 0 for a in self.atoms}
        for b in self.bonds:
            if not b.is_hydrogen_bond:
                if b.atom1_id in bond_counts:
                    bond_counts[b.atom1_id] += b.order
                if b.atom2_id in bond_counts:
                    bond_counts[b.atom2_id] += b.order

        results: dict[int, tuple[int, int, bool]] = {}
        for a in self.atoms:
            actual = bond_counts[a.id]
            target = a.valence
            results[a.id] = (actual, target, actual == target)
        return results


@dataclass
class IsomerPreset:
    """Pre-configured authentic curriculum organic isomer pair."""
    code: str
    title_vi: str
    isomer_type_vi: str
    formula: str
    molecule_a: MoleculeStructure
    molecule_b: MoleculeStructure
    pedagogical_insight_vi: str


def _create_c2h6o_preset() -> IsomerPreset:
    """C2H6O: Ethanol (Alcohol) vs Dimethyl Ether (Ether) - Functional Group Isomers."""
    # Ethanol: CH3-CH2-OH
    # C1 at (170, 240), C2 at (240, 240), O3 at (310, 240), H_O at (350, 210)
    # Hydrogens around C1 and C2
    eth_atoms = [
        AtomNode(1, "C", 170.0, 240.0),
        AtomNode(2, "C", 240.0, 240.0),
        AtomNode(3, "O", 310.0, 240.0),
        AtomNode(4, "H", 355.0, 210.0),  # H on O (active H)
        # C1 Hydrogens
        AtomNode(5, "H", 170.0, 180.0),
        AtomNode(6, "H", 170.0, 300.0),
        AtomNode(7, "H", 115.0, 240.0),
        # C2 Hydrogens
        AtomNode(8, "H", 240.0, 180.0),
        AtomNode(9, "H", 240.0, 300.0),
    ]
    eth_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=1),
        ChemicalBond(3, 4, order=1),
        ChemicalBond(1, 5, order=1),
        ChemicalBond(1, 6, order=1),
        ChemicalBond(1, 7, order=1),
        ChemicalBond(2, 8, order=1),
        ChemicalBond(2, 9, order=1),
    ]
    mol_ethanol = MoleculeStructure(
        name_vi="Etanol (Ancol etylic)",
        iupac_name="Ethanol",
        formula="C₂H₆O",
        class_vi="Ancol no, đơn chức (Alcohol)",
        boiling_point_c=78.37,
        melting_point_c=-114.1,
        dipole_moment_debye=1.69,
        intermolecular_forces="Liên kết Hiđro liên phân tử rất bền (-O-H···O-)",
        physical_state_vi="Chất lỏng không màu, tan vô hạn trong nước, mùi rượu nồng nhẹ",
        chemical_reactions_vi="Tác dụng với Na giải phóng H₂: 2C₂H₅OH + 2Na → 2C₂H₅ONa + H₂↑; Phản ứng este hóa",
        functional_groups_vi=["-OH (Hydroxyl)"],
        atoms=eth_atoms,
        bonds=eth_bonds,
        skeletal_path_segments=[(1, 2), (2, 3)],
    )

    # Dimethyl ether: CH3-O-CH3
    # C1 at (160, 240), O2 at (240, 210), C3 at (320, 240)
    dme_atoms = [
        AtomNode(1, "C", 160.0, 240.0),
        AtomNode(2, "O", 240.0, 200.0),
        AtomNode(3, "C", 320.0, 240.0),
        # C1 Hydrogens
        AtomNode(4, "H", 110.0, 240.0),
        AtomNode(5, "H", 160.0, 180.0),
        AtomNode(6, "H", 160.0, 300.0),
        # C3 Hydrogens
        AtomNode(7, "H", 370.0, 240.0),
        AtomNode(8, "H", 320.0, 180.0),
        AtomNode(9, "H", 320.0, 300.0),
    ]
    dme_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=1),
        ChemicalBond(1, 4, order=1),
        ChemicalBond(1, 5, order=1),
        ChemicalBond(1, 6, order=1),
        ChemicalBond(3, 7, order=1),
        ChemicalBond(3, 8, order=1),
        ChemicalBond(3, 9, order=1),
    ]
    mol_dme = MoleculeStructure(
        name_vi="Đimetyl ete (Dimethyl ether)",
        iupac_name="Methoxymethane",
        formula="C₂H₆O",
        class_vi="Ete đơn giản (Ether)",
        boiling_point_c=-24.0,
        melting_point_c=-141.5,
        dipole_moment_debye=1.30,
        intermolecular_forces="Lực hút lưỡng cực Van der Waals yếu; KHÔNG CÓ liên kết hiđro liên phân tử",
        physical_state_vi="Chất khí ở điều kiện thường, ít tan trong nước, dễ bay hơi và dễ cháy",
        chemical_reactions_vi="Không phản ứng với kim loại Na; Trơ về mặt hóa học so với ancol",
        functional_groups_vi=["-O- (Ete / Ether linkage)"],
        atoms=dme_atoms,
        bonds=dme_bonds,
        skeletal_path_segments=[(1, 2), (2, 3)],
    )

    return IsomerPreset(
        code="C2H6O_ALCOHOL_ETHER",
        title_vi="C₂H₆O: Ancol vs Ete (Đồng phân nhóm chức)",
        isomer_type_vi="Đồng phân nhóm chức (Functional Group Isomerism)",
        formula="C₂H₆O",
        molecule_a=mol_ethanol,
        molecule_b=mol_dme,
        pedagogical_insight_vi=(
            "Cùng CTHH C₂H₆O và phân tử khối 46.07 g/mol, nhưng Etanol có nhóm -OH tạo liên kết hiđro "
            "liên phân tử bền vững làm nhiệt độ sôi tăng vọt lên +78.4°C (chất lỏng), trong khi Đimetyl ete "
            "không có liên kết hiđro nên sôi ở -24.0°C (chất khí). Chênh lệch nhiệt độ sôi lên tới 102.4°C!"
        ),
    )


def _create_c4h8_position_preset() -> IsomerPreset:
    """C4H8: But-1-ene vs But-2-ene - Position Isomers (Double Bond Position)."""
    # But-1-ene: CH2=CH-CH2-CH3
    b1_atoms = [
        AtomNode(1, "C", 140.0, 240.0),
        AtomNode(2, "C", 205.0, 240.0),
        AtomNode(3, "C", 270.0, 240.0),
        AtomNode(4, "C", 335.0, 240.0),
        # C1 H's
        AtomNode(5, "H", 95.0, 210.0),
        AtomNode(6, "H", 95.0, 270.0),
        # C2 H
        AtomNode(7, "H", 205.0, 180.0),
        # C3 H's
        AtomNode(8, "H", 270.0, 180.0),
        AtomNode(9, "H", 270.0, 300.0),
        # C4 H's
        AtomNode(10, "H", 380.0, 240.0),
        AtomNode(11, "H", 335.0, 180.0),
        AtomNode(12, "H", 335.0, 300.0),
    ]
    b1_bonds = [
        ChemicalBond(1, 2, order=2),  # C1=C2
        ChemicalBond(2, 3, order=1),
        ChemicalBond(3, 4, order=1),
        ChemicalBond(1, 5, order=1),
        ChemicalBond(1, 6, order=1),
        ChemicalBond(2, 7, order=1),
        ChemicalBond(3, 8, order=1),
        ChemicalBond(3, 9, order=1),
        ChemicalBond(4, 10, order=1),
        ChemicalBond(4, 11, order=1),
        ChemicalBond(4, 12, order=1),
    ]
    mol_b1 = MoleculeStructure(
        name_vi="But-1-en (1-Butene)",
        iupac_name="But-1-ene",
        formula="C₄H₈",
        class_vi="Anken mạch thẳng (Alkene)",
        boiling_point_c=-6.3,
        melting_point_c=-185.3,
        dipole_moment_debye=0.36,
        intermolecular_forces="Lực khuếch tán London Van der Waals",
        physical_state_vi="Khí không màu, mùi hydrocacbon nhẹ, không tan trong nước",
        chemical_reactions_vi="Cộng bất đối xứng theo quy tắc Mac-côp-nhi-côp: cộng H₂O cho Butan-2-ol (sản phẩm chính)",
        functional_groups_vi=["C=C (Liên kết đôi đầu mạch)"],
        atoms=b1_atoms,
        bonds=b1_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (3, 4)],
    )

    # But-2-ene: CH3-CH=CH-CH3
    b2_atoms = [
        AtomNode(1, "C", 140.0, 240.0),
        AtomNode(2, "C", 205.0, 240.0),
        AtomNode(3, "C", 270.0, 240.0),
        AtomNode(4, "C", 335.0, 240.0),
        # C1 H's
        AtomNode(5, "H", 95.0, 240.0),
        AtomNode(6, "H", 140.0, 180.0),
        AtomNode(7, "H", 140.0, 300.0),
        # C2 H
        AtomNode(8, "H", 205.0, 180.0),
        # C3 H
        AtomNode(9, "H", 270.0, 300.0),
        # C4 H's
        AtomNode(10, "H", 380.0, 240.0),
        AtomNode(11, "H", 335.0, 180.0),
        AtomNode(12, "H", 335.0, 300.0),
    ]
    b2_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=2),  # C2=C3
        ChemicalBond(3, 4, order=1),
        ChemicalBond(1, 5, order=1),
        ChemicalBond(1, 6, order=1),
        ChemicalBond(1, 7, order=1),
        ChemicalBond(2, 8, order=1),
        ChemicalBond(3, 9, order=1),
        ChemicalBond(4, 10, order=1),
        ChemicalBond(4, 11, order=1),
        ChemicalBond(4, 12, order=1),
    ]
    mol_b2 = MoleculeStructure(
        name_vi="But-2-en (2-Butene)",
        iupac_name="But-2-ene",
        formula="C₄H₈",
        class_vi="Anken đối xứng (Symmetric Alkene)",
        boiling_point_c=1.0,
        melting_point_c=-138.9,
        dipole_moment_debye=0.15,
        intermolecular_forces="Lực phân tán London giữa các phân tử",
        physical_state_vi="Khí hóa lỏng gần 0°C, bền nhiệt hơn But-1-en do quy tắc Zaitsev",
        chemical_reactions_vi="Cộng đối xứng: cộng H₂O chỉ tạo ra duy nhất 1 sản phẩm ancol là Butan-2-ol; có đồng phân hình học cis/trans",
        functional_groups_vi=["C=C (Liên kết đôi giữa mạch)"],
        atoms=b2_atoms,
        bonds=b2_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (3, 4)],
    )

    return IsomerPreset(
        code="C4H8_POSITION",
        title_vi="C₄H₈: Vị trí liên kết đôi (But-1-en vs But-2-en)",
        isomer_type_vi="Đồng phân vị trí nhóm chức / liên kết bội (Position Isomerism)",
        formula="C₄H₈",
        molecule_a=mol_b1,
        molecule_b=mol_b2,
        pedagogical_insight_vi=(
            "Vị trí liên kết đôi C=C thay đổi từ đầu mạch (C₁=C₂) sang giữa mạch (C₂=C₃). "
            "But-2-en đối xứng bền hơn về mặt nhiệt động học, chỉ tạo 1 sản phẩm khi cộng HBr/H₂O, "
            "đồng thời xuất hiện tính chất lập thể hình học (cis/trans)."
        ),
    )


def _create_c4h8_cistrans_preset() -> IsomerPreset:
    """C4H8: cis-But-2-ene vs trans-But-2-ene - Geometric Stereoisomers."""
    # cis-But-2-ene: Both -CH3 on top (y < 240), both -H on bottom (y > 240)
    cis_atoms = [
        AtomNode(1, "C", 160.0, 180.0),  # Methyl 1
        AtomNode(2, "C", 205.0, 240.0),  # C2
        AtomNode(3, "C", 275.0, 240.0),  # C3
        AtomNode(4, "C", 320.0, 180.0),  # Methyl 4
        # Hydrogens on C2 and C3
        AtomNode(5, "H", 185.0, 290.0),
        AtomNode(6, "H", 295.0, 290.0),
        # C1 H's
        AtomNode(7, "H", 115.0, 175.0),
        AtomNode(8, "H", 160.0, 130.0),
        AtomNode(9, "H", 195.0, 175.0),
        # C4 H's
        AtomNode(10, "H", 365.0, 175.0),
        AtomNode(11, "H", 320.0, 130.0),
        AtomNode(12, "H", 285.0, 175.0),
    ]
    cis_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=2),  # C=C
        ChemicalBond(3, 4, order=1),
        ChemicalBond(2, 5, order=1),
        ChemicalBond(3, 6, order=1),
        ChemicalBond(1, 7, order=1),
        ChemicalBond(1, 8, order=1),
        ChemicalBond(1, 9, order=1),
        ChemicalBond(4, 10, order=1),
        ChemicalBond(4, 11, order=1),
        ChemicalBond(4, 12, order=1),
    ]
    mol_cis = MoleculeStructure(
        name_vi="cis-But-2-en ((Z)-2-Butene)",
        iupac_name="(2Z)-But-2-ene",
        formula="C₄H₈",
        class_vi="Đồng phân hình học cis / Z (Stereoisomer)",
        boiling_point_c=3.7,
        melting_point_c=-138.9,
        dipole_moment_debye=0.33,
        intermolecular_forces="Hút lưỡng cực vĩnh viễn (μ = 0.33 D) làm tăng nhiệt độ sôi",
        physical_state_vi="Khí dễ hóa lỏng, hai nhóm metyl cùng phía mặt phẳng liên kết đôi",
        chemical_reactions_vi="Phản ứng cộng halogen Br₂ cho hỗn hợp đối quang threo-2,3-đibrombutan",
        functional_groups_vi=["C=C (cis / Z)"],
        atoms=cis_atoms,
        bonds=cis_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (3, 4)],
    )

    # trans-But-2-ene: -CH3 (1) at top-left, -CH3 (4) at bottom-right
    trans_atoms = [
        AtomNode(1, "C", 160.0, 180.0),  # Methyl 1 (top)
        AtomNode(2, "C", 205.0, 240.0),  # C2
        AtomNode(3, "C", 275.0, 240.0),  # C3
        AtomNode(4, "C", 320.0, 300.0),  # Methyl 4 (bottom)
        # H's on C2 (bottom) and C3 (top)
        AtomNode(5, "H", 185.0, 290.0),
        AtomNode(6, "H", 295.0, 190.0),
        # C1 H's
        AtomNode(7, "H", 115.0, 175.0),
        AtomNode(8, "H", 160.0, 130.0),
        AtomNode(9, "H", 195.0, 175.0),
        # C4 H's
        AtomNode(10, "H", 365.0, 305.0),
        AtomNode(11, "H", 320.0, 350.0),
        AtomNode(12, "H", 285.0, 305.0),
    ]
    trans_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=2),  # C=C
        ChemicalBond(3, 4, order=1),
        ChemicalBond(2, 5, order=1),
        ChemicalBond(3, 6, order=1),
        ChemicalBond(1, 7, order=1),
        ChemicalBond(1, 8, order=1),
        ChemicalBond(1, 9, order=1),
        ChemicalBond(4, 10, order=1),
        ChemicalBond(4, 11, order=1),
        ChemicalBond(4, 12, order=1),
    ]
    mol_trans = MoleculeStructure(
        name_vi="trans-But-2-en ((E)-2-Butene)",
        iupac_name="(2E)-But-2-ene",
        formula="C₄H₈",
        class_vi="Đồng phân hình học trans / E (Stereoisomer)",
        boiling_point_c=0.9,
        melting_point_c=-105.5,
        dipole_moment_debye=0.00,
        intermolecular_forces="Momen lưỡng cực triệt tiêu hoàn toàn (μ = 0 D) do đối xứng tâm",
        physical_state_vi="Khí, hai nhóm metyl khác phía; tinh thể xếp chặt hơn nên nhiệt độ nóng chảy cao hơn (-105.5°C vs -138.9°C)",
        chemical_reactions_vi="Phản ứng cộng Br₂ cho hợp chất meso-2,3-đibrombutan không quang hoạt",
        functional_groups_vi=["C=C (trans / E)"],
        atoms=trans_atoms,
        bonds=trans_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (3, 4)],
    )

    return IsomerPreset(
        code="C4H8_CISTRANS",
        title_vi="C₄H₈: Đồng phân hình học (cis vs trans But-2-en)",
        isomer_type_vi="Đồng phân lập thể hình học (Geometric Stereoisomerism)",
        formula="C₄H₈",
        molecule_a=mol_cis,
        molecule_b=mol_trans,
        pedagogical_insight_vi=(
            "Liên kết đôi π cản trở sự quay tự do quanh trục C=C. Dạng cis có 2 nhóm -CH₃ cùng phía, "
            "momen lưỡng cực tổng hợp μ = 0.33 D > 0 khiến nhiệt độ sôi cao hơn (3.7°C > 0.9°C). "
            "Ngược lại, dạng trans đối xứng tâm nên μ = 0, mạng tinh thể xếp khít chặt hơn khiến "
            "nhiệt độ nóng chảy cao hơn dạng cis tới 33.4°C (-105.5°C so với -138.9°C)!"
        ),
    )


def _create_c3h6o_preset() -> IsomerPreset:
    """C3H6O: Propanal (Aldehyde) vs Acetone (Ketone) - Functional Group Isomers."""
    # Propanal: CH3-CH2-CH=O
    prop_atoms = [
        AtomNode(1, "C", 140.0, 240.0),
        AtomNode(2, "C", 210.0, 240.0),
        AtomNode(3, "C", 280.0, 240.0),
        AtomNode(4, "O", 330.0, 195.0),  # Carbonyl O
        AtomNode(5, "H", 325.0, 285.0),  # Formyl H
        # C1 H's
        AtomNode(6, "H", 95.0, 240.0),
        AtomNode(7, "H", 140.0, 180.0),
        AtomNode(8, "H", 140.0, 300.0),
        # C2 H's
        AtomNode(9, "H", 210.0, 180.0),
        AtomNode(10, "H", 210.0, 300.0),
    ]
    prop_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=1),
        ChemicalBond(3, 4, order=2),  # C=O
        ChemicalBond(3, 5, order=1),  # C-H formyl
        ChemicalBond(1, 6, order=1),
        ChemicalBond(1, 7, order=1),
        ChemicalBond(1, 8, order=1),
        ChemicalBond(2, 9, order=1),
        ChemicalBond(2, 10, order=1),
    ]
    mol_prop = MoleculeStructure(
        name_vi="Propanal (Andehit propionic)",
        iupac_name="Propanal",
        formula="C₃H₆O",
        class_vi="Andehit no, đơn chức (Aldehyde)",
        boiling_point_c=48.0,
        melting_point_c=-81.0,
        dipole_moment_debye=2.52,
        intermolecular_forces="Hút lưỡng cực - lưỡng cực mạnh từ nhóm cacbonyl C=O",
        physical_state_vi="Chất lỏng không màu, mùi hăng cay đặc trưng của andehit",
        chemical_reactions_vi="Có tính khử mạnh: tráng bạc với thuốc thử Tollens (tạo 2Ag↓) và Cu(OH)₂ (tạo Cu₂O đỏ gạch)",
        functional_groups_vi=["-CHO (Nhóm Formyl / Andehit)"],
        atoms=prop_atoms,
        bonds=prop_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (3, 4)],
    )

    # Acetone: CH3-C(=O)-CH3
    ace_atoms = [
        AtomNode(1, "C", 140.0, 260.0),
        AtomNode(2, "C", 210.0, 240.0),
        AtomNode(3, "C", 280.0, 260.0),
        AtomNode(4, "O", 210.0, 175.0),  # Ketone C=O
        # C1 H's
        AtomNode(5, "H", 95.0, 260.0),
        AtomNode(6, "H", 140.0, 205.0),
        AtomNode(7, "H", 140.0, 315.0),
        # C3 H's
        AtomNode(8, "H", 325.0, 260.0),
        AtomNode(9, "H", 280.0, 205.0),
        AtomNode(10, "H", 280.0, 315.0),
    ]
    ace_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=1),
        ChemicalBond(2, 4, order=2),  # C=O internal
        ChemicalBond(1, 5, order=1),
        ChemicalBond(1, 6, order=1),
        ChemicalBond(1, 7, order=1),
        ChemicalBond(3, 8, order=1),
        ChemicalBond(3, 9, order=1),
        ChemicalBond(3, 10, order=1),
    ]
    mol_ace = MoleculeStructure(
        name_vi="Axeton (Đimetyl xeton)",
        iupac_name="Propan-2-one",
        formula="C₃H₆O",
        class_vi="Xeton no, đơn chức (Ketone)",
        boiling_point_c=56.1,
        melting_point_c=-94.7,
        dipole_moment_debye=2.88,
        intermolecular_forces="Hút lưỡng cực rất mạnh (μ = 2.88 D); dung môi phân cực phi proton tuyệt vời",
        physical_state_vi="Chất lỏng dễ bay hơi, mùi thơm ngọt nhẹ thanh khiết, tan vô hạn trong nước",
        chemical_reactions_vi="Không phản ứng với thuốc thử Tollens hay Cu(OH)₂; phản ứng iodoform với I₂/NaOH tạo kết tủa vàng CHI₃",
        functional_groups_vi=[">C=O (Nhóm Cacbonyl / Xeton)"],
        atoms=ace_atoms,
        bonds=ace_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (2, 4)],
    )

    return IsomerPreset(
        code="C3H6O_CARBONYL",
        title_vi="C₃H₆O: Andehit vs Xeton (Đồng phân nhóm chức)",
        isomer_type_vi="Đồng phân nhóm chức (Functional Group Isomerism)",
        formula="C₃H₆O",
        molecule_a=mol_prop,
        molecule_b=mol_ace,
        pedagogical_insight_vi=(
            "Vị trí nhóm cacbonyl C=O ở đầu mạch tạo thành Andehit (-CHO), còn ở giữa mạch tạo thành Xeton (>C=O). "
            "Andehit có nguyên tử H liên kết trực tiếp với cacbonyl nên rất dễ bị oxi hóa (tạo phản ứng tráng gương "
            "AgNO₃/NH₃ sinh ra lớp bạc sáng bóng), trong khi Xeton không tráng gương."
        ),
    )


def _create_c4h8o2_preset() -> IsomerPreset:
    """C4H8O2: Butanoic acid (Carboxylic Acid) vs Ethyl Acetate (Ester)."""
    # Butanoic acid: CH3-CH2-CH2-COOH
    ba_atoms = [
        AtomNode(1, "C", 115.0, 240.0),
        AtomNode(2, "C", 175.0, 240.0),
        AtomNode(3, "C", 235.0, 240.0),
        AtomNode(4, "C", 295.0, 240.0),
        AtomNode(5, "O", 340.0, 195.0),  # =O
        AtomNode(6, "O", 345.0, 280.0),  # -O-H
        AtomNode(7, "H", 385.0, 275.0),  # H on O
        # Hydrogens on C1, C2, C3
        AtomNode(8, "H", 80.0, 240.0),
        AtomNode(9, "H", 115.0, 185.0),
        AtomNode(10, "H", 115.0, 295.0),
        AtomNode(11, "H", 175.0, 185.0),
        AtomNode(12, "H", 175.0, 295.0),
        AtomNode(13, "H", 235.0, 185.0),
        AtomNode(14, "H", 235.0, 295.0),
    ]
    ba_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=1),
        ChemicalBond(3, 4, order=1),
        ChemicalBond(4, 5, order=2),  # C=O
        ChemicalBond(4, 6, order=1),  # C-O
        ChemicalBond(6, 7, order=1),  # O-H
        ChemicalBond(1, 8, order=1),
        ChemicalBond(1, 9, order=1),
        ChemicalBond(1, 10, order=1),
        ChemicalBond(2, 11, order=1),
        ChemicalBond(2, 12, order=1),
        ChemicalBond(3, 13, order=1),
        ChemicalBond(3, 14, order=1),
    ]
    mol_ba = MoleculeStructure(
        name_vi="Axit butanoic (Axit butyric)",
        iupac_name="Butanoic acid",
        formula="C₄H₈O₂",
        class_vi="Axit cacboxylic no, đơn chức (Carboxylic Acid)",
        boiling_point_c=163.7,
        melting_point_c=-7.9,
        dipole_moment_debye=1.05,
        intermolecular_forces="Tạo dạng dime (dimer) liên kết hiđro vòng kép siêu bền giữa 2 phân tử axit",
        physical_state_vi="Chất lỏng nhờn dính, mùi cực kỳ khó chịu của bơ ôi thiu và mồ hôi chua",
        chemical_reactions_vi="Tính axit: làm đỏ quỳ tím, phản ứng với NaHCO₃ sủi bọt khí CO₂↑, phản ứng este hóa với ancol",
        functional_groups_vi=["-COOH (Nhóm Cacboxyl)"],
        atoms=ba_atoms,
        bonds=ba_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (3, 4), (4, 5), (4, 6)],
    )

    # Ethyl acetate: CH3-COO-CH2-CH3
    ea_atoms = [
        AtomNode(1, "C", 115.0, 240.0),  # Acetyl CH3
        AtomNode(2, "C", 175.0, 240.0),  # C=O
        AtomNode(3, "O", 175.0, 180.0),  # =O
        AtomNode(4, "O", 235.0, 240.0),  # -O- ester
        AtomNode(5, "C", 295.0, 240.0),  # -CH2-
        AtomNode(6, "C", 355.0, 240.0),  # -CH3
        # C1 H's
        AtomNode(7, "H", 80.0, 240.0),
        AtomNode(8, "H", 115.0, 185.0),
        AtomNode(9, "H", 115.0, 295.0),
        # C5 H's
        AtomNode(10, "H", 295.0, 185.0),
        AtomNode(11, "H", 295.0, 295.0),
        # C6 H's
        AtomNode(12, "H", 390.0, 240.0),
        AtomNode(13, "H", 355.0, 185.0),
        AtomNode(14, "H", 355.0, 295.0),
    ]
    ea_bonds = [
        ChemicalBond(1, 2, order=1),
        ChemicalBond(2, 3, order=2),  # C=O
        ChemicalBond(2, 4, order=1),  # C-O
        ChemicalBond(4, 5, order=1),  # O-C
        ChemicalBond(5, 6, order=1),  # C-C
        ChemicalBond(1, 7, order=1),
        ChemicalBond(1, 8, order=1),
        ChemicalBond(1, 9, order=1),
        ChemicalBond(5, 10, order=1),
        ChemicalBond(5, 11, order=1),
        ChemicalBond(6, 12, order=1),
        ChemicalBond(6, 13, order=1),
        ChemicalBond(6, 14, order=1),
    ]
    mol_ea = MoleculeStructure(
        name_vi="Etyl axetat (Ethyl acetate)",
        iupac_name="Ethyl ethanoate",
        formula="C₄H₈O₂",
        class_vi="Este no, đơn chức (Ester)",
        boiling_point_c=77.1,
        melting_point_c=-83.6,
        dipole_moment_debye=1.78,
        intermolecular_forces="Hút lưỡng cực Van der Waals; KHÔNG CÓ liên kết hiđro giữa các phân tử este",
        physical_state_vi="Chất lỏng linh động trong suốt, mùi thơm ngọt ngào của quả chín (táo, dứa), nhẹ hơn nước",
        chemical_reactions_vi="Không đổi màu quỳ tím; phản ứng xà phòng hóa thủy phân trong môi trường kiềm (NaOH)",
        functional_groups_vi=["-COO- (Nhóm chức Este)"],
        atoms=ea_atoms,
        bonds=ea_bonds,
        skeletal_path_segments=[(1, 2), (2, 3), (2, 4), (4, 5), (5, 6)],
    )

    return IsomerPreset(
        code="C4H8O2_ACID_ESTER",
        title_vi="C₄H₈O₂: Axit vs Este (Đồng phân nhóm chức)",
        isomer_type_vi="Đồng phân nhóm chức (Functional Group Isomerism)",
        formula="C₄H₈O₂",
        molecule_a=mol_ba,
        molecule_b=mol_ea,
        pedagogical_insight_vi=(
            "Axit butanoic tạo mạng liên kết hiđro dạng dime vòng kép khiến nhiệt độ sôi vọt lên 163.7°C "
            "và có mùi bơ ôi chua nồng. Trái lại, đồng phân este Etyl axetat không có H linh động nên không tạo "
            "liên kết hiđro, sôi ở 77.1°C (thấp hơn tới 86.6°C!) và mang hương thơm hoa quả quyến rũ."
        ),
    )


PRESETS: dict[str, IsomerPreset] = {
    "C2H6O_ALCOHOL_ETHER": _create_c2h6o_preset(),
    "C4H8_POSITION": _create_c4h8_position_preset(),
    "C4H8_CISTRANS": _create_c4h8_cistrans_preset(),
    "C3H6O_CARBONYL": _create_c3h6o_preset(),
    "C4H8O2_ACID_ESTER": _create_c4h8o2_preset(),
}


class InteractiveOrganicChemistryItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for Organic Chemistry Hydrocarbons & Isomerism simulation.
    Features:
      - 5 Authentic Curriculum Presets (Alcohol/Ether, Alkene Position, Alkene Cis/Trans, Aldehyde/Ketone, Acid/Ester).
      - Live toggle between Isomer A and Isomer B.
      - Dual view modes: Lewis Structural Formula vs Skeletal Formula.
      - Real-time valence validation (C: 4, H: 1, O: 2, N: 3, Halogen: 1).
      - Interactive atom dragging with mouse or touchless air pinch.
      - Scientific comparison card explaining physical and chemical divergence.
      - Vector diagram stamping onto LessonBoard.
    """
    closed = Signal()
    preset_changed = Signal(str)
    isomer_switched = Signal(str)  # "A" or "B"

    WIDTH = 780.0
    HEIGHT = 520.0

    def __init__(
        self,
        pos: tuple[float, float] = (300.0, 120.0),
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self._active_preset_code: str = "C2H6O_ALCOHOL_ETHER"
        self._preset: IsomerPreset = PRESETS[self._active_preset_code]
        self._active_isomer: str = "A"  # "A" or "B"
        self._view_mode: str = "LEWIS"  # "LEWIS" or "SKELETAL"

        self._dragged_atom_id: Optional[int] = None
        self._hovered_atom_id: Optional[int] = None
        self._hovered_button: Optional[str] = None

        # Button bounds
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    
    def _init_button_rects(self) -> None:
        """Pre-populate button and canvas bounding rectangles."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        preset_codes = [
            "C2H6O_ALCOHOL_ETHER",
            "C4H8_POSITION",
            "C4H8_CISTRANS",
            "C3H6O_CARBONYL",
            "C4H8O2_ACID_ESTER",
        ]
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        self._btn_rects["canvas"] = QRectF(16.0, 88.0, 440.0, 360.0)
        btn_y = 460.0
        h = 32.0
        self._btn_rects["switch_isomer"] = QRectF(16.0, btn_y, 175.0, h)
        self._btn_rects["toggle_view"] = QRectF(201.0, btn_y, 160.0, h)
        self._btn_rects["reset"] = QRectF(371.0, btn_y, 85.0, h)
        self._btn_rects["stamp"] = QRectF(466.0, btn_y, 298.0, h)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    @property
    def current_molecule(self) -> MoleculeStructure:
        """Return currently active molecule (A or B)."""
        return self._preset.molecule_a if self._active_isomer == "A" else self._preset.molecule_b

    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic isomer preset by code."""
        if preset_code in PRESETS:
            self._active_preset_code = preset_code
            self._preset = PRESETS[preset_code]
            self._active_isomer = "A"
            self._dragged_atom_id = None
            self._hovered_atom_id = None
            self.preset_changed.emit(preset_code)
            self.update()
            return True
        return False

    def switch_isomer(self, isomer: str = "TOGGLE") -> str:
        """Switch between molecule A and molecule B."""
        if isomer.upper() == "A":
            self._active_isomer = "A"
        elif isomer.upper() == "B":
            self._active_isomer = "B"
        else:
            self._active_isomer = "B" if self._active_isomer == "A" else "A"
        self._dragged_atom_id = None
        self._hovered_atom_id = None
        self.isomer_switched.emit(self._active_isomer)
        self.update()
        return self._active_isomer

    def toggle_view_mode(self) -> str:
        """Toggle between LEWIS and SKELETAL view modes."""
        self._view_mode = "SKELETAL" if self._view_mode == "LEWIS" else "LEWIS"
        self.update()
        return self._view_mode

    def set_view_mode(self, mode: str) -> None:
        """Set view mode to 'LEWIS' or 'SKELETAL'."""
        m_up = mode.upper()
        if m_up in ("LEWIS", "SKELETAL"):
            self._view_mode = m_up
            self.update()

    def reset_positions(self) -> None:
        """Reset atom positions from preset baseline."""
        # Re-fetch clean copy
        if self._active_preset_code == "C2H6O_ALCOHOL_ETHER":
            PRESETS["C2H6O_ALCOHOL_ETHER"] = _create_c2h6o_preset()
        elif self._active_preset_code == "C4H8_POSITION":
            PRESETS["C4H8_POSITION"] = _create_c4h8_position_preset()
        elif self._active_preset_code == "C4H8_CISTRANS":
            PRESETS["C4H8_CISTRANS"] = _create_c4h8_cistrans_preset()
        elif self._active_preset_code == "C3H6O_CARBONYL":
            PRESETS["C3H6O_CARBONYL"] = _create_c3h6o_preset()
        elif self._active_preset_code == "C4H8O2_ACID_ESTER":
            PRESETS["C4H8O2_ACID_ESTER"] = _create_c4h8o2_preset()

        self._preset = PRESETS[self._active_preset_code]
        self.update()

    def cleanup(self) -> None:
        """Clean up resources before removal."""
        self._dragged_atom_id = None
        self._hovered_atom_id = None

    def stamp_to_board(self) -> None:
        """Stamp current molecular diagram as permanent vector items on LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import ChemistryTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        mol = self.current_molecule
        is_skeletal = (self._view_mode == "SKELETAL")

        # Stamped vector group
        items = ChemistryTools.create_organic_isomer_diagram(
            center=center,
            molecule=mol,
            preset=self._preset,
            is_skeletal=is_skeletal,
        )
        for itm in items:
            scene.addItem(itm)

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card (Dark Frosted Acrylic)
        bg_rect = self.boundingRect()
        card_path = QPainterPath()
        card_path.addRoundedRect(bg_rect, 14.0, 14.0)
        painter.fillPath(card_path, QColor(15, 23, 42, 245))  # Deep Slate #0f172a
        painter.setPen(QPen(QColor(56, 189, 248, 180), 1.5))  # Cyan border
        painter.drawPath(card_path)

        # 2. Header
        self._paint_header(painter)

        # 3. Preset selector bar
        self._paint_preset_bar(painter)

        # 4. Canvas for Molecule (Lewis / Skeletal)
        self._paint_molecule_canvas(painter)

        # 5. Scientific Comparison & Properties Card
        self._paint_properties_card(painter)

        # 6. Action buttons (Switch A/B, View mode, Reset, Stamp, Close)
        self._paint_action_buttons(painter)

    def _paint_header(self, painter: QPainter) -> None:
        """Paint header with title, formula, and close button."""
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(
            QRectF(16.0, 12.0, 520.0, 24.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "🧪 HÓA HỌC HỮU CƠ: ĐỒNG PHÂN & NHÓM CHỨC",
        )

        sub_font = QFont("Segoe UI", 9)
        painter.setFont(sub_font)
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(16.0, 34.0, 520.0, 18.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"CTPT: {self._preset.formula}  •  Dạng: {self._preset.isomer_type_vi}",
        )

        # Close button
        close_rect = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)
        self._btn_rects["close"] = close_rect
        is_hover = (self._hovered_button == "close")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(239, 68, 68, 200 if is_hover else 120))
        painter.drawRoundedRect(close_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(close_rect, int(Qt.AlignmentFlag.AlignCenter), "✕")

    def _paint_preset_bar(self, painter: QPainter) -> None:
        """Paint 5 preset buttons."""
        preset_items = [
            ("C2H6O_ALCOHOL_ETHER", "C₂H₆O: Ancol / Ete"),
            ("C4H8_POSITION", "C₄H₈: Vị trí C=C"),
            ("C4H8_CISTRANS", "C₄H₈: Cis / Trans"),
            ("C3H6O_CARBONYL", "C₃H₆O: Andehit / Xeton"),
            ("C4H8O2_ACID_ESTER", "C₄H₈O₂: Axit / Este"),
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0

        font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        painter.setFont(font)

        for i, (code, label) in enumerate(preset_items):
            rect = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)
            self._btn_rects[f"preset_{code}"] = rect
            is_active = (self._active_preset_code == code)
            is_hover = (self._hovered_button == f"preset_{code}")

            if is_active:
                painter.setBrush(QColor(14, 165, 233, 220))  # Sky-500
                painter.setPen(QPen(QColor(255, 255, 255), 1.2))
            elif is_hover:
                painter.setBrush(QColor(51, 65, 85, 220))
                painter.setPen(QPen(QColor(56, 189, 248), 1.0))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

    def _paint_molecule_canvas(self, painter: QPainter) -> None:
        """Paint 2D molecular structural formula on dedicated canvas."""
        canvas_rect = QRectF(16.0, 88.0, 440.0, 360.0)
        self._btn_rects["canvas"] = canvas_rect

        # Canvas Frame
        painter.setBrush(QColor(15, 23, 42, 200))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(canvas_rect, 10.0, 10.0)

        # Molecule Title and IUPAC Name
        mol = self.current_molecule
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        isomer_tag = "ĐỒNG PHÂN A" if self._active_isomer == "A" else "ĐỒNG PHÂN B"
        painter.drawText(
            QRectF(26.0, 96.0, 420.0, 20.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"[{isomer_tag}] {mol.name_vi} ({mol.iupac_name})",
        )

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        mode_text = "Công thức Lewis (Đầy đủ liên kết & nguyên tử)" if self._view_mode == "LEWIS" else "Công thức Khung mạch Cacbon (Skeletal)"
        painter.drawText(
            QRectF(26.0, 116.0, 420.0, 18.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            mode_text,
        )

        # Clip painting inside canvas
        painter.save()
        painter.setClipRect(canvas_rect.adjusted(2.0, 2.0, -2.0, -2.0))

        if self._view_mode == "LEWIS":
            self._paint_lewis_structure(painter, mol)
        else:
            self._paint_skeletal_structure(painter, mol)

        painter.restore()

    def _paint_lewis_structure(self, painter: QPainter, mol: MoleculeStructure) -> None:
        """Paint explicit Lewis structural formula with atoms and bonds."""
        # 1. Bonds
        for bond in mol.bonds:
            a1 = mol.get_atom(bond.atom1_id)
            a2 = mol.get_atom(bond.atom2_id)
            if not a1 or not a2:
                continue

            p1 = QPointF(a1.x, a1.y)
            p2 = QPointF(a2.x, a2.y)

            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                continue

            nx = -dy / dist
            ny = dx / dist

            if bond.order == 1:
                painter.setPen(QPen(QColor(148, 163, 184), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(p1, p2)
            elif bond.order == 2:
                offset = 3.5
                painter.setPen(QPen(QColor(148, 163, 184), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(
                    QPointF(p1.x() + nx * offset, p1.y() + ny * offset),
                    QPointF(p2.x() + nx * offset, p2.y() + ny * offset),
                )
                painter.drawLine(
                    QPointF(p1.x() - nx * offset, p1.y() - ny * offset),
                    QPointF(p2.x() - nx * offset, p2.y() - ny * offset),
                )
            elif bond.order == 3:
                offset = 4.5
                painter.setPen(QPen(QColor(148, 163, 184), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(p1, p2)
                painter.drawLine(
                    QPointF(p1.x() + nx * offset, p1.y() + ny * offset),
                    QPointF(p2.x() + nx * offset, p2.y() + ny * offset),
                )
                painter.drawLine(
                    QPointF(p1.x() - nx * offset, p1.y() - ny * offset),
                    QPointF(p2.x() - nx * offset, p2.y() - ny * offset),
                )

        # 2. Atoms
        for atom in mol.atoms:
            self._paint_atom_node(painter, atom)

    def _paint_skeletal_structure(self, painter: QPainter, mol: MoleculeStructure) -> None:
        """Paint skeletal formula (carbon backbone lines + explicit heteroatoms)."""
        # Draw backbone segments
        for seg in mol.skeletal_path_segments:
            a1 = mol.get_atom(seg[0])
            a2 = mol.get_atom(seg[1])
            if not a1 or not a2:
                continue
            # Find bond order between a1 and a2
            order = 1
            for b in mol.bonds:
                if (b.atom1_id == a1.id and b.atom2_id == a2.id) or (b.atom1_id == a2.id and b.atom2_id == a1.id):
                    order = b.order
                    break

            p1 = QPointF(a1.x, a1.y)
            p2 = QPointF(a2.x, a2.y)
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                continue
            nx = -dy / dist
            ny = dx / dist

            if order == 1:
                painter.setPen(QPen(QColor(56, 189, 248), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(p1, p2)
            elif order == 2:
                offset = 3.5
                painter.setPen(QPen(QColor(56, 189, 248), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(
                    QPointF(p1.x() + nx * offset, p1.y() + ny * offset),
                    QPointF(p2.x() + nx * offset, p2.y() + ny * offset),
                )
                painter.drawLine(
                    QPointF(p1.x() - nx * offset, p1.y() - ny * offset),
                    QPointF(p2.x() - nx * offset, p2.y() - ny * offset),
                )

        # Draw explicit heteroatoms or end groups
        for atom in mol.atoms:
            if atom.symbol in ("O", "N", "Cl", "Br"):
                self._paint_atom_node(painter, atom)
            elif atom.symbol == "H" and any(
                b.is_hydrogen_bond or (mol.get_atom(b.atom1_id if b.atom2_id == atom.id else b.atom2_id) and
                mol.get_atom(b.atom1_id if b.atom2_id == atom.id else b.atom2_id).symbol in ("O", "N"))
                for b in mol.bonds if b.atom1_id == atom.id or b.atom2_id == atom.id
            ):
                self._paint_atom_node(painter, atom)
            elif atom.symbol == "C" and len(mol.skeletal_path_segments) == 0:
                self._paint_atom_node(painter, atom)

    def _paint_atom_node(self, painter: QPainter, atom: AtomNode) -> None:
        """Paint a single atom node circle with symbol and lone pairs if O/N."""
        bg_col, txt_col = CPK_COLORS.get(atom.symbol, (QColor(100, 116, 139), QColor(255, 255, 255)))
        r = 13.0 if atom.symbol != "H" else 10.0

        is_hover = (self._hovered_atom_id == atom.id)
        is_drag = (self._dragged_atom_id == atom.id)

        painter.setBrush(bg_col)
        if is_drag:
            painter.setPen(QPen(QColor(234, 179, 8), 2.5))  # Yellow highlight
        elif is_hover:
            painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        else:
            painter.setPen(QPen(QColor(248, 250, 252, 200), 1.2))

        painter.drawEllipse(QPointF(atom.x, atom.y), r, r)

        # Symbol text
        painter.setFont(QFont("Segoe UI", 9 if atom.symbol != "H" else 8, QFont.Weight.Bold))
        painter.setPen(txt_col)
        painter.drawText(
            QRectF(atom.x - r, atom.y - r, r * 2.0, r * 2.0),
            int(Qt.AlignmentFlag.AlignCenter),
            atom.symbol,
        )

        # Lone pairs on Oxygen (2 lone pairs: ..)
        if atom.symbol == "O":
            painter.setBrush(QColor(254, 202, 202))
            painter.setPen(Qt.PenStyle.NoPen)
            # Dot 1 pair (top)
            painter.drawEllipse(QPointF(atom.x - 3.5, atom.y - r - 3.0), 1.5, 1.5)
            painter.drawEllipse(QPointF(atom.x + 3.5, atom.y - r - 3.0), 1.5, 1.5)
            # Dot 2 pair (bottom)
            painter.drawEllipse(QPointF(atom.x - 3.5, atom.y + r + 3.0), 1.5, 1.5)
            painter.drawEllipse(QPointF(atom.x + 3.5, atom.y + r + 3.0), 1.5, 1.5)

    def _paint_properties_card(self, painter: QPainter) -> None:
        """Paint properties and comparison card on right column."""
        card_rect = QRectF(466.0, 88.0, 298.0, 360.0)
        painter.setBrush(QColor(30, 41, 59, 220))  # Slate-800
        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(card_rect, 10.0, 10.0)

        mol = self.current_molecule
        val_map = mol.validate_valences()
        all_valid = all(v[2] for v in val_map.values())

        # Header
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(
            QRectF(476.0, 96.0, 278.0, 20.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "📊 THUỘC TÍNH & HÓA TRỊ (ZERO MOCK)",
        )

        # Property List
        y = 124.0
        row_h = 19.0
        prop_font = QFont("Segoe UI", 8)
        bold_font = QFont("Segoe UI", 8, QFont.Weight.Bold)

        props = [
            ("Công thức phân tử:", mol.formula),
            ("Phân tử khối (M):", f"{mol.molecular_mass:.3f} g/mol"),
            ("Độ bất bão hòa (k):", f"{mol.degree_of_unsaturation:.1f} (số vòng + liên kết π)"),
            ("Loại hợp chất:", mol.class_vi),
            ("Nhiệt độ sôi (Ts):", f"{mol.boiling_point_c:+.1f} °C"),
            ("Nhiệt độ nóng chảy:", f"{mol.melting_point_c:+.1f} °C"),
            ("Momen lưỡng cực (μ):", f"{mol.dipole_moment_debye:.2f} Debye"),
        ]

        for label, val in props:
            painter.setFont(prop_font)
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(QRectF(476.0, y, 130.0, row_h), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), label)
            painter.setFont(bold_font)
            painter.setPen(QColor(241, 245, 249))
            painter.drawText(QRectF(606.0, y, 150.0, row_h), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), val)
            y += row_h

        # Valence indicator badge
        y += 4.0
        val_rect = QRectF(476.0, y, 278.0, 22.0)
        painter.setBrush(QColor(16, 185, 129, 200) if all_valid else QColor(239, 68, 68, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(val_rect, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        badge_text = "✓ Hóa trị hợp lệ (C:4, H:1, O:2, N:3)" if all_valid else "⚠ Vi phạm hóa trị cấu tạo!"
        painter.drawText(val_rect, int(Qt.AlignmentFlag.AlignCenter), badge_text)
        y += 28.0

        # Intermolecular Force & Explanation
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(251, 191, 36))  # Amber-400
        painter.drawText(QRectF(476.0, y, 278.0, 16.0), int(Qt.AlignmentFlag.AlignLeft), "Lực liên kết liên phân tử:")
        y += 18.0

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(
            QRectF(476.0, y, 278.0, 36.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
            mol.intermolecular_forces,
        )
        y += 40.0

        # Pedagogical Comparison Banner
        comp_rect = QRectF(476.0, y, 278.0, 80.0)
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.0))
        painter.drawRoundedRect(comp_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(482.0, y + 4.0, 266.0, 14.0), int(Qt.AlignmentFlag.AlignLeft), "💡 BẢN CHẤT HÓA HỌC SƯ PHẠM:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(203, 213, 225))
        painter.drawText(
            QRectF(482.0, y + 20.0, 266.0, 56.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
            self._preset.pedagogical_insight_vi,
        )

    def _paint_action_buttons(self, painter: QPainter) -> None:
        """Paint bottom toolbar: Switch Isomer A/B, View mode, Reset, Stamp."""
        y = 460.0
        h = 32.0

        # 1. Switch Isomer Button
        sw_rect = QRectF(16.0, y, 175.0, h)
        self._btn_rects["switch_isomer"] = sw_rect
        is_hover = (self._hovered_button == "switch_isomer")
        painter.setBrush(QColor(59, 130, 246, 220 if is_hover else 180))  # Blue-500
        painter.setPen(QPen(QColor(255, 255, 255), 1.0))
        painter.drawRoundedRect(sw_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        sw_label = "🔄 Chuyển sang ĐP B" if self._active_isomer == "A" else "🔄 Chuyển sang ĐP A"
        painter.drawText(sw_rect, int(Qt.AlignmentFlag.AlignCenter), sw_label)

        # 2. View Mode Toggle (Lewis vs Skeletal)
        vm_rect = QRectF(201.0, y, 160.0, h)
        self._btn_rects["toggle_view"] = vm_rect
        is_hover = (self._hovered_button == "toggle_view")
        painter.setBrush(QColor(16, 185, 129, 220 if is_hover else 180))  # Emerald-500
        painter.setPen(QPen(QColor(255, 255, 255), 1.0))
        painter.drawRoundedRect(vm_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        vm_label = "📐 Xem Khung C" if self._view_mode == "LEWIS" else "👁️ Xem Lewis"
        painter.drawText(vm_rect, int(Qt.AlignmentFlag.AlignCenter), vm_label)

        # 3. Reset Positions Button
        rst_rect = QRectF(371.0, y, 85.0, h)
        self._btn_rects["reset"] = rst_rect
        is_hover = (self._hovered_button == "reset")
        painter.setBrush(QColor(71, 85, 105, 220 if is_hover else 160))  # Slate-600
        painter.setPen(QPen(QColor(148, 163, 184), 1.0))
        painter.drawRoundedRect(rst_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(rst_rect, int(Qt.AlignmentFlag.AlignCenter), "↺ Tọa độ")

        # 4. Stamp Vector Diagram Button
        stamp_rect = QRectF(466.0, y, 298.0, h)
        self._btn_rects["stamp"] = stamp_rect
        is_hover = (self._hovered_button == "stamp")
        painter.setBrush(QColor(217, 119, 6, 230 if is_hover else 180))  # Amber-600
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        painter.drawRoundedRect(stamp_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(stamp_rect, int(Qt.AlignmentFlag.AlignCenter), "📌 Đóng dấu Vector lên Bảng vẽ")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Air Gesture Interaction
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        if self._btn_rects.get("close", QRectF()).contains(pos):
            self.closed.emit()
            self.setVisible(False)
            event.accept()
            return

        # 2. Preset buttons
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(pos):
                self.load_preset(code)
                event.accept()
                return

        # 3. Action buttons
        if self._btn_rects.get("switch_isomer", QRectF()).contains(pos):
            self.switch_isomer()
            event.accept()
            return

        if self._btn_rects.get("toggle_view", QRectF()).contains(pos):
            self.toggle_view_mode()
            event.accept()
            return

        if self._btn_rects.get("reset", QRectF()).contains(pos):
            self.reset_positions()
            event.accept()
            return

        if self._btn_rects.get("stamp", QRectF()).contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        # 4. Canvas atom dragging
        canvas_r = self._btn_rects.get("canvas", QRectF())
        if canvas_r.contains(pos):
            mol = self.current_molecule
            hit_atom = None
            for atom in mol.atoms:
                dist = math.hypot(pos.x() - atom.x, pos.y() - atom.y)
                if dist <= 16.0:
                    hit_atom = atom
                    break
            if hit_atom:
                self._dragged_atom_id = hit_atom.id
                hit_atom.is_dragged = True
                self.update()
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragged_atom_id is not None:
            mol = self.current_molecule
            atom = mol.get_atom(self._dragged_atom_id)
            if atom:
                pos = event.pos()
                canvas_r = self._btn_rects.get("canvas", QRectF()).adjusted(15.0, 30.0, -15.0, -15.0)
                # Clamp within canvas
                atom.x = max(canvas_r.left(), min(canvas_r.right(), pos.x()))
                atom.y = max(canvas_r.top(), min(canvas_r.bottom(), pos.y()))
                self.update()
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragged_atom_id is not None:
            mol = self.current_molecule
            atom = mol.get_atom(self._dragged_atom_id)
            if atom:
                atom.is_dragged = False
            self._dragged_atom_id = None
            self.update()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_button
        old_atom = self._hovered_atom_id

        self._hovered_button = None
        for name, r in self._btn_rects.items():
            if r.contains(pos):
                self._hovered_button = name
                break

        self._hovered_atom_id = None
        canvas_r = self._btn_rects.get("canvas", QRectF())
        if canvas_r.contains(pos):
            mol = self.current_molecule
            for atom in mol.atoms:
                if math.hypot(pos.x() - atom.x, pos.y() - atom.y) <= 15.0:
                    self._hovered_atom_id = atom.id
                    break

        if self._hovered_button != old_btn or self._hovered_atom_id != old_atom:
            self.update()

        super().hoverMoveEvent(event)

    def handle_air_pinch(self, point: QPointF) -> bool:
        """Handle touchless air pinch gesture for dragging atoms or clicking buttons."""
        local_p = self.mapFromScene(point)

        # Check buttons first
        for name, r in self._btn_rects.items():
            if r.contains(local_p):
                if name == "close":
                    self.closed.emit()
                    self.setVisible(False)
                    return True
                elif name.startswith("preset_"):
                    code = name.replace("preset_", "")
                    return self.load_preset(code)
                elif name == "switch_isomer":
                    self.switch_isomer()
                    return True
                elif name == "toggle_view":
                    self.toggle_view_mode()
                    return True
                elif name == "reset":
                    self.reset_positions()
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        # Check atom drag
        canvas_r = self._btn_rects.get("canvas", QRectF())
        if canvas_r.contains(local_p):
            mol = self.current_molecule
            for atom in mol.atoms:
                if math.hypot(local_p.x() - atom.x, local_p.y() - atom.y) <= 20.0:
                    self._dragged_atom_id = atom.id
                    atom.is_dragged = True
                    self.update()
                    return True

        return False

    def handle_air_move(self, point: QPointF) -> bool:
        """Move dragged atom during continuous air pinch."""
        if self._dragged_atom_id is not None:
            local_p = self.mapFromScene(point)
            mol = self.current_molecule
            atom = mol.get_atom(self._dragged_atom_id)
            if atom:
                canvas_r = self._btn_rects.get("canvas", QRectF()).adjusted(15.0, 30.0, -15.0, -15.0)
                atom.x = max(canvas_r.left(), min(canvas_r.right(), local_p.x()))
                atom.y = max(canvas_r.top(), min(canvas_r.bottom(), local_p.y()))
                self.update()
                return True
        return False

    def handle_air_release(self) -> None:
        """Release atom on gesture release."""
        if self._dragged_atom_id is not None:
            mol = self.current_molecule
            atom = mol.get_atom(self._dragged_atom_id)
            if atom:
                atom.is_dragged = False
            self._dragged_atom_id = None
            self.update()
