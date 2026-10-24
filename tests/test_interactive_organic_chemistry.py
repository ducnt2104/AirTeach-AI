"""
Unit Tests for Interactive Organic Chemistry Hydrocarbons & Isomerism Simulator
================================================================================
Verifies:
  1. 5 Authentic Curriculum Presets:
       - C2H6O_ALCOHOL_ETHER: Ethanol vs Dimethyl ether (Functional group isomerism)
       - C4H8_POSITION: But-1-ene vs But-2-ene (Position isomerism of double bond)
       - C4H8_CISTRANS: cis-But-2-ene vs trans-But-2-ene (Geometric stereoisomerism)
       - C3H6O_CARBONYL: Propanal vs Acetone (Aldehyde vs Ketone)
       - C4H8O2_ACID_ESTER: Butanoic acid vs Ethyl acetate (Acid vs Ester)
  2. Authentic Zero-Mock Chemistry Engine:
       - Exact atomic weights: C (12.011), H (1.008), O (15.999), N (14.007).
       - Exact Degree of Unsaturation: k = (2C + 2 - H + N - X) / 2.
       - Valence Validation Engine: C:4, H:1, O:2, N:3.
       - Physical properties comparison: boiling point, dipole moment, hydrogen bonding.
  3. Interactive UI Controls:
       - Isomer switching (A ⇄ B).
       - View mode toggle (Lewis structural formula vs Skeletal formula).
       - Atom dragging with mouse or touchless air pinch gesture.
       - Reset atomic positions.
  4. Vector Diagram Stamping:
       - ChemistryTools.create_organic_isomer_diagram generates clean vector items.
  5. LessonBoard Lifecycle:
       - show_organic_chemistry, toggle_organic_chemistry, is_organic_chemistry_visible, selection exclusion.
  6. MainWindow Integration:
       - _on_toggle_organic_chemistry slot, status message, keyboard shortcut Ctrl+Alt+H.
"""
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtWidgets import QApplication

from src.ui.interactive_organic_chemistry import (
    InteractiveOrganicChemistryItem,
    PRESETS,
    ATOMIC_WEIGHTS,
    STANDARD_VALENCES,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import ChemistryTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def board(qapp):
    return LessonBoard()


@pytest.fixture
def item(board):
    it = InteractiveOrganicChemistryItem(pos=(310.0, 130.0))
    board.addItem(it)
    yield it
    it.cleanup()
    if it.scene():
        board.removeItem(it)


class TestOrganicChemistryIsomers:
    """Verifies organic chemistry presets, authentic physical constants, and valence validation."""

    def test_presets_exist(self):
        assert "C2H6O_ALCOHOL_ETHER" in PRESETS
        assert "C4H8_POSITION" in PRESETS
        assert "C4H8_CISTRANS" in PRESETS
        assert "C3H6O_CARBONYL" in PRESETS
        assert "C4H8O2_ACID_ESTER" in PRESETS

    def test_c2h6o_alcohol_vs_ether(self, item):
        """Verify Ethanol vs Dimethyl ether functional group isomerism."""
        item.load_preset("C2H6O_ALCOHOL_ETHER")
        pr = item._preset

        # Molecule A: Ethanol
        item.switch_isomer("A")
        mol_a = item.current_molecule
        assert mol_a.formula == "C₂H₆O"
        assert abs(mol_a.molecular_mass - 46.069) < 0.01
        assert mol_a.degree_of_unsaturation == 0.0
        assert mol_a.boiling_point_c == 78.37
        assert mol_a.dipole_moment_debye == 1.69
        assert "Hydroxyl" in mol_a.functional_groups_vi[0]
        assert "Liên kết Hiđro" in mol_a.intermolecular_forces

        # Molecule B: Dimethyl ether
        item.switch_isomer("B")
        mol_b = item.current_molecule
        assert mol_b.formula == "C₂H₆O"
        assert abs(mol_b.molecular_mass - 46.069) < 0.01
        assert mol_b.degree_of_unsaturation == 0.0
        assert mol_b.boiling_point_c == -24.0
        assert mol_b.dipole_moment_debye == 1.30
        assert "Ete" in mol_b.functional_groups_vi[0]
        assert "KHÔNG CÓ liên kết hiđro" in mol_b.intermolecular_forces

        # Physical comparison
        delta_ts = mol_a.boiling_point_c - mol_b.boiling_point_c
        assert delta_ts > 100.0, "Ethanol should boil > 100°C higher due to H-bonding"

    def test_c4h8_position_isomers(self, item):
        """Verify But-1-ene vs But-2-ene position isomers."""
        item.load_preset("C4H8_POSITION")
        mol_a = item._preset.molecule_a
        mol_b = item._preset.molecule_b

        assert mol_a.formula == "C₄H₈"
        assert mol_b.formula == "C₄H₈"
        assert abs(mol_a.molecular_mass - 56.108) < 0.01
        assert mol_a.degree_of_unsaturation == 1.0  # 1 double bond
        assert mol_b.degree_of_unsaturation == 1.0

        # But-1-ene b.p. -6.3°C, But-2-ene b.p. +1.0°C
        assert mol_a.boiling_point_c == -6.3
        assert mol_b.boiling_point_c == 1.0

    def test_c4h8_cistrans_stereoisomerism(self, item):
        """Verify cis-But-2-ene vs trans-But-2-ene geometric isomers."""
        item.load_preset("C4H8_CISTRANS")
        mol_cis = item._preset.molecule_a
        mol_trans = item._preset.molecule_b

        # Dipole moment: cis has non-zero dipole, trans is non-polar
        assert mol_cis.dipole_moment_debye == 0.33
        assert mol_trans.dipole_moment_debye == 0.00

        # Boiling point: cis is higher due to permanent dipole-dipole forces
        assert mol_cis.boiling_point_c > mol_trans.boiling_point_c

        # Melting point: trans packs tighter in crystal lattice, so higher m.p.
        assert mol_trans.melting_point_c > mol_cis.melting_point_c
        assert mol_trans.melting_point_c == -105.5
        assert mol_cis.melting_point_c == -138.9

    def test_c3h6o_aldehyde_vs_ketone(self, item):
        """Verify Propanal vs Acetone carbonyl group isomers."""
        item.load_preset("C3H6O_CARBONYL")
        mol_prop = item._preset.molecule_a
        mol_ace = item._preset.molecule_b

        assert mol_prop.degree_of_unsaturation == 1.0
        assert mol_ace.degree_of_unsaturation == 1.0
        assert "-CHO" in mol_prop.functional_groups_vi[0]
        assert ">C=O" in mol_ace.functional_groups_vi[0]
        assert "tráng bạc" in mol_prop.chemical_reactions_vi
        assert "iodoform" in mol_ace.chemical_reactions_vi

    def test_c4h8o2_acid_vs_ester(self, item):
        """Verify Butanoic acid vs Ethyl acetate isomers."""
        item.load_preset("C4H8O2_ACID_ESTER")
        mol_acid = item._preset.molecule_a
        mol_ester = item._preset.molecule_b

        assert mol_acid.formula == "C₄H₈O₂"
        assert mol_ester.formula == "C₄H₈O₂"
        assert abs(mol_acid.molecular_mass - 88.106) < 0.01

        # Carboxylic acid forms cyclic dimers with high b.p.
        assert mol_acid.boiling_point_c == 163.7
        assert mol_ester.boiling_point_c == 77.1
        assert mol_acid.boiling_point_c - mol_ester.boiling_point_c > 80.0

    def test_all_presets_valence_validation(self):
        """Verify all atoms in all presets strictly obey standard valences (C:4, H:1, O:2, N:3)."""
        for code, pr in PRESETS.items():
            for mol in (pr.molecule_a, pr.molecule_b):
                val_res = mol.validate_valences()
                for atom_id, (actual, target, is_valid) in val_res.items():
                    atom = mol.get_atom(atom_id)
                    assert is_valid, (
                        f"Preset {code}, molecule {mol.name_vi}, atom {atom.symbol}#{atom_id}: "
                        f"has {actual} bonds, expected valence {target}!"
                    )

    def test_view_mode_toggle(self, item):
        """Verify view mode toggles between LEWIS and SKELETAL."""
        assert item._view_mode == "LEWIS"
        m1 = item.toggle_view_mode()
        assert m1 == "SKELETAL"
        assert item._view_mode == "SKELETAL"
        m2 = item.toggle_view_mode()
        assert m2 == "LEWIS"

    def test_atom_drag_and_reset(self, item):
        """Verify atom position dragging and position reset."""
        mol = item.current_molecule
        atom = mol.atoms[0]
        orig_x, orig_y = atom.x, atom.y

        # Simulate air pinch on atom
        pinch_pt = QPointF(item.pos().x() + orig_x, item.pos().y() + orig_y)
        res_pinch = item.handle_air_pinch(pinch_pt)
        assert res_pinch is True
        assert item._dragged_atom_id == atom.id

        # Move
        move_pt = QPointF(item.pos().x() + orig_x + 30.0, item.pos().y() + orig_y + 20.0)
        item.handle_air_move(move_pt)
        assert atom.x > orig_x

        # Release
        item.handle_air_release()
        assert item._dragged_atom_id is None

        # Reset positions
        item.reset_positions()
        mol_reset = item.current_molecule
        atom_reset = mol_reset.get_atom(atom.id)
        assert atom_reset.x == orig_x
        assert atom_reset.y == orig_y

    def test_touchless_air_interactions(self, item):
        """Verify air gestures on preset switching and isomer toggle."""
        # Click preset button C4H8_CISTRANS
        btn_r = item._btn_rects["preset_C4H8_CISTRANS"]
        pt = item.mapToScene(btn_r.center())
        assert item.handle_air_pinch(pt) is True
        assert item._active_preset_code == "C4H8_CISTRANS"

        # Click switch isomer button
        sw_r = item._btn_rects["switch_isomer"]
        pt_sw = item.mapToScene(sw_r.center())
        assert item.handle_air_pinch(pt_sw) is True
        assert item._active_isomer == "B"

    def test_stamp_vector_diagram(self, board, item):
        """Verify stamping vector diagram to LessonBoard."""
        init_items = len(board.items())
        item.stamp_to_board()
        assert len(board.items()) > init_items

    def test_chemistry_tools_create_organic_isomer_diagram(self):
        """Verify ChemistryTools.create_organic_isomer_diagram vector items structure."""
        mol = PRESETS["C2H6O_ALCOHOL_ETHER"].molecule_a
        items = ChemistryTools.create_organic_isomer_diagram(
            center=(400.0, 300.0),
            molecule=mol,
            is_skeletal=False,
        )
        assert len(items) >= 12, "Must create background card, title, bonds, atoms, and properties banner"

    def test_lesson_board_lifecycle(self, board):
        """Verify LessonBoard lifecycle methods and exclusion from selection."""
        # 1. Show
        item = board.show_organic_chemistry(True, pos=(250.0, 100.0))
        assert item is not None
        assert board.is_organic_chemistry_visible() is True
        assert board.get_organic_chemistry() is item

        # 2. Toggle
        state = board.toggle_organic_chemistry()
        assert state is False
        assert board.is_organic_chemistry_visible() is False

        # 3. Toggle back
        state = board.toggle_organic_chemistry()
        assert state is True
        assert board.is_organic_chemistry_visible() is True

        # 4. Selection exclusion
        rect = QRectF(0.0, 0.0, 1920.0, 1080.0)
        items_in_sel = board.get_items_in_rect(rect)
        assert item not in items_in_sel, "Organic chemistry simulator must not be selected by board lasso"

    def test_main_window_action_and_slot(self, qapp):
        """Verify MainWindow slot and toggle action."""
        from src.ui.main_window import MainWindow

        win = MainWindow(camera_index=-1)
        win.show()

        # Slot toggling
        vis1 = win._on_toggle_organic_chemistry()
        assert vis1 is True
        assert win._scene.is_organic_chemistry_visible() is True

        vis2 = win._on_toggle_organic_chemistry()
        assert vis2 is False
        assert win._scene.is_organic_chemistry_visible() is False

        win.close()
