"""
Unit test suite for InteractivePeriodicTableItem (Milestone 106).
Covers IUPAC element database integrity, Aufbau subshell anomalies, Bohr orbital electron shells,
Pauling electronegativity trends, touchless air gestures, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_periodic_table import (
    InteractivePeriodicTableItem,
    ELEMENT_DB,
    PRESETS,
    PeriodicPreset,
    ElementData,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import ChemistryTools


class TestInteractivePeriodicTable:

    def test_initialization(self, qapp):
        """Verifies default properties, dimensions, and flags of InteractivePeriodicTableItem."""
        item = InteractivePeriodicTableItem(pos=(180.0, 100.0))
        assert item.pos() == QPointF(180.0, 100.0)
        assert item.WIDTH == 640.0
        assert item.HEIGHT == 380.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 640.0, 380.0)
        assert item.active_preset.code == "CHONPS_LIFE_ELEMENTS"
        assert item.selected_z == 6  # Carbon
        assert item.current_element.symbol == "C"
        assert item.view_mode == "BOHR_SHELLS"
        item.cleanup()

    def test_all_five_presets(self, qapp):
        """Verifies that all 5 periodic presets load cleanly and update target element and view mode."""
        item = InteractivePeriodicTableItem()
        preset_codes = [
            "CHONPS_LIFE_ELEMENTS",
            "ELECTRONEGATIVITY_TRENDS",
            "NOBLE_GAS_OCTET_STABILITY",
            "TRANSITION_METALS_AUFBAU",
            "ALKALI_VS_HALOGENS",
        ]
        for code in preset_codes:
            item.load_preset(code)
            assert item.active_preset.code == code
            assert 1 <= item.selected_z <= 36
            assert item.view_mode in ("STANDARD", "ELECTRONEGATIVITY", "BOHR_SHELLS")
        item.cleanup()

    def test_element_database_integrity(self, qapp):
        """Verifies chemical database integrity for all 36 elements (Z=1 to 36)."""
        assert len(ELEMENT_DB) == 36
        for z in range(1, 37):
            elem = ELEMENT_DB[z]
            assert elem.z == z
            assert len(elem.symbol) in (1, 2)
            assert elem.atomic_mass > 0.0
            assert 1 <= elem.period <= 4
            assert 1 <= elem.group <= 18
            assert len(elem.electron_config) > 0
            # Bohr shells sum must exactly equal atomic number Z!
            assert sum(elem.bohr_shells) == z
            assert elem.cpk_hex.startswith("#")

    def test_aufbau_anomalies(self, qapp):
        """Verifies transition metal subshell half-filled and fully-filled stability anomalies."""
        # Chromium Z=24 has anomalous [Ar] 3d5 4s1 instead of 3d4 4s2
        cr = ELEMENT_DB[24]
        assert "3d⁵ 4s¹" in cr.electron_config
        assert cr.bohr_shells == (2, 8, 13, 1)

        # Copper Z=29 has anomalous [Ar] 3d10 4s1 instead of 3d9 4s2
        cu = ELEMENT_DB[29]
        assert "3d¹⁰ 4s¹" in cu.electron_config
        assert cu.bohr_shells == (2, 8, 18, 1)

    def test_pauling_electronegativity_trends(self, qapp):
        """Verifies Pauling electronegativity values and periodic trends."""
        # Fluorine (Z=9) is the most electronegative element
        f = ELEMENT_DB[9]
        assert f.electronegativity == 3.98

        # Oxygen (Z=8) is second most electronegative
        o = ELEMENT_DB[8]
        assert o.electronegativity == 3.44

        # Alkali metals have low electronegativity (< 1.0)
        na = ELEMENT_DB[11]
        k = ELEMENT_DB[19]
        assert na.electronegativity < 1.0
        assert k.electronegativity < na.electronegativity  # Decreases down the group!

        # Noble gases have no electronegativity on Pauling scale
        he = ELEMENT_DB[2]
        ne = ELEMENT_DB[10]
        ar = ELEMENT_DB[18]
        assert he.electronegativity is None
        assert ne.electronegativity is None
        assert ar.electronegativity is None

    def test_element_selection_and_clamping(self, qapp):
        """Verifies element selection by Z, bounds clamping [1, 36], and step_z wrapping."""
        item = InteractivePeriodicTableItem()

        # Selection & clamping
        item.set_element_z(8)
        assert item.selected_z == 8
        assert item.current_element.symbol == "O"

        item.set_element_z(-5)
        assert item.selected_z == 1
        item.set_element_z(100)
        assert item.selected_z == 36

        # Stepping with wrap-around
        item.set_element_z(36)
        item.step_z(+1)
        assert item.selected_z == 1

        item.step_z(-1)
        assert item.selected_z == 36
        item.cleanup()

    def test_view_mode_toggle(self, qapp):
        """Verifies view mode cycling between STANDARD, ELECTRONEGATIVITY, and BOHR_SHELLS."""
        item = InteractivePeriodicTableItem()
        item._view_mode = "STANDARD"

        m1 = item.toggle_view_mode()
        assert m1 == "ELECTRONEGATIVITY"

        m2 = item.toggle_view_mode()
        assert m2 == "BOHR_SHELLS"

        m3 = item.toggle_view_mode()
        assert m3 == "STANDARD"
        item.cleanup()

    def test_orbital_animation_tick(self, qapp):
        """Verifies Bohr orbital electron rotation angle updates on timer tick."""
        item = InteractivePeriodicTableItem()
        ang0 = item._electron_orbit_angle
        item._on_orbit_tick()
        assert item._electron_orbit_angle != ang0
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        """Verifies touchless air pinch on preset buttons, steppers, mode toggle, and element tiles."""
        item = InteractivePeriodicTableItem(pos=(100.0, 100.0))

        # Air pinch outside
        assert item.handle_air_pinch(QPointF(50.0, 50.0)) is False

        # Air pinch on Noble Gas preset button
        r_noble = item._btn_rects.get("preset_NOBLE_GAS_OCTET_STABILITY")
        assert r_noble is not None
        item.handle_air_pinch(item.mapToScene(r_noble.center()))
        assert item.active_preset.code == "NOBLE_GAS_OCTET_STABILITY"
        assert item.selected_z == 10  # Neon

        # Air pinch on Next Z stepper
        r_next = item._btn_rects.get("next_z")
        item.handle_air_pinch(item.mapToScene(r_next.center()))
        assert item.selected_z == 11  # Sodium

        # Air pinch on Mode toggle
        r_mode = item._btn_rects.get("toggle_mode")
        m0 = item.view_mode
        item.handle_air_pinch(item.mapToScene(r_mode.center()))
        assert item.view_mode != m0

        # Air pinch on direct element tile: Iron Fe (Z=26)
        r_fe = item._btn_rects.get("elem_26")
        assert r_fe is not None
        item.handle_air_pinch(item.mapToScene(r_fe.center()))
        assert item.selected_z == 26
        assert item.current_element.symbol == "Fe"
        item.cleanup()

    def test_stem_tools_diagram_generation_and_stamp(self, qapp):
        """Verifies ChemistryTools.create_periodic_table_diagram and stamping to LessonBoard."""
        board = LessonBoard()
        item = InteractivePeriodicTableItem(pos=QPointF(100.0, 100.0))
        board.addItem(item)
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count

        # Test direct diagram creation via ChemistryTools
        items = ChemistryTools.create_periodic_table_diagram(
            center=(400.0, 300.0),
            atomic_z=6,  # Carbon
            preset_code="CHONPS_LIFE_ELEMENTS",
            view_mode="BOHR_SHELLS",
        )
        assert len(items) >= 12
        text_contents = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("BẢNG TUẦN HOÀN" in t for t in text_contents)
        assert any("Cacbon" in t or "C" in t for t in text_contents)
        item.cleanup()

    def test_lesson_board_lifecycle_and_lasso_exclusion(self, qapp):
        """Verifies LessonBoard show/toggle/getter lifecycle and lasso exclusion."""
        board = LessonBoard()

        # Initially not visible
        assert board.is_periodic_table_visible() is False
        assert board.get_periodic_table() is None

        # Show
        item = board.show_periodic_table(True, pos=(200.0, 150.0))
        assert item is not None
        assert board.is_periodic_table_visible() is True
        assert board.get_periodic_table() is item
        assert board.get_periodic_table_item() is item
        assert item.pos() == QPointF(200.0, 150.0)

        # Toggle hide
        new_state = board.toggle_periodic_table()
        assert new_state is False
        assert board.is_periodic_table_visible() is False
        assert board.get_periodic_table() is None

        # Toggle show
        new_state_2 = board.toggle_periodic_table()
        assert new_state_2 is True
        assert board.is_periodic_table_visible() is True

        # Verify exclusion in lasso selection
        rect = QRectF(100.0, 100.0, 500.0, 500.0)
        items_in_rect = board.get_items_in_rect(rect)
        assert item not in items_in_rect
        item.cleanup()

    def test_main_window_toggle_slot(self, qapp):
        """Verifies MainWindow slot _on_toggle_periodic_table properly delegates to LessonBoard."""
        from src.ui.main_window import MainWindow
        win = MainWindow()
        assert win._scene.is_periodic_table_visible() is False

        # Toggle via slot
        res = win._on_toggle_periodic_table()
        assert res is True
        assert win._scene.is_periodic_table_visible() is True

        # Toggle off
        res_off = win._on_toggle_periodic_table()
        assert res_off is False
        assert win._scene.is_periodic_table_visible() is False
        win.close()
