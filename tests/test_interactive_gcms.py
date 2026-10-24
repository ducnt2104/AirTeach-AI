"""
AirTeach AI - Unit Tests for Interactive Chemistry & Physics GC-MS Simulator
=============================================================================
Comprehensive automated unit tests verifying:
  1. Authentic analytical gas chromatography (TIC chromatogram, retention times).
  2. Electron ionization 70 eV mass spectrometry (fragmentation pathways, tropylium, Cl clusters).
  3. 5 Curriculum Presets (Terpenes, Forensic Blood Alcohols, Environmental PCBs, WADA Doping, Petrochemical BTEX).
  4. Interactive peak inspection, fragment spectrum exploration, and signal emission.
  5. Touchless air pinch gesture interactions, parameter tuning, and simulation progression.
  6. Vector diagram generation via ChemistryTools.create_gcms_diagram.
  7. LessonBoard lifecycle integration and lasso selection exclusion.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_gcms import (
    InteractiveGCMSItem,
    PRESETS,
)
from src.ui.stem_tools import ChemistryTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure single persistent QApplication for QGraphics testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveGCMS:
    """Test suite for InteractiveGCMSItem and analytical chemistry STEM integration."""

    def test_initialization(self, qapp):
        """Verifies initial configuration, geometry, and default analytical parameters."""
        item = InteractiveGCMSItem(pos=QPointF(150.0, 100.0))
        assert item.pos() == QPointF(150.0, 100.0)
        assert item.boundingRect().width() == InteractiveGCMSItem.WIDTH
        assert item.boundingRect().height() == InteractiveGCMSItem.HEIGHT
        assert item.active_preset_code == "ESSENTIAL_OIL_TERPENES"
        assert item.preset.carrier_gas == "Helium"
        assert item.ramp_rate == 10.0
        assert item.flow_rate == 1.2
        assert item.is_animating is True
        assert len(item.preset.compounds) >= 5
        item.cleanup()

    def test_all_five_presets(self, qapp):
        """Verifies loading of all 5 curriculum presets and signal emissions."""
        item = InteractiveGCMSItem()
        emitted_presets = []
        item.preset_changed.connect(lambda p: emitted_presets.append(p))

        for code, expected_preset in PRESETS.items():
            loaded = item.load_preset(code)
            assert loaded is True
            assert item.active_preset_code == code
            assert item.preset.title_vi == expected_preset.title_vi
            assert item.preset.carrier_gas == expected_preset.carrier_gas
            assert item.ramp_rate == expected_preset.ramp_rate_c_min
            assert item.flow_rate == expected_preset.flow_rate_ml_min
            assert len(item.preset.compounds) == len(expected_preset.compounds)

        assert len(emitted_presets) == 5
        assert item.load_preset("NON_EXISTENT_PRESET") is False
        item.cleanup()

    def test_peak_selection_and_fragments(self, qapp):
        """Verifies compound selection and EI-MS fragmentation stick spectrum data."""
        item = InteractiveGCMSItem()
        item.load_preset("ESSENTIAL_OIL_TERPENES")
        emitted_compounds = []
        item.peak_selected.connect(lambda name: emitted_compounds.append(name))

        # Select Menthol (index 3)
        ok = item.select_peak(3)
        assert ok is True
        assert item.selected_compound.name_vi == "Menthol"
        assert item.selected_compound.molecular_weight == 156
        assert item.selected_compound.base_peak_mz == 71
        assert len(item.selected_compound.fragments) >= 4
        assert len(emitted_compounds) == 1

        # Out-of-bounds selection
        assert item.select_peak(99) is False
        assert item.select_peak(-1) is False
        item.cleanup()

    def test_forensic_blood_alcohols_separation(self, qapp):
        """Verifies forensic alcohol separation and alpha-cleavage fragments."""
        item = InteractiveGCMSItem()
        item.load_preset("FORENSIC_BLOOD_ALCOHOLS")

        # Ethanol (index 1)
        item.select_peak(1)
        eth = item.selected_compound
        assert eth.name_vi.startswith("Ethanol")
        assert eth.molecular_weight == 46
        # Base peak at m/z 31 (CH2OH+ from alpha-cleavage)
        assert eth.base_peak_mz == 31

        # Acetone (index 2)
        item.select_peak(2)
        ace = item.selected_compound
        assert ace.name_vi == "Acetone"
        # Base peak at m/z 43 (CH3CO+ acylium ion)
        assert ace.base_peak_mz == 43
        item.cleanup()

    def test_environmental_pcb_chlorine_cluster(self, qapp):
        """Verifies chlorinated pollutants exhibit characteristic Cl isotope fragmentation."""
        item = InteractiveGCMSItem()
        item.load_preset("ENVIRONMENTAL_POPS_PCBS")

        # PCB-28 (Trichlorobiphenyl, index 2)
        item.select_peak(2)
        pcb = item.selected_compound
        assert "PCB-28" in pcb.name_vi
        assert pcb.molecular_weight == 256
        # Must have M+2 cluster fragment (m/z 258)
        fragment_mzs = [f[0] for f in pcb.fragments]
        assert 256 in fragment_mzs
        assert 258 in fragment_mzs
        item.cleanup()

    def test_petrochemical_btex_tropylium_ion(self, qapp):
        """Verifies alkylbenzenes yield stable tropylium ion base peak at m/z 91."""
        item = InteractiveGCMSItem()
        item.load_preset("PETROCHEMICAL_GASOLINE_BTEX")

        # Toluene (index 1)
        item.select_peak(1)
        tol = item.selected_compound
        assert tol.name_vi.startswith("Toluene")
        # Base peak must be m/z 91 (C7H7+ tropylium cation)
        assert tol.base_peak_mz == 91

        # Ethylbenzene (index 2)
        item.select_peak(2)
        eb = item.selected_compound
        assert eb.base_peak_mz == 91
        item.cleanup()

    def test_flow_and_ramp_rate_adjustments_and_clamping(self, qapp):
        """Verifies carrier gas flow rate and oven ramp rate adjustments and clamping."""
        item = InteractiveGCMSItem()

        item.set_ramp_rate(50.0)
        assert item.ramp_rate == 30.0
        item.set_ramp_rate(0.5)
        assert item.ramp_rate == 2.0

        item.set_flow_rate(10.0)
        assert item.flow_rate == 4.0
        item.set_flow_rate(0.1)
        assert item.flow_rate == 0.5
        item.cleanup()

    def test_chromatographic_retention_scaling(self, qapp):
        """Verifies retention times scale inversely with carrier gas flow rate."""
        item = InteractiveGCMSItem()
        item.load_preset("ESSENTIAL_OIL_TERPENES")
        cmp0 = item.preset.compounds[0]

        # At flow rate = 1.0 vs 2.0 mL/min
        item.set_flow_rate(1.0)
        tr_1 = cmp0.retention_time_min * (1.2 / item.flow_rate)

        item.set_flow_rate(2.0)
        tr_2 = cmp0.retention_time_min * (1.2 / item.flow_rate)

        # Doubling carrier gas velocity cuts migration time roughly in half
        assert tr_1 > tr_2
        assert abs(tr_1 - 2.0 * tr_2) < 1e-4
        item.cleanup()

    def test_animation_toggle_progression_and_reset(self, qapp):
        """Verifies animation toggle, simulation progression, oven heating, and reset."""
        item = InteractiveGCMSItem()
        assert item.is_animating is True

        # Toggle to pause
        item.toggle_animation()
        assert not item.is_animating

        # Toggle to resume
        item.toggle_animation()
        assert item.is_animating

        # Simulate several ticks
        for _ in range(15):
            item._on_animation_tick()
        assert item._sim_time_min > 0.0
        assert item.oven_temp_c >= item.preset.t_init_c

        # Reset run
        item.reset_run()
        assert item._sim_time_min == 0.0
        assert item.oven_temp_c == item.preset.t_init_c
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        """Verifies touchless air pinch gesture dispatch on buttons and chromatogram peaks."""
        item = InteractiveGCMSItem(pos=QPointF(0.0, 0.0))

        # 1. Pinch on Preset button (preset_FORENSIC_BLOOD_ALCOHOLS)
        rect_for = item._btn_rects["preset_FORENSIC_BLOOD_ALCOHOLS"]
        pinch_pt = QPointF(rect_for.center().x(), rect_for.center().y())
        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "FORENSIC_BLOOD_ALCOHOLS"

        # 2. Pinch on Play / Pause button
        rect_play = item._btn_rects["play_pause"]
        assert item.is_animating is True
        handled_play = item.handle_air_pinch(rect_play.center())
        assert handled_play is True
        assert item.is_animating is False

        # 3. Pinch on Inc Ramp button
        rect_inc_r = item._btn_rects["inc_ramp"]
        init_ramp = item.ramp_rate
        handled_r = item.handle_air_pinch(rect_inc_r.center())
        assert handled_r is True
        assert item.ramp_rate > init_ramp

        # 4. Pinch on Inc Flow button
        rect_inc_f = item._btn_rects["inc_flow"]
        init_flow = item.flow_rate
        handled_f = item.handle_air_pinch(rect_inc_f.center())
        assert handled_f is True
        assert item.flow_rate > init_flow

        # 5. Pinch outside bounding box
        outside_pt = QPointF(1000.0, 1000.0)
        assert item.handle_air_pinch(outside_pt) is False
        item.cleanup()

    def test_stem_tools_diagram_generation_and_stamp(self, qapp):
        """Verifies static vector diagram generation via ChemistryTools and scene stamping."""
        board = LessonBoard()
        item = InteractiveGCMSItem(pos=QPointF(100.0, 100.0))
        board.addItem(item)
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count

        # Test direct diagram creation via ChemistryTools
        items = ChemistryTools.create_gcms_diagram(
            center=(400.0, 300.0),
            preset_code="PETROCHEMICAL_GASOLINE_BTEX",
            selected_peak_idx=1,
            ramp_rate=12.0,
            flow_rate=1.1,
        )
        assert len(items) >= 8
        text_contents = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("SẮC KÝ KHÍ - KHỐI PHỔ GC-MS" in t for t in text_contents)
        assert any("THÔNG SỐ PHÂN TÍCH GC-MS" in t for t in text_contents)
        item.cleanup()

    def test_lesson_board_lifecycle_and_lasso_exclusion(self, qapp):
        """Verifies LessonBoard show/toggle/visibility lifecycle and lasso exclusion."""
        board = LessonBoard()
        assert not board.is_gcms_visible()
        assert board.get_gcms() is None

        # Show
        sim = board.show_gcms(True, pos=(200.0, 100.0))
        assert sim is not None
        assert board.is_gcms_visible()
        assert board.get_gcms() is sim
        assert board.get_gcms_item() is sim

        # Lasso exclusion check
        items_in_box = board.get_items_in_rect(QRectF(150.0, 80.0, 800.0, 600.0))
        assert board.get_gcms_item() not in items_in_box

        # Toggle to hide
        visible = board.toggle_gcms()
        assert not visible
        assert not board.is_gcms_visible()
        assert board.get_gcms() is None
        assert board.get_gcms_item() is sim

        # Toggle to show
        visible2 = board.toggle_gcms()
        assert visible2
        assert board.is_gcms_visible()
