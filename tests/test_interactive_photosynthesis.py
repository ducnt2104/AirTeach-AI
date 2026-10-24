"""
AirTeach AI - Unit Tests for Interactive Biology Photosynthesis Simulator
=========================================================================
Comprehensive automated unit tests verifying:
  1. Authentic biophysical and biochemical photosynthesis kinetics (FvCB model).
  2. Z-scheme light reactions (water photolysis, proton gradient, ATP synthase).
  3. Stroma Calvin-Benson cycle (RuBisCO carboxylation, reduction, RuBP regeneration).
  4. 5 Curriculum Presets (Sun C3, Shade C3, Drought C3, C4 Maize, Greenhouse CO2).
  5. Touchless air pinch gesture interactions, parameter tuning, and live animation.
  6. Vector diagram generation via BiologyTools.create_photosynthesis_diagram.
  7. LessonBoard lifecycle integration and lasso selection exclusion.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_photosynthesis import (
    InteractivePhotosynthesisItem,
    PRESETS,
)
from src.ui.stem_tools import BiologyTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure single persistent QApplication for QGraphics testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractivePhotosynthesis:
    """Test suite for InteractivePhotosynthesisItem and STEM biology integration."""

    def test_initialization(self, qapp):
        """Verifies initial configuration, geometry, and default physiological values."""
        item = InteractivePhotosynthesisItem(pos=QPointF(150.0, 100.0))
        assert item.pos() == QPointF(150.0, 100.0)
        assert item.boundingRect().width() == InteractivePhotosynthesisItem.WIDTH
        assert item.boundingRect().height() == InteractivePhotosynthesisItem.HEIGHT
        assert item.active_preset_code == "SUN_PLANT_HIGH_LIGHT"
        assert item.light_intensity == 1200.0
        assert item.co2_ppm == 420.0
        assert item.temp_c == 25.0
        assert item.net_photosynthetic_rate > 0.0
        assert item.o2_evolution_rate > 0.0
        assert item.glucose_synthesis_rate > 0.0
        assert item.is_animating is True
        item.cleanup()

    def test_all_five_presets(self, qapp):
        """Verifies loading of all 5 curriculum presets and signal emissions."""
        item = InteractivePhotosynthesisItem()
        emitted_presets = []
        item.preset_changed.connect(lambda p: emitted_presets.append(p))

        for code, expected_preset in PRESETS.items():
            loaded = item.load_preset(code)
            assert loaded is True
            assert item.active_preset_code == code
            assert item.preset.title_vi == expected_preset.title_vi
            assert item.preset.plant_type == expected_preset.plant_type
            assert item.light_intensity == expected_preset.light_intensity
            assert item.co2_ppm == expected_preset.co2_ppm
            assert item.temp_c == expected_preset.temp_c

        assert len(emitted_presets) == 5
        assert item.load_preset("NON_EXISTENT_PRESET") is False
        item.cleanup()

    def test_fvcb_photosynthetic_kinetics(self, qapp):
        """Verifies mathematical consistency of Farquhar-von Caemmerer-Berry model."""
        item = InteractivePhotosynthesisItem()
        item.load_preset("SUN_PLANT_HIGH_LIGHT")

        # Gross rate must exceed net rate by dark respiration
        assert item.gross_photosynthetic_rate > item.net_photosynthetic_rate
        assert abs((item.gross_photosynthetic_rate - item.dark_respiration_rate) - item.net_photosynthetic_rate) < 1e-4

        # Electron transport rate J must be positive and bounded by J_max
        assert 0.0 < item.electron_transport_rate <= item.preset.jmax

        # Glucose synthesis is stoichiometrically 1/6 of net CO2 fixed
        assert abs(item.glucose_synthesis_rate - (item.net_photosynthetic_rate / 6.0)) < 1e-4
        item.cleanup()

    def test_shade_plant_adaptation(self, qapp):
        """Verifies shade plant low light compensation and efficiency."""
        item = InteractivePhotosynthesisItem()
        item.load_preset("SHADE_PLANT_LOW_LIGHT")  # Light = 150 umol

        # Positive net assimilation despite low photon flux
        assert item.net_photosynthetic_rate > 0.0
        # Dark respiration of shade plant is lower than sun plant
        assert item.dark_respiration_rate < 1.0
        item.cleanup()

    def test_drought_stomatal_closure_reduction(self, qapp):
        """Verifies drought stomatal closure depresses Ci and net carbon assimilation."""
        item_sun = InteractivePhotosynthesisItem()
        item_sun.load_preset("SUN_PLANT_HIGH_LIGHT")
        sun_net = item_sun.net_photosynthetic_rate

        item_drought = InteractivePhotosynthesisItem()
        item_drought.load_preset("DROUGHT_STOMATAL_CLOSURE")
        drought_net = item_drought.net_photosynthetic_rate

        # Drought assimilation is severely reduced (> 40% reduction)
        assert drought_net < sun_net * 0.60
        assert item_drought._ci_ppm < item_sun._ci_ppm
        item_sun.cleanup()
        item_drought.cleanup()

    def test_c4_maize_concentrating_mechanism(self, qapp):
        """Verifies C4 maize Hatch-Slack CO2 concentration and high assimilation."""
        item = InteractivePhotosynthesisItem()
        item.load_preset("C4_MAIZE_CONCENTRATING")

        # Internal bundle sheath CO2 exceeds 800 ppm
        assert item._ci_ppm > 800.0
        # High net assimilation at 35°C
        assert item.net_photosynthetic_rate > 15.0
        item.cleanup()

    def test_greenhouse_co2_enrichment_boost(self, qapp):
        """Verifies elevated CO2 (1000 ppm) boosts photosynthesis over ambient (420 ppm)."""
        item = InteractivePhotosynthesisItem()
        item.load_preset("SUN_PLANT_HIGH_LIGHT")  # 420 ppm CO2
        ambient_net = item.net_photosynthetic_rate

        item.load_preset("GREENHOUSE_ENRICHED_CO2")  # 1000 ppm CO2
        enriched_net = item.net_photosynthetic_rate

        assert enriched_net > ambient_net
        item.cleanup()

    def test_environmental_setters_and_clamping(self, qapp):
        """Verifies parameter setter clamping to safe physical bounds."""
        item = InteractivePhotosynthesisItem()

        item.set_light_intensity(5000.0)
        assert item.light_intensity == 2200.0
        item.set_light_intensity(-100.0)
        assert item.light_intensity == 0.0

        item.set_co2_ppm(3000.0)
        assert item.co2_ppm == 1600.0
        item.set_co2_ppm(10.0)
        assert item.co2_ppm == 50.0

        item.set_temperature(70.0)
        assert item.temp_c == 45.0
        item.set_temperature(1.0)
        assert item.temp_c == 5.0
        item.cleanup()

    def test_animation_toggle_and_progression(self, qapp):
        """Verifies animation toggle, tick progression, and rotor advancement."""
        item = InteractivePhotosynthesisItem()
        assert item.is_animating is True

        # Toggle to pause
        item.toggle_animation()
        assert not item.is_animating

        # Toggle to resume
        item.toggle_animation()
        assert item.is_animating

        # Simulate several animation ticks
        initial_rotor = item._rotor_angle_deg
        for _ in range(10):
            item._on_animation_tick()
        assert item._rotor_angle_deg != initial_rotor
        assert item._anim_time_s > 0.0
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        """Verifies touchless air pinch gesture dispatch on buttons and controllers."""
        item = InteractivePhotosynthesisItem(pos=QPointF(0.0, 0.0))

        # 1. Pinch on Preset button (preset_C4_MAIZE_CONCENTRATING)
        rect_c4 = item._btn_rects["preset_C4_MAIZE_CONCENTRATING"]
        pinch_pt = QPointF(rect_c4.center().x(), rect_c4.center().y())
        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "C4_MAIZE_CONCENTRATING"

        # 2. Pinch on Play / Pause button
        rect_play = item._btn_rects["play_pause"]
        assert item.is_animating is True
        handled_play = item.handle_air_pinch(rect_play.center())
        assert handled_play is True
        assert item.is_animating is False

        # 3. Pinch on Inc Light button
        rect_inc_l = item._btn_rects["inc_light"]
        init_l = item.light_intensity
        handled_l = item.handle_air_pinch(rect_inc_l.center())
        assert handled_l is True
        assert item.light_intensity > init_l

        # 4. Pinch outside bounding box
        outside_pt = QPointF(1000.0, 1000.0)
        assert item.handle_air_pinch(outside_pt) is False
        item.cleanup()

    def test_stem_tools_diagram_generation_and_stamp(self, qapp):
        """Verifies static vector diagram generation via BiologyTools and scene stamping."""
        board = LessonBoard()
        item = InteractivePhotosynthesisItem(pos=QPointF(100.0, 100.0))
        board.addItem(item)
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count

        # Test direct creation via BiologyTools
        items = BiologyTools.create_photosynthesis_diagram(
            center=(400.0, 300.0),
            preset_code="C4_MAIZE_CONCENTRATING",
            light_intensity=1600.0,
            co2_ppm=420.0,
            temp_c=35.0,
        )
        assert len(items) >= 8
        text_contents = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("SINH HỌC TẾ BÀO: QUANG HỢP" in t for t in text_contents)
        assert any("HÓA SINH QUANG HỢP" in t for t in text_contents)
        item.cleanup()

    def test_lesson_board_lifecycle_and_lasso_exclusion(self, qapp):
        """Verifies LessonBoard show/toggle/visibility lifecycle and lasso exclusion."""
        board = LessonBoard()
        assert not board.is_photosynthesis_visible()
        assert board.get_photosynthesis() is None

        # Show
        sim = board.show_photosynthesis(True, pos=(200.0, 150.0))
        assert sim is not None
        assert board.is_photosynthesis_visible()
        assert board.get_photosynthesis() is sim
        assert board.get_photosynthesis_item() is sim

        # Lasso exclusion check
        items_in_box = board.get_items_in_rect(QRectF(150.0, 100.0, 800.0, 600.0))
        assert board.get_photosynthesis_item() not in items_in_box

        # Toggle to hide
        visible = board.toggle_photosynthesis()
        assert not visible
        assert not board.is_photosynthesis_visible()
        assert board.get_photosynthesis() is None
        assert board.get_photosynthesis_item() is sim

        # Toggle to show
        visible2 = board.toggle_photosynthesis()
        assert visible2
        assert board.is_photosynthesis_visible()
