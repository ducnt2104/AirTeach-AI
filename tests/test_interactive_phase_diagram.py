"""
Unit tests for InteractivePhaseDiagramItem and ChemistryTools.create_phase_diagram_diagram.
Verifies thermodynamic P-T phase diagram calculations, Clausius-Clapeyron vapor pressure,
anomalous negative melting slope of ice (Water), CO2 dry ice sublimation at 1 atm,
supercritical fluid phase transitions, microscopic particle dynamics, touchless air gestures,
and lesson board lifecycle integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_phase_diagram import InteractivePhaseDiagramItem, PHASE_PRESETS
from src.ui.stem_tools import ChemistryTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_phase_diagram_initialization(qapp):
    item = InteractivePhaseDiagramItem()
    assert item.active_preset.code == "WATER_H2O_ANOMALOUS_MELTING"
    assert item.temperature_c == 25.0
    assert item.pressure_atm == 1.0
    assert item.current_phase == "LỎNG (LIQUID)"
    assert item.boundingRect().width() == InteractivePhaseDiagramItem.CARD_W
    assert item.boundingRect().height() == InteractivePhaseDiagramItem.CARD_H


def test_water_anomalous_negative_melting_slope(qapp):
    item = InteractivePhaseDiagramItem()
    assert item.active_preset.melting_slope_negative is True

    # At 1 atm, melting is ~ 0.0 °C
    tm_1atm = item.melting_temperature_at_p(1.0)
    # At 100 atm, melting temperature decreases due to negative slope (regelation)
    tm_100atm = item.melting_temperature_at_p(100.0)

    assert tm_100atm < tm_1atm
    assert -1.0 < tm_100atm < 0.0


def test_co2_dry_ice_sublimation_at_1atm(qapp):
    item = InteractivePhaseDiagramItem()
    item.load_preset("CARBON_DIOXIDE_DRY_ICE_SUBLIMATION")

    assert item.active_preset.p_triple_atm > 1.0  # 5.11 atm > 1 atm
    assert item.active_preset.melting_slope_negative is False

    # At 1 atm, -90 °C is Solid Dry Ice
    assert item.determine_phase(-90.0, 1.0) == "RẮN (SOLID)"
    # At 1 atm, -50 °C is Gas CO2 (sublimed directly from solid)
    assert item.determine_phase(-50.0, 1.0) == "KHÍ (GAS / VAPOR)"

    # Liquid CO2 can only exist when P > P_triple (5.11 atm)
    assert item.determine_phase(-40.0, 10.0) == "LỎNG (LIQUID)"


def test_clausius_clapeyron_vapor_pressure_monotonicity(qapp):
    item = InteractivePhaseDiagramItem()
    item.load_preset("WATER_H2O_ANOMALOUS_MELTING")

    p_25 = item.vapor_pressure_clausius_clapeyron(25.0)
    p_50 = item.vapor_pressure_clausius_clapeyron(50.0)
    p_100 = item.vapor_pressure_clausius_clapeyron(100.0)

    assert p_25 < p_50 < p_100
    # Normal boiling point of water is 100°C where P_vap ≈ 1.0 atm
    assert math.isclose(p_100, 1.0, rel_tol=0.15)


def test_supercritical_fluid_transition(qapp):
    item = InteractivePhaseDiagramItem()
    # Water supercritical: T >= 373.95°C and P >= 217.75 atm
    phase_water_sc = item.determine_phase(400.0, 250.0)
    assert phase_water_sc == "CHẤT LƯU SIÊU TỚI HẠN (SUPERCRITICAL FLUID)"

    # CO2 supercritical: T >= 31.1°C and P >= 72.8 atm
    item.load_preset("CARBON_DIOXIDE_DRY_ICE_SUBLIMATION")
    phase_co2_sc = item.determine_phase(40.0, 85.0)
    assert phase_co2_sc == "CHẤT LƯU SIÊU TỚI HẠN (SUPERCRITICAL FLUID)"


def test_all_five_presets_load(qapp):
    item = InteractivePhaseDiagramItem()
    preset_keys = [
        "WATER_H2O_ANOMALOUS_MELTING",
        "CARBON_DIOXIDE_DRY_ICE_SUBLIMATION",
        "ARGON_NOBLE_GAS_CLASSICAL",
        "SULFUR_ALLOTROPIC_POLYMORPHISM",
        "SUPERCRITICAL_EXTRACTION_INDUSTRIAL",
    ]
    for key in preset_keys:
        item.load_preset(key)
        assert item.active_preset.code == key
        assert len(item.active_preset.title_vi) > 0
        assert item.active_preset.t_critical_c > item.active_preset.t_triple_c
        assert item.active_preset.p_critical_atm > item.active_preset.p_triple_atm
        assert len(item.current_phase) > 0


def test_parameter_steppers_and_clamping(qapp):
    item = InteractivePhaseDiagramItem()
    initial_t = item.temperature_c
    item.step_temperature(10.0)
    assert item.temperature_c == initial_t + 10.0
    item.step_temperature(-20.0)
    assert item.temperature_c == initial_t - 10.0

    initial_p = item.pressure_atm
    item.step_pressure_log(1.0)  # multiply by 10
    assert math.isclose(item.pressure_atm, initial_p * 10.0, rel_tol=1e-3)

    # Clamping test
    t_min, t_max = item.active_preset.temp_range_c
    item.set_temperature_pressure(-9999.0, 1e12)
    assert item.temperature_c == t_min
    assert item.pressure_atm <= math.pow(10, item.active_preset.press_range_log[1])


def test_microscopic_particles_movement_by_phase(qapp):
    item = InteractivePhaseDiagramItem()
    assert len(item._particles) == 24

    # In solid phase (ice at -20°C, 1 atm)
    item.set_temperature_pressure(-20.0, 1.0)
    item._update_particles()
    # Particles should remain close to their lattice origin orig_x
    for p in item._particles:
        dist = math.hypot(p["x"] - p["orig_x"], p["y"] - p["orig_y"])
        assert dist < 10.0, "Solid particles must vibrate around lattice points"

    # In gas phase (steam at 150°C, 1 atm)
    item.set_temperature_pressure(150.0, 1.0)
    for _ in range(5):
        item._update_particles()
    # Particles disperse freely within bounding box
    for p in item._particles:
        assert 15.0 <= p["x"] <= 180.0
        assert 15.0 <= p["y"] <= 150.0


def test_automated_trajectory_modes(qapp):
    item = InteractivePhaseDiagramItem()
    item._handle_button_click("btn_mode")
    assert item._trajectory_mode == "ISOBARIC_HEATING"

    item._advance_trajectory()
    assert item.pressure_atm == 1.0  # Isobaric held at 1 atm

    item._handle_button_click("btn_mode")
    assert item._trajectory_mode == "ISOTHERMAL_COMPRESSION"

    item._advance_trajectory()
    assert item.temperature_c == 20.0  # Isothermal held at 20°C


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractivePhaseDiagramItem(pos=(100.0, 80.0))
    init_pos = item.pos()
    item.handle_air_drag(QPointF(30.0, -10.0))
    assert item.pos().x() == init_pos.x() + 30.0
    assert item.pos().y() == init_pos.y() - 10.0

    # Test air pinch on T+ button
    btn_t_up_rect = item._buttons.get("btn_t_up")
    assert btn_t_up_rect is not None
    init_t = item.temperature_c
    target_scene = item.mapToScene(btn_t_up_rect.center())
    handled = item.handle_air_pinch(target_scene)
    assert handled is True
    assert item.temperature_c == init_t + 10.0

    # Toggle animation
    init_anim = item._is_animating
    item.toggle_animation()
    assert item._is_animating != init_anim


def test_vector_diagram_generation_and_stamping(qapp):
    items = ChemistryTools.create_phase_diagram_diagram(
        center=(340.0, 190.0),
        width=680.0,
        height=380.0,
        preset_code="WATER_H2O_ANOMALOUS_MELTING",
        substance_name="Nước",
        formula="H₂O",
        temp_c=25.0,
        press_atm=1.0,
        current_phase="LỎNG (LIQUID)",
        t_triple=0.01,
        p_triple=0.006,
        t_crit=373.95,
        p_crit=217.75,
    )
    assert len(items) >= 12
    bg = items[0]
    assert bg.rect().width() == 680.0
    assert bg.rect().height() == 380.0

    # Test stamp signal emission
    item = InteractivePhaseDiagramItem()
    stamped = []
    item.stamp_requested.connect(lambda data: stamped.append(data))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert stamped[0]["formula"] == "H₂O"
    assert stamped[0]["preset_code"] == "WATER_H2O_ANOMALOUS_MELTING"


def test_lesson_board_phase_diagram_lifecycle(qapp):
    board = LessonBoard()
    assert not board.is_phase_diagram_visible()
    assert board.get_phase_diagram() is None

    # Show item
    item = board.show_phase_diagram(True, pos=(120.0, 80.0))
    assert item is not None
    assert board.is_phase_diagram_visible()
    assert board.get_phase_diagram() == item
    assert board.get_phase_diagram_item() == item

    # Toggle off
    res = board.toggle_phase_diagram()
    assert not res
    assert not board.is_phase_diagram_visible()
    assert board.get_phase_diagram() is None
    assert board.get_phase_diagram_item() is not None

    # Toggle on
    res2 = board.toggle_phase_diagram()
    assert res2
    assert board.is_phase_diagram_visible()
