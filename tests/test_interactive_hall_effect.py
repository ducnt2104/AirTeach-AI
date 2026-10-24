"""
AirTeach AI — Unit Tests for Interactive Solid State Physics & Semiconductor Electronics:
Hall Effect, Lorentz Drift Velocity, Carrier Density & Hall Voltage Simulator (Milestone 147 / Task I.134).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_hall_effect import (
    InteractiveHallEffectItem,
    HallPreset,
    HALL_PRESETS,
    E_CHARGE_C,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_hall_presets_and_physical_parameters():
    """Verifies that all 5 curriculum presets are properly defined with physical/semiconductor parameters."""
    expected_codes = [
        "INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR",
        "GERMANIUM_P_TYPE_SEMICONDUCTOR",
        "COPPER_THIN_FOIL_METAL",
        "GALLIUM_ARSENIDE_HEMT_2DEG",
        "SILICON_N_TYPE_INTEGRATED_SENSOR",
    ]
    for code in expected_codes:
        assert code in HALL_PRESETS
        preset = HALL_PRESETS[code]
        assert preset.carrier_density_m3 > 0.0
        assert preset.default_thickness_um > 0.0
        assert preset.default_current_mA > 0.0
        assert preset.default_b_field_T > 0.0
        assert preset.carrier_mobility_m2_Vs > 0.0
        assert preset.carrier_charge_sign in (-1, +1)
        assert preset.accent_color.isValid()


def test_exact_hall_voltage_and_sign_convention(qapp):
    """
    Verifies analytical Hall voltage formula:
    V_H = (I * B) / (n * q * d)
    and sign distinction: V_H < 0 for electrons (n-type), V_H > 0 for holes (p-type).
    """
    item = InteractiveHallEffectItem()

    # 1. InSb n-type sensor (electrons: q = -e)
    item.set_preset("INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR")
    i_val = 0.020        # 20 mA = 0.020 A
    b_val = 0.50         # 0.50 T
    n_val = 2.0e22       # m^-3
    d_val = 10.0e-6      # 10 µm
    q_val = -E_CHARGE_C  # -1.602176634e-19 C

    expected_vh_insb = (i_val * b_val) / (n_val * q_val * d_val)
    assert math.isclose(item.hall_voltage_volts, expected_vh_insb, rel_tol=1e-4)
    assert item.hall_voltage_volts < 0.0  # Must be negative for electrons

    # 2. Germanium p-type semiconductor (holes: q = +e)
    item.set_preset("GERMANIUM_P_TYPE_SEMICONDUCTOR")
    i_ge = 0.015         # 15 mA
    b_ge = 0.40         # 0.40 T
    n_ge = 5.0e21       # m^-3
    d_ge = 50.0e-6      # 50 µm
    q_ge = +E_CHARGE_C  # +1.602176634e-19 C

    expected_vh_ge = (i_ge * b_ge) / (n_ge * q_ge * d_ge)
    assert math.isclose(item.hall_voltage_volts, expected_vh_ge, rel_tol=1e-4)
    assert item.hall_voltage_volts > 0.0  # Must be positive for holes


def test_hall_coefficient_and_carrier_type_inversion(qapp):
    """
    Verifies Hall coefficient R_H = 1 / (n * q).
    Toggling carrier type must flip the sign of R_H and V_H.
    """
    item = InteractiveHallEffectItem()
    item.set_preset("INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR")

    rh_initial = item.hall_coefficient_m3_C
    vh_initial = item.hall_voltage_volts
    assert rh_initial < 0.0

    # Flip carrier type
    item.toggle_carrier_type()
    rh_flipped = item.hall_coefficient_m3_C
    vh_flipped = item.hall_voltage_volts

    assert math.isclose(rh_flipped, -rh_initial, rel_tol=1e-5)
    assert math.isclose(vh_flipped, -vh_initial, rel_tol=1e-5)


def test_b_field_inversion_and_scaling(qapp):
    """
    Verifies V_H scaling laws:
    - V_H is linear with B (inverting B flips sign)
    - V_H is linear with I (doubling I doubles V_H)
    - V_H is inversely proportional to d (doubling d cuts V_H in half)
    """
    item = InteractiveHallEffectItem()
    item.set_preset("INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR")
    base_vh = item.hall_voltage_volts

    # Invert B field
    item.invert_b_field()
    assert math.isclose(item.hall_voltage_volts, -base_vh, rel_tol=1e-4)

    # Restore B field
    item.invert_b_field()

    # Double current I
    item.set_current_mA(item._current_mA * 2.0)
    assert math.isclose(item.hall_voltage_volts, base_vh * 2.0, rel_tol=1e-4)

    # Reset defaults and double thickness d
    item.reset_defaults()
    base_vh = item.hall_voltage_volts
    item.set_thickness_um(item._thickness_um * 2.0)
    assert math.isclose(item.hall_voltage_volts, base_vh * 0.5, rel_tol=1e-4)


def test_carrier_drift_velocity_and_lorentz_force(qapp):
    """
    Verifies drift velocity v_d = J / (n * |q|) and Lorentz force F_B = |q| * v_d * |B|.
    Compares extreme difference between metal (copper foil, tiny v_d)
    and semiconductor (InSb, huge v_d).
    """
    item = InteractiveHallEffectItem()

    # InSb semiconductor
    item.set_preset("INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR")
    v_insb = item.drift_velocity_m_s
    f_insb = item.lorentz_force_newtons
    assert v_insb > 100.0  # Drift velocity ~124.8 m/s
    assert math.isclose(f_insb, 1.0e-17, rel_tol=1e-3)

    # Copper foil metal
    item.set_preset("COPPER_THIN_FOIL_METAL")
    v_cu = item.drift_velocity_m_s
    f_cu = item.lorentz_force_newtons
    # In copper, huge n (~8.5e28) causes v_d to be microscopic (~1e-4 m/s)
    assert v_cu < 1e-3
    assert v_cu < v_insb * 1e-5


def test_interactive_controls_and_buttons(qapp):
    """Verifies button handlers and parameter adjustments."""
    item = InteractiveHallEffectItem()
    item.set_preset("INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR")

    # B field stepping
    initial_b = item._b_field_T
    item._handle_button_click("b_inc")
    assert math.isclose(item._b_field_T, initial_b + 0.10, abs_tol=1e-3)
    item._handle_button_click("b_dec")
    assert math.isclose(item._b_field_T, initial_b, abs_tol=1e-3)

    # Current stepping
    initial_i = item._current_mA
    item._handle_button_click("i_inc")
    assert math.isclose(item._current_mA, initial_i + 10.0, abs_tol=1e-3)

    # Thickness stepping
    initial_d = item._thickness_um
    item._handle_button_click("d_inc")
    assert math.isclose(item._thickness_um, initial_d + 5.0, abs_tol=1e-3)

    # Reset
    item._handle_button_click("reset")
    assert math.isclose(item._b_field_T, item._preset.default_b_field_T)
    assert math.isclose(item._current_mA, item._preset.default_current_mA)
    assert math.isclose(item._thickness_um, item._preset.default_thickness_um)


def test_stem_tools_hall_diagram():
    """Verifies vector diagram creation in PhysicsTools for Hall Effect."""
    items = PhysicsTools.create_hall_effect_diagram(
        center=(400.0, 300.0),
        preset_code="INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR",
        material_name="Indium Antimonide (InSb n-type)",
        carrier_type="electron",
        current_mA=20.0,
        b_field_T=0.50,
        thickness_um=10.0,
        carrier_density_m3=2.0e22,
        hall_voltage_V=-0.0156,
        hall_coefficient_m3_C=-3.12e-4,
        drift_velocity_m_s=1248.0,
    )
    assert len(items) >= 15
    types = [type(itm).__name__ for itm in items]
    assert "QGraphicsRectItem" in types
    assert "QGraphicsTextItem" in types
    assert "QGraphicsPolygonItem" in types


def test_lesson_board_hall_lifecycle(qapp):
    """Verifies integration, toggling, and stamping on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_hall_effect_visible()

    # Show
    item = board.show_hall_effect(True)
    assert item is not None
    assert board.is_hall_effect_visible()
    assert board.get_hall_effect() is item

    # Toggle off
    state = board.toggle_hall_effect()
    assert not state
    assert not board.is_hall_effect_visible()
    assert board.get_hall_effect() is None

    # Toggle on
    state = board.toggle_hall_effect()
    assert state
    assert board.is_hall_effect_visible()

    # Test stamp
    initial_count = len(board.items())
    board._on_stamp_hall_effect({
        "preset_code": "INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR",
        "material_name": "InSb",
        "carrier_type": "electron",
        "current_mA": 20.0,
        "b_field_T": 0.50,
        "thickness_um": 10.0,
        "carrier_density_m3": 2.0e22,
        "hall_voltage_V": -0.0156,
        "hall_coefficient_m3_C": -3.12e-4,
        "drift_velocity_m_s": 1248.0,
    })
    after_count = len(board.items())
    assert after_count > initial_count + 10


def test_paint_rendering_zero_crash(qapp):
    """Ensures QPainter rendering executes without raising any exception."""
    item = InteractiveHallEffectItem()
    image = QImage(700, 430, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    assert not image.isNull()
