"""
AirTeach AI — Unit Tests for Applied Electromagnetism & Modern Physics:
Mass Spectrometer, Wien Velocity Selector & Isotopic Deflection Simulator (Milestone 145 / Task I.132).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_mass_spectrometer import (
    InteractiveMassSpectrometerItem,
    MassSpectrometerPreset,
    SPECTROMETER_PRESETS,
    AMU_KG,
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


def test_spectrometer_presets_and_parameters():
    """Verifies that all 5 curriculum presets are properly defined with physical parameters."""
    expected_codes = [
        "CHLORINE_ISOTOPE_35_37",
        "CARBON_12_13_GEOCHEMISTRY",
        "URANIUM_235_238_CALUTRON",
        "HYDROGEN_DEUTERIUM_HEAVY_WATER",
        "LITHIUM_6_7_BATTERY_TECH",
    ]
    for code in expected_codes:
        assert code in SPECTROMETER_PRESETS
        preset = SPECTROMETER_PRESETS[code]
        assert preset.default_electric_field_V_m > 0.0
        assert preset.default_b1_Tesla > 0.0
        assert preset.default_b2_Tesla > 0.0
        assert len(preset.isotopes) >= 2
        total_abundance = sum(iso.abundance_pct for iso in preset.isotopes)
        assert math.isclose(total_abundance, 100.0, abs_tol=0.1)


def test_wien_velocity_selector_filtering(qapp):
    """
    Verifies Wien cross-field velocity selector condition:
    v_0 = E / B_1
    """
    item = InteractiveMassSpectrometerItem()
    item.set_preset("CHLORINE_ISOTOPE_35_37")
    e_val = item._e_field_V_m  # 50,000 V/m
    b1_val = item._b1_Tesla     # 0.20 T

    expected_v0 = e_val / b1_val  # 250,000 m/s
    assert math.isclose(item.selected_velocity_m_s, expected_v0, rel_tol=1e-4)


def test_semicircular_deflection_radius_mass_proportionality(qapp):
    """
    Verifies that deflection radius inside magnetic chamber B_2:
    r = (m * v_0) / (q * B_2)
    is strictly proportional to isotope mass m.
    For Chlorine-35 and Chlorine-37, the radius ratio equals the mass ratio.
    """
    item = InteractiveMassSpectrometerItem()
    item.set_preset("CHLORINE_ISOTOPE_35_37")
    iso35 = item._preset.isotopes[0]
    iso37 = item._preset.isotopes[1]

    r35 = item.deflection_radius_m(iso35)
    r37 = item.deflection_radius_m(iso37)

    # Theoretical values: (m_u * AMU_KG * v0) / (q * E_CHARGE_C * B2)
    v0 = item.selected_velocity_m_s
    b2 = item._b2_Tesla
    expected_r35 = (iso35.mass_u * AMU_KG * v0) / (iso35.charge_e * E_CHARGE_C * b2)
    expected_r37 = (iso37.mass_u * AMU_KG * v0) / (iso37.charge_e * E_CHARGE_C * b2)

    assert math.isclose(r35, expected_r35, rel_tol=1e-4)
    assert math.isclose(r37, expected_r37, rel_tol=1e-4)

    # Mass proportionality ratio
    ratio_r = r37 / r35
    ratio_m = iso37.mass_u / iso35.mass_u
    assert math.isclose(ratio_r, ratio_m, rel_tol=1e-4)


def test_hydrogen_deuterium_mass_doubling(qapp):
    """
    Verifies that Deuterium (2H) trajectory radius is approximately double
    that of regular Hydrogen (1H).
    """
    item = InteractiveMassSpectrometerItem()
    item.set_preset("HYDROGEN_DEUTERIUM_HEAVY_WATER")
    iso_h = item._preset.isotopes[0]
    iso_d = item._preset.isotopes[1]

    rH = item.deflection_radius_mm(iso_h)
    rD = item.deflection_radius_mm(iso_d)

    assert math.isclose(rD / rH, iso_d.mass_u / iso_h.mass_u, rel_tol=1e-4)
    assert math.isclose(rD / rH, 2.0, rel_tol=0.01)


def test_detector_separation_distance(qapp):
    """
    Verifies impact diameter D = 2 * r and isotope separation ΔD = 2 * (r_2 - r_1).
    """
    item = InteractiveMassSpectrometerItem()
    item.set_preset("URANIUM_235_238_CALUTRON")
    iso235 = item._preset.isotopes[0]
    iso238 = item._preset.isotopes[1]

    d235 = item.detector_impact_distance_mm(iso235)
    d238 = item.detector_impact_distance_mm(iso238)

    assert d238 > d235
    assert math.isclose(d235, 2.0 * item.deflection_radius_mm(iso235), rel_tol=1e-4)
    assert math.isclose(d238, 2.0 * item.deflection_radius_mm(iso238), rel_tol=1e-4)


def test_interactive_controls_and_buttons(qapp):
    """Verifies button adjustments for E-field, B2-field, and presets."""
    item = InteractiveMassSpectrometerItem()
    item.set_preset("CHLORINE_ISOTOPE_35_37")
    init_e = item._e_field_V_m
    init_b = item._b2_Tesla

    # E-field adjustments: 1.10 * 0.90 = 0.99
    item._on_button_clicked("btn_e_up")
    assert item._e_field_V_m > init_e
    item._on_button_clicked("btn_e_down")
    assert math.isclose(item._e_field_V_m, init_e * 1.10 * 0.90, rel_tol=1e-4)

    # B2-field adjustments: 1.10 * 0.90 = 0.99
    item._on_button_clicked("btn_b_up")
    assert item._b2_Tesla > init_b
    item._on_button_clicked("btn_b_down")
    assert math.isclose(item._b2_Tesla, init_b * 1.10 * 0.90, rel_tol=1e-4)

    # Preset cycle
    item._on_button_clicked("btn_preset")
    assert item._active_preset_code != "CHLORINE_ISOTOPE_35_37"

    # Reset
    item._on_button_clicked("btn_reset")
    assert math.isclose(item._e_field_V_m, item._preset.default_electric_field_V_m, rel_tol=1e-4)


def test_stamping_and_stem_tools_vector_generation(qapp):
    """Verifies that PhysicsTools.create_mass_spectrometer_diagram generates valid vector items."""
    diagram_items = PhysicsTools.create_mass_spectrometer_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="CHLORINE_ISOTOPE_35_37",
        element_name="Clo (Chlorine)",
        e_field_V_m=50000.0,
        b1_Tesla=0.20,
        b2_Tesla=0.30,
        velocity_m_s=250000.0,
        iso0_symbol="³⁵Cl⁺",
        iso0_radius_mm=45.3,
        iso1_symbol="³⁷Cl⁺",
        iso1_radius_mm=47.9,
    )
    assert len(diagram_items) >= 20

    # Test stamp signal emission
    item = InteractiveMassSpectrometerItem()
    emitted = []
    item.stamp_requested.connect(lambda d: emitted.append(d))
    item.request_stamp()
    assert len(emitted) == 1
    assert "velocity_m_s" in emitted[0]
    assert "iso0_radius_mm" in emitted[0]
    assert "iso1_radius_mm" in emitted[0]


def test_lesson_board_integration(qapp):
    """Verifies lifecycle methods on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_mass_spectrometer_visible()
    assert board.get_mass_spectrometer() is None

    # Show
    item = board.show_mass_spectrometer(True, pos=(110.0, 45.0))
    assert item is not None
    assert board.is_mass_spectrometer_visible()
    assert board.get_mass_spectrometer() is item

    # Toggle hide
    is_vis = board.toggle_mass_spectrometer()
    assert not is_vis
    assert not board.is_mass_spectrometer_visible()

    # Toggle show
    is_vis = board.toggle_mass_spectrometer()
    assert is_vis
    assert board.is_mass_spectrometer_visible()

    # Stamping
    cnt_before = len(board.items())
    board._on_stamp_mass_spectrometer({
        "preset_code": "CHLORINE_ISOTOPE_35_37",
        "element_name": "Clo (Chlorine)",
        "e_field_V_m": 50000.0,
        "b1_Tesla": 0.20,
        "b2_Tesla": 0.30,
        "velocity_m_s": 250000.0,
        "iso0_symbol": "³⁵Cl⁺",
        "iso0_radius_mm": 45.3,
        "iso1_symbol": "³⁷Cl⁺",
        "iso1_radius_mm": 47.9,
    })
    assert len(board.items()) > cnt_before


def test_paint_render_pipeline(qapp):
    """Verifies that the item can paint onto a QImage without throwing exceptions."""
    item = InteractiveMassSpectrometerItem()
    image = QImage(800, 700, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    item.paint(painter, None, None)
    painter.end()
    assert not image.isNull()
