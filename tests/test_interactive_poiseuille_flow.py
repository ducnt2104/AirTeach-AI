"""
AirTeach AI — Unit Tests for Interactive Fluid Dynamics & Aerodynamics:
Poiseuille Flow, Hagen-Poiseuille Viscous Tube Flow & Reynolds Number Turbulence Simulator (Milestone 142 / Task I.129).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_poiseuille_flow import (
    InteractivePoiseuilleFlowItem,
    PoiseuillePreset,
    POISEUILLE_PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_poiseuille_presets_and_physical_constants():
    """Verifies that all 5 curriculum presets are properly defined with physical constants."""
    expected_codes = [
        "BLOOD_ARTERIOLE_VASODILATION",
        "WATER_CAPILLARY_LAMINAR",
        "GLYCERIN_HIGH_VISCOSITY_CREEPING",
        "TURBULENT_WATER_PIPE_BURST",
        "AIR_RESPIRATORY_BRONCHIOLE",
    ]
    for code in expected_codes:
        assert code in POISEUILLE_PRESETS
        preset = POISEUILLE_PRESETS[code]
        assert preset.density_kg_per_m3 > 0.0
        assert preset.viscosity_mPa_s > 0.0
        assert preset.default_radius_mm > 0.0
        assert preset.default_delta_p_Pa > 0.0
        assert preset.pipe_length_m > 0.0
        assert len(preset.title_vi) > 0
        assert len(preset.fluid_name) > 0


def test_parabolic_velocity_profile_and_no_slip(qapp):
    """
    Verifies exact Navier-Stokes analytical solution for laminar pipe flow:
    v(r) = v_max * (1 - r^2 / R^2)
    1. At r = 0, v = v_max
    2. At r = ±R, v = 0 (no-slip boundary condition)
    3. At r = ±R/2, v = 0.75 * v_max
    """
    item = InteractivePoiseuilleFlowItem()
    item.set_preset("WATER_CAPILLARY_LAMINAR")
    v_max = item.max_velocity_m_per_s
    assert v_max > 0.0

    # Centerline r = 0
    v_center = item.local_velocity(0.0)
    assert math.isclose(v_center, v_max, rel_tol=1e-4)

    # Tube wall r = ±1.0
    v_wall_top = item.local_velocity(1.0)
    v_wall_bot = item.local_velocity(-1.0)
    assert math.isclose(v_wall_top, 0.0, abs_tol=1e-6)
    assert math.isclose(v_wall_bot, 0.0, abs_tol=1e-6)

    # Halfway radius r = ±0.5 => 1 - 0.25 = 0.75 v_max
    v_half_pos = item.local_velocity(0.5)
    v_half_neg = item.local_velocity(-0.5)
    assert math.isclose(v_half_pos, 0.75 * v_max, rel_tol=1e-4)
    assert math.isclose(v_half_neg, 0.75 * v_max, rel_tol=1e-4)


def test_hagen_poiseuille_fourth_power_radius_law(qapp):
    """
    Verifies the fundamental 4th-power radius law: Q = (π * R^4 * ΔP) / (8 * η * L).
    Doubling the radius (R -> 2R) increases flow rate by exactly 2^4 = 16 times.
    """
    item = InteractivePoiseuilleFlowItem()
    item.set_preset("WATER_CAPILLARY_LAMINAR")
    base_r = item._radius_mm
    q_base = item.volumetric_flow_rate_m3_per_s

    # Double the radius
    item.set_radius_mm(base_r * 2.0)
    q_double = item.volumetric_flow_rate_m3_per_s

    # In laminar regime, ratio is exactly 16
    assert math.isclose(q_double / q_base, 16.0, rel_tol=0.05)


def test_hydraulic_resistance_ohms_law_analogy(qapp):
    """
    Verifies fluid dynamic Ohm's law: ΔP = Q * R_h.
    R_h = (8 * η * L) / (π * R^4)
    """
    item = InteractivePoiseuilleFlowItem()
    item.set_preset("GLYCERIN_HIGH_VISCOSITY_CREEPING")
    dp = item._delta_p_Pa
    q = item.volumetric_flow_rate_m3_per_s
    rh = item.hydraulic_resistance_Pa_s_per_m3

    computed_dp = q * rh
    assert math.isclose(computed_dp, dp, rel_tol=1e-4)


def test_average_velocity_ratio_in_laminar_flow(qapp):
    """
    Verifies that for laminar pipe flow, the cross-sectional average velocity
    is precisely half of the maximum centerline velocity: v_avg = v_max / 2.
    """
    item = InteractivePoiseuilleFlowItem()
    item.set_preset("WATER_CAPILLARY_LAMINAR")
    assert item.reynolds_number < 2000.0  # Laminar
    v_max = item.max_velocity_m_per_s
    v_avg = item.average_velocity_m_per_s

    assert math.isclose(v_avg, v_max / 2.0, rel_tol=1e-4)


def test_reynolds_number_and_turbulent_transition(qapp):
    """
    Verifies Reynolds number calculations Re = (ρ * v_avg * 2R) / η
    and flow regime classification (Laminar < 2000, Turbulent > 4000).
    """
    item = InteractivePoiseuilleFlowItem()

    # Laminar preset
    item.set_preset("WATER_CAPILLARY_LAMINAR")
    assert item.reynolds_number < 2000.0
    regime, _, _ = item.flow_regime_label
    assert regime == "LAMINAR"

    # High-speed water pipe preset
    item.set_preset("TURBULENT_WATER_PIPE_BURST")
    assert item.reynolds_number > 4000.0
    regime, _, _ = item.flow_regime_label
    assert regime == "TURBULENT"


def test_wall_shear_stress(qapp):
    """
    Verifies wall shear stress τ_w = (ΔP * R) / (2 * L).
    """
    item = InteractivePoiseuilleFlowItem()
    item.set_preset("WATER_CAPILLARY_LAMINAR")
    r_m = item.radius_m
    length = item._pipe_length_m
    dp = item._delta_p_Pa
    expected_tau_w = (dp * r_m) / (2.0 * length)

    assert math.isclose(item.wall_shear_stress_Pa, expected_tau_w, rel_tol=1e-4)


def test_interactive_controls_and_buttons(qapp):
    """Verifies button clicks and parameter adjustments."""
    item = InteractivePoiseuilleFlowItem()
    item.set_preset("WATER_CAPILLARY_LAMINAR")
    initial_r = item._radius_mm
    initial_p = item._delta_p_Pa

    # Radius step up and down
    item._on_button_clicked("btn_r_up")
    assert item._radius_mm > initial_r
    item._on_button_clicked("btn_r_down")
    assert math.isclose(item._radius_mm, initial_r * 1.10 * 0.90, rel_tol=1e-4)

    # Pressure step up and down
    item._on_button_clicked("btn_p_up")
    assert item._delta_p_Pa > initial_p
    item._on_button_clicked("btn_p_down")
    assert math.isclose(item._delta_p_Pa, initial_p * 1.20 * 0.80, rel_tol=1e-4)

    # Preset cycle
    item._on_button_clicked("btn_preset")
    assert item._active_preset_code != "WATER_CAPILLARY_LAMINAR"

    # Reset
    item._on_button_clicked("btn_reset")
    assert math.isclose(item._radius_mm, item._preset.default_radius_mm, rel_tol=1e-4)


def test_stamping_and_stem_tools_vector_generation(qapp):
    """Verifies that PhysicsTools.create_poiseuille_flow_diagram creates valid vector items."""
    items = PhysicsTools.create_poiseuille_flow_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="WATER_CAPILLARY_LAMINAR",
        fluid_name="Nước Cất ở 20°C (Water)",
        radius_mm=0.5,
        delta_p_Pa=1200.0,
        viscosity_mPa_s=1.002,
        pipe_length_m=0.20,
        flow_rate_mL_per_s=1.25,
        max_velocity_m_per_s=0.75,
        average_velocity_m_per_s=0.375,
        reynolds_number=450.0,
        flow_regime="LAMINAR",
    )
    assert len(items) >= 20

    # Test stamp signal emission
    item = InteractivePoiseuilleFlowItem()
    emitted = []
    item.stamp_requested.connect(lambda d: emitted.append(d))
    item.request_stamp()
    assert len(emitted) == 1
    assert "reynolds_number" in emitted[0]
    assert "flow_rate_mL_per_s" in emitted[0]
    assert "max_velocity_m_per_s" in emitted[0]


def test_lesson_board_integration(qapp):
    """Verifies lifecycle methods on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_poiseuille_flow_visible()
    assert board.get_poiseuille_flow() is None

    # Show
    item = board.show_poiseuille_flow(True, pos=(120.0, 50.0))
    assert item is not None
    assert board.is_poiseuille_flow_visible()
    assert board.get_poiseuille_flow() is item

    # Toggle hide
    is_vis = board.toggle_poiseuille_flow()
    assert not is_vis
    assert not board.is_poiseuille_flow_visible()

    # Toggle show
    is_vis = board.toggle_poiseuille_flow()
    assert is_vis
    assert board.is_poiseuille_flow_visible()

    # Stamping
    cnt_before = len(board.items())
    board._on_stamp_poiseuille_flow({
        "preset_code": "WATER_CAPILLARY_LAMINAR",
        "fluid_name": "Nước Cất ở 20°C (Water)",
        "radius_mm": 0.5,
        "delta_p_Pa": 1200.0,
        "viscosity_mPa_s": 1.002,
        "pipe_length_m": 0.20,
        "flow_rate_mL_per_s": 1.25,
        "max_velocity_m_per_s": 0.75,
        "average_velocity_m_per_s": 0.375,
        "reynolds_number": 450.0,
        "flow_regime": "LAMINAR",
    })
    assert len(board.items()) > cnt_before


def test_paint_render_pipeline(qapp):
    """Verifies that the item can paint onto a QImage without throwing exceptions."""
    item = InteractivePoiseuilleFlowItem()
    image = QImage(800, 700, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    item.paint(painter, None, None)
    painter.end()
    assert not image.isNull()
