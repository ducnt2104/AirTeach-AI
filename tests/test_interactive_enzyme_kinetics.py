"""
AirTeach AI — Unit Tests for Interactive Biochemistry & Pharmacology:
Michaelis-Menten Enzyme Kinetics, Lineweaver-Burk Double Reciprocal Plot & Enzyme Inhibition Simulator (Milestone 141 / Task I.128).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_enzyme_kinetics import (
    InteractiveEnzymeKineticsItem,
    EnzymePreset,
    ENZYME_PRESETS,
)
from src.ui.stem_tools import BiologyTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_enzyme_presets_and_constants():
    """Verifies that all 5 curriculum presets are properly defined with physical constants."""
    expected_codes = [
        "CHYMOTRYPSIN_PEPTIDE_CLEAVAGE",
        "COMPETITIVE_INHIBITION_STATIN",
        "NONCOMPETITIVE_ALLOSTERIC_INHIBITION",
        "UNCOMPETITIVE_INHIBITION_MODEL",
        "CATALASE_ULTRA_FAST_DIFFUSION_LIMIT",
    ]
    for code in expected_codes:
        assert code in ENZYME_PRESETS
        preset = ENZYME_PRESETS[code]
        assert preset.km_mM > 0.0
        assert preset.vmax_uM_per_s > 0.0
        assert preset.kcat_per_s > 0.0
        assert preset.ki_uM > 0.0
        assert preset.inhibition_type in ("NONE", "COMPETITIVE", "NON_COMPETITIVE", "UNCOMPETITIVE")
        assert len(preset.title_vi) > 0
        assert len(preset.enzyme_name) > 0


def test_michaelis_menten_hyperbolic_saturation(qapp):
    """
    Verifies that v_0 = (V_max * [S]) / (K_m + [S]):
    1. At [S] = 0, v_0 = 0.
    2. At [S] = K_m, v_0 = V_max / 2 (50% saturation).
    3. At [S] >> K_m, v_0 approaches V_max asymptotically.
    """
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("CHYMOTRYPSIN_PEPTIDE_CLEAVAGE")
    km = item.apparent_km_mM          # 1.5 mM
    vmax = item.apparent_vmax_uM_per_s  # 100.0 uM/s

    # At [S] = Km
    item.set_substrate_concentration(km)
    assert math.isclose(item.current_velocity_v0, vmax / 2.0, rel_tol=1e-4)
    assert math.isclose(item.fractional_saturation_theta, 0.50, rel_tol=1e-4)

    # At minimum limit [S] = 0.1 mM
    item.set_substrate_concentration(0.1)
    expected_min_v0 = (vmax * 0.1) / (km + 0.1)
    assert math.isclose(item.current_velocity_v0, expected_min_v0, rel_tol=1e-4)

    # At [S] = 4 * Km
    item.set_substrate_concentration(4.0 * km)
    expected_v0 = (vmax * 4.0 * km) / (km + 4.0 * km)  # 0.8 * vmax = 80.0
    assert math.isclose(item.current_velocity_v0, expected_v0, rel_tol=1e-4)
    assert math.isclose(item.fractional_saturation_theta, 0.80, rel_tol=1e-4)


def test_competitive_inhibition_mechanics(qapp):
    """
    Competitive Inhibition:
    - Inhibitor competes for free enzyme active site.
    - alpha = 1 + [I] / Ki
    - Apparent Km increases: Km_app = alpha * Km
    - Apparent Vmax remains unchanged: Vmax_app = Vmax
    """
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("COMPETITIVE_INHIBITION_STATIN")
    base_km = 2.0
    base_vmax = 100.0
    ki = 4.0

    # With default [I] = 8.0 uM: alpha = 1 + 8/4 = 3.0
    assert math.isclose(item.alpha_inhibitor_factor, 3.0, rel_tol=1e-4)
    assert math.isclose(item.apparent_km_mM, base_km * 3.0, rel_tol=1e-4)
    assert math.isclose(item.apparent_vmax_uM_per_s, base_vmax, rel_tol=1e-4)

    # Increase inhibitor to [I] = 12.0 uM: alpha = 1 + 12/4 = 4.0
    item.set_inhibitor_concentration(12.0)
    assert math.isclose(item.alpha_inhibitor_factor, 4.0, rel_tol=1e-4)
    assert math.isclose(item.apparent_km_mM, base_km * 4.0, rel_tol=1e-4)
    assert math.isclose(item.apparent_vmax_uM_per_s, base_vmax, rel_tol=1e-4)


def test_noncompetitive_inhibition_mechanics(qapp):
    """
    Non-Competitive (Allosteric) Inhibition:
    - Inhibitor binds allosteric site on both E and ES.
    - alpha = 1 + [I] / Ki
    - Apparent Vmax decreases: Vmax_app = Vmax / alpha
    - Apparent Km remains unchanged: Km_app = Km
    """
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("NONCOMPETITIVE_ALLOSTERIC_INHIBITION")
    base_km = 1.8
    base_vmax = 100.0
    ki = 5.0

    # Default [I] = 5.0 uM => alpha = 1 + 5/5 = 2.0
    assert math.isclose(item.alpha_inhibitor_factor, 2.0, rel_tol=1e-4)
    assert math.isclose(item.apparent_km_mM, base_km, rel_tol=1e-4)
    assert math.isclose(item.apparent_vmax_uM_per_s, base_vmax / 2.0, rel_tol=1e-4)


def test_uncompetitive_inhibition_mechanics(qapp):
    """
    Uncompetitive Inhibition:
    - Inhibitor binds only to the enzyme-substrate complex [ES].
    - Both Km and Vmax decrease by the same factor alpha:
      Km_app = Km / alpha
      Vmax_app = Vmax / alpha
    - The slope of Lineweaver-Burk (Km_app / Vmax_app) remains invariant (parallel lines).
    """
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("UNCOMPETITIVE_INHIBITION_MODEL")
    base_km = 3.0
    base_vmax = 100.0
    ki = 6.0

    # Default [I] = 6.0 uM => alpha = 1 + 6/6 = 2.0
    assert math.isclose(item.alpha_inhibitor_factor, 2.0, rel_tol=1e-4)
    assert math.isclose(item.apparent_km_mM, base_km / 2.0, rel_tol=1e-4)
    assert math.isclose(item.apparent_vmax_uM_per_s, base_vmax / 2.0, rel_tol=1e-4)

    # Invariant slope check
    slope_uninhibited = base_km / base_vmax
    slope_inhibited = item.apparent_km_mM / item.apparent_vmax_uM_per_s
    assert math.isclose(slope_uninhibited, slope_inhibited, rel_tol=1e-4)


def test_lineweaver_burk_intercepts(qapp):
    """
    Verifies the double reciprocal Lineweaver-Burk coordinates:
    1 / v_0 = (Km / Vmax) * (1 / [S]) + 1 / Vmax
    - y-intercept = 1 / Vmax_app
    - x-intercept = -1 / Km_app
    """
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("CHYMOTRYPSIN_PEPTIDE_CLEAVAGE")
    km = item.apparent_km_mM
    vmax = item.apparent_vmax_uM_per_s

    # Inverted values
    inv_vmax = 1.0 / vmax
    inv_km_neg = -1.0 / km

    assert math.isclose(inv_vmax, 1.0 / 100.0, rel_tol=1e-4)
    assert math.isclose(inv_km_neg, -1.0 / 1.5, rel_tol=1e-4)


def test_turnover_number_and_catalytic_efficiency(qapp):
    """
    Verifies catalytic turnover number kcat and specificity constant kcat / Km.
    Catalase should exhibit ultra-fast diffusion limit (~10^7 - 10^8 M^-1 s^-1).
    """
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("CATALASE_ULTRA_FAST_DIFFUSION_LIMIT")
    kcat = item._preset.kcat_per_s
    km_mM = item._preset.km_mM
    # Catalytic efficiency = kcat / Km (converted to M^-1 s^-1: kcat / (km_mM * 1e-3))
    eff_m_per_s = kcat / (km_mM * 1e-3)
    assert eff_m_per_s > 1e7  # Ultra fast diffusion limit


def test_interactive_controls_and_buttons(qapp):
    """Verifies cycling modes, adjusting concentrations, and reset actions."""
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("CHYMOTRYPSIN_PEPTIDE_CLEAVAGE")
    initial_s = item._substrate_mM

    # Substrate concentration adjustment
    item._on_button_clicked("btn_s_up")
    assert item._substrate_mM > initial_s
    item._on_button_clicked("btn_s_down")
    assert math.isclose(item._substrate_mM, initial_s, abs_tol=1e-4)

    # Mode toggle
    assert item._inhibition_type == "NONE"
    item._on_button_clicked("btn_mode")
    assert item._inhibition_type == "COMPETITIVE"
    item._on_button_clicked("btn_mode")
    assert item._inhibition_type == "NON_COMPETITIVE"
    item._on_button_clicked("btn_mode")
    assert item._inhibition_type == "UNCOMPETITIVE"
    item._on_button_clicked("btn_mode")
    assert item._inhibition_type == "NONE"

    # Reset
    item.set_substrate_concentration(20.0)
    item._on_button_clicked("btn_reset")
    assert math.isclose(item._substrate_mM, 3.0, rel_tol=1e-4)


def test_stamping_and_stem_tools_vector_generation(qapp):
    """Verifies that BiologyTools.create_enzyme_kinetics_diagram generates valid vector items."""
    diagram_items = BiologyTools.create_enzyme_kinetics_diagram(
        center=(500.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="COMPETITIVE_INHIBITION_STATIN",
        enzyme_name="HMG-CoA Reductase",
        substrate_name="HMG-CoA",
        km_mM=6.0,
        vmax_uM_per_s=100.0,
        substrate_mM=2.0,
        v0_uM_per_s=25.0,
        inhibition_type="COMPETITIVE",
        alpha=3.0,
    )
    assert len(diagram_items) >= 20

    # Test stamp signal emission
    item = InteractiveEnzymeKineticsItem()
    item.set_preset("CHYMOTRYPSIN_PEPTIDE_CLEAVAGE")
    emitted_data = []
    item.stamp_requested.connect(lambda d: emitted_data.append(d))
    item.request_stamp()
    assert len(emitted_data) == 1
    data = emitted_data[0]
    assert data["enzyme_name"] == "Chymotrypsin"
    assert "km_mM" in data
    assert "vmax_uM_per_s" in data
    assert "v0_uM_per_s" in data


def test_lesson_board_integration(qapp):
    """Verifies lifecycle methods on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_enzyme_kinetics_visible()
    assert board.get_enzyme_kinetics() is None

    # Show
    item = board.show_enzyme_kinetics(True, pos=(150.0, 60.0))
    assert item is not None
    assert board.is_enzyme_kinetics_visible()
    assert board.get_enzyme_kinetics() is item

    # Toggle hide
    is_vis = board.toggle_enzyme_kinetics()
    assert not is_vis
    assert not board.is_enzyme_kinetics_visible()

    # Toggle show
    is_vis = board.toggle_enzyme_kinetics()
    assert is_vis
    assert board.is_enzyme_kinetics_visible()

    # Stamping
    initial_items_count = len(board.items())
    board._on_stamp_enzyme_kinetics({
        "preset_code": "CHYMOTRYPSIN_PEPTIDE_CLEAVAGE",
        "enzyme_name": "Chymotrypsin",
        "substrate_name": "ATEE",
        "km_mM": 1.5,
        "vmax_uM_per_s": 100.0,
        "substrate_mM": 1.5,
        "v0_uM_per_s": 50.0,
        "inhibition_type": "NONE",
        "alpha": 1.0,
    })
    assert len(board.items()) > initial_items_count


def test_paint_render_pipeline(qapp):
    """Verifies that the item can paint onto a QImage without throwing exceptions."""
    item = InteractiveEnzymeKineticsItem()
    image = QImage(800, 600, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    item.paint(painter, None, None)
    painter.end()
    assert not image.isNull()
