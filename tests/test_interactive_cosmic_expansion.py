"""
AirTeach AI — Unit Tests for Interactive Astrophysics & Cosmology:
Hubble-Lemaître Law, Cosmological Redshift & Cosmic Expansion Rate Simulator (Milestone 143 / Task I.130).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_cosmic_expansion import (
    InteractiveCosmicExpansionItem,
    CosmicPreset,
    COSMIC_PRESETS,
    BALMER_LINES_NM,
    SPEED_OF_LIGHT_KM_S,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_cosmic_presets_and_parameters():
    """Verifies that all 5 curriculum presets are properly defined with physical parameters."""
    expected_codes = [
        "MODERN_PLANCK_LAMBDA_CDM",
        "SH0ES_SUPERNOVAE_TENSION",
        "EDWIN_HUBBLE_1929_ORIGINAL",
        "HIGH_REDSHIFT_JWST_GALAXY",
        "SUPERLUMINAL_HUBBLE_HORIZON",
    ]
    for code in expected_codes:
        assert code in COSMIC_PRESETS
        preset = COSMIC_PRESETS[code]
        assert preset.hubble_constant_km_s_mpc > 0.0
        assert preset.default_distance_mpc > 0.0
        assert preset.default_redshift_z > 0.0
        assert preset.omega_matter > 0.0
        assert preset.omega_lambda > 0.0
        assert len(preset.title_vi) > 0
        assert len(preset.target_name) > 0


def test_hubble_lemaitre_linear_law(qapp):
    """
    Verifies Hubble-Lemaître linear recessional velocity:
    v = H_0 * d
    """
    item = InteractiveCosmicExpansionItem()
    item.set_preset("MODERN_PLANCK_LAMBDA_CDM")
    h0 = item._hubble_constant  # 67.4 km/s/Mpc
    dist = 100.0  # 100 Mpc

    item.set_distance_mpc(dist)
    expected_v = h0 * dist  # 6740.0 km/s
    assert math.isclose(item.recessional_velocity_km_s, expected_v, rel_tol=1e-4)


def test_hydrogen_balmer_line_redshifts(qapp):
    """
    Verifies cosmological spectral line redshift:
    λ_obs = λ_emit * (1 + z)
    """
    item = InteractiveCosmicExpansionItem()
    item.set_preset("MODERN_PLANCK_LAMBDA_CDM")
    z = item.effective_redshift_z

    # Test H-alpha line (656.3 nm rest)
    rest_ha = BALMER_LINES_NM["Hα"]
    expected_obs_ha = rest_ha * (1.0 + z)
    obs_ha = item.get_observed_wavelength_nm(rest_ha)
    assert math.isclose(obs_ha, expected_obs_ha, rel_tol=1e-4)

    # Test H-beta line (486.1 nm rest)
    rest_hb = BALMER_LINES_NM["Hβ"]
    expected_obs_hb = rest_hb * (1.0 + z)
    obs_hb = item.get_observed_wavelength_nm(rest_hb)
    assert math.isclose(obs_hb, expected_obs_hb, rel_tol=1e-4)


def test_hubble_time_age_of_universe(qapp):
    """
    Verifies Hubble time calculation t_H = 1 / H_0 ≈ 13.8 - 14.5 Gyr.
    """
    item = InteractiveCosmicExpansionItem()
    item.set_preset("MODERN_PLANCK_LAMBDA_CDM")
    age_planck = item.hubble_time_Gyr
    assert 13.0 < age_planck < 15.0

    item.set_preset("SH0ES_SUPERNOVAE_TENSION")
    age_shoes = item.hubble_time_Gyr
    # Higher H0 implies younger expansion timescale
    assert age_shoes < age_planck


def test_hubble_horizon_and_superluminal_recession(qapp):
    """
    Verifies Hubble sphere horizon d_H = c / H_0.
    At distance d > d_H, spatial expansion recessional velocity exceeds c (v > c).
    """
    item = InteractiveCosmicExpansionItem()
    item.set_preset("SUPERLUMINAL_HUBBLE_HORIZON")
    h0 = item._hubble_constant  # 70.0 km/s/Mpc
    horizon_mpc = item.hubble_horizon_distance_Mpc
    expected_dh = SPEED_OF_LIGHT_KM_S / h0
    assert math.isclose(horizon_mpc, expected_dh, rel_tol=1e-4)

    # Preset distance is 4500 Mpc > horizon ≈ 4286 Mpc
    assert item._distance_mpc > horizon_mpc
    assert item.recessional_velocity_km_s > SPEED_OF_LIGHT_KM_S


def test_interactive_controls_and_buttons(qapp):
    """Verifies button adjustments and parameter resets."""
    item = InteractiveCosmicExpansionItem()
    item.set_preset("MODERN_PLANCK_LAMBDA_CDM")
    initial_d = item._distance_mpc
    initial_h0 = item._hubble_constant

    # Distance adjustments: 1.25 * 0.75 = 0.9375
    item._on_button_clicked("btn_d_up")
    assert item._distance_mpc > initial_d
    item._on_button_clicked("btn_d_down")
    assert math.isclose(item._distance_mpc, initial_d * 1.25 * 0.75, rel_tol=1e-4)

    # Hubble constant adjustments: 1.05 * 0.95 = 0.9975
    item._on_button_clicked("btn_h0_up")
    assert item._hubble_constant > initial_h0
    item._on_button_clicked("btn_h0_down")
    assert math.isclose(item._hubble_constant, initial_h0 * 1.05 * 0.95, rel_tol=1e-4)

    # Preset cycle
    item._on_button_clicked("btn_preset")
    assert item._active_preset_code != "MODERN_PLANCK_LAMBDA_CDM"

    # Reset
    item._on_button_clicked("btn_reset")
    assert math.isclose(item._hubble_constant, item._preset.hubble_constant_km_s_mpc, rel_tol=1e-4)


def test_stamping_and_stem_tools_vector_generation(qapp):
    """Verifies that PhysicsTools.create_cosmic_expansion_diagram generates valid vector items."""
    diagram_items = PhysicsTools.create_cosmic_expansion_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="MODERN_PLANCK_LAMBDA_CDM",
        target_name="Thiên hà Chuẩn Cụm Coma",
        hubble_constant=67.4,
        distance_mpc=100.0,
        recessional_velocity_km_s=6740.0,
        redshift_z=0.0225,
        hubble_time_Gyr=14.51,
        hubble_horizon_distance_Mpc=4448.0,
    )
    assert len(diagram_items) >= 20

    # Test stamp signal emission
    item = InteractiveCosmicExpansionItem()
    emitted = []
    item.stamp_requested.connect(lambda d: emitted.append(d))
    item.request_stamp()
    assert len(emitted) == 1
    assert "hubble_constant" in emitted[0]
    assert "recessional_velocity_km_s" in emitted[0]
    assert "redshift_z" in emitted[0]


def test_lesson_board_integration(qapp):
    """Verifies lifecycle methods on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_cosmic_expansion_visible()
    assert board.get_cosmic_expansion() is None

    # Show
    item = board.show_cosmic_expansion(True, pos=(130.0, 55.0))
    assert item is not None
    assert board.is_cosmic_expansion_visible()
    assert board.get_cosmic_expansion() is item

    # Toggle hide
    is_vis = board.toggle_cosmic_expansion()
    assert not is_vis
    assert not board.is_cosmic_expansion_visible()

    # Toggle show
    is_vis = board.toggle_cosmic_expansion()
    assert is_vis
    assert board.is_cosmic_expansion_visible()

    # Stamping
    cnt_before = len(board.items())
    board._on_stamp_cosmic_expansion({
        "preset_code": "MODERN_PLANCK_LAMBDA_CDM",
        "target_name": "Thiên hà Chuẩn Cụm Coma",
        "hubble_constant": 67.4,
        "distance_mpc": 100.0,
        "recessional_velocity_km_s": 6740.0,
        "redshift_z": 0.0225,
        "hubble_time_Gyr": 14.51,
        "hubble_horizon_distance_Mpc": 4448.0,
    })
    assert len(board.items()) > cnt_before


def test_paint_render_pipeline(qapp):
    """Verifies that the item can paint onto a QImage without throwing exceptions."""
    item = InteractiveCosmicExpansionItem()
    image = QImage(800, 700, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    item.paint(painter, None, None)
    painter.end()
    assert not image.isNull()
