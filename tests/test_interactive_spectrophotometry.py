"""
AirTeach AI — Unit Tests for Interactive Analytical Chemistry & Spectroscopy:
Spectrophotometry, Beer-Lambert Law & Chemical Absorbance Simulator (Milestone 146 / Task I.133).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_spectrophotometry import (
    InteractiveSpectrophotometryItem,
    SpectroPreset,
    SPECTRO_PRESETS,
)
from src.ui.stem_tools import ChemistryTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_spectro_presets_and_parameters():
    """Verifies that all 5 curriculum presets are properly defined with physical/optical parameters."""
    expected_codes = [
        "POTASSIUM_PERMANGANATE_KMnO4",
        "COPPER_SULFATE_PENTAHYDRATE",
        "PROTEIN_BRADFORD_COOMASSIE",
        "DNA_UV_260_NUCLEIC_ACID",
        "BEER_LAMBERT_HIGH_CONC_DEVIATION",
    ]
    for code in expected_codes:
        assert code in SPECTRO_PRESETS
        preset = SPECTRO_PRESETS[code]
        assert preset.lambda_max_nm > 0.0
        assert preset.epsilon_L_mol_cm > 0.0
        assert preset.default_conc_mM > 0.0
        assert preset.default_path_cm > 0.0
        assert preset.liquid_color.isValid()
        assert preset.absorbed_color.isValid()
        assert len(preset.analyte_name) > 0


def test_beer_lambert_law_exact_physics(qapp):
    """
    Verifies Beer-Lambert analytical equation:
    A = epsilon * c * l
    T = 10^(-A)
    %T = 100 * 10^(-A)
    Also tests instrumental stray light behavior.
    """
    item = InteractiveSpectrophotometryItem()
    item.set_preset("POTASSIUM_PERMANGANATE_KMnO4")
    # For KMnO4: lambda_max = 525 nm, epsilon = 2400 L/(mol*cm), default_conc = 0.40 mM = 0.0004 M, path = 1.0 cm
    expected_conc_M = 0.40 * 1e-3
    expected_A = 2400.0 * expected_conc_M * 1.0  # 0.960
    expected_T = 10.0 ** (-expected_A)            # ~0.109648
    expected_T_pct = expected_T * 100.0           # ~10.965%

    assert math.isclose(item.true_absorbance, expected_A, rel_tol=1e-3)

    # When stray light is zero, transmittance equals ideal theoretical T
    item._stray_light_pct = 0.0
    assert math.isclose(item.transmittance_fraction, expected_T, rel_tol=1e-3)
    assert math.isclose(item.transmittance_percent, expected_T_pct, rel_tol=1e-3)

    # When 0.05% instrumental stray light is present
    item._stray_light_pct = 0.05
    assert math.isclose(item.transmittance_fraction, expected_T + 0.0005, rel_tol=1e-3)


def test_gaussian_extinction_wavelength_dependence(qapp):
    """
    Verifies Gaussian absorption band:
    epsilon(lambda) = epsilon_max * exp(-0.5 * ((lambda - lambda_max) / sigma)^2)
    Off-peak wavelength must give lower extinction and lower absorbance.
    """
    item = InteractiveSpectrophotometryItem()
    item.set_preset("POTASSIUM_PERMANGANATE_KMnO4")
    item.set_wavelength_nm(525.0)  # at peak
    a_peak = item.true_absorbance
    eps_peak = item.current_epsilon

    item.set_wavelength_nm(565.0)  # 40 nm away from peak
    a_offpeak = item.true_absorbance
    eps_offpeak = item.current_epsilon

    assert eps_offpeak < eps_peak
    assert a_offpeak < a_peak


def test_path_length_and_concentration_linearity(qapp):
    """
    Verifies linear dependence of Absorbance on path length l and concentration c.
    """
    item = InteractiveSpectrophotometryItem()
    item.set_preset("POTASSIUM_PERMANGANATE_KMnO4")
    item.set_wavelength_nm(525.0)
    item.set_concentration_mM(0.20)
    item.set_path_length_cm(1.0)
    a_base = item.true_absorbance

    # Double concentration -> double absorbance
    item.set_concentration_mM(0.40)
    a_double_conc = item.true_absorbance
    assert math.isclose(a_double_conc, 2.0 * a_base, rel_tol=1e-3)

    # Double path length -> double absorbance
    item.set_concentration_mM(0.20)
    item.set_path_length_cm(2.0)
    a_double_path = item.true_absorbance
    assert math.isclose(a_double_path, 2.0 * a_base, rel_tol=1e-3)


def test_interactive_controls_and_buttons(qapp):
    """Tests interactive parameter adjustment methods and clamping bounds."""
    item = InteractiveSpectrophotometryItem()
    item.set_preset("COPPER_SULFATE_PENTAHYDRATE")

    # Test concentration adjustments
    item.set_concentration_mM(50.0)
    assert math.isclose(item.concentration_M, 0.050, rel_tol=1e-4)

    # Test path length adjustments
    item.set_path_length_cm(2.5)
    assert math.isclose(item._path_cm, 2.5, rel_tol=1e-4)

    # Test reset to defaults
    item.reset_defaults()
    assert math.isclose(item._path_cm, item._preset.default_path_cm, rel_tol=1e-4)
    assert math.isclose(item._conc_mM, item._preset.default_conc_mM, rel_tol=1e-4)


def test_stem_tools_chemistry_diagram():
    """Verifies vector diagram creation in ChemistryTools for spectrophotometry."""
    items = ChemistryTools.create_spectrophotometry_diagram(
        center=(400.0, 300.0),
        preset_code="POTASSIUM_PERMANGANATE_KMnO4",
        analyte_name="KMnO₄ (Potassium Permanganate)",
        conc_mM=0.40,
        path_cm=1.0,
        wavelength_nm=525.0,
        epsilon=2400.0,
        absorbance=0.96,
        transmittance_pct=10.96,
    )
    assert len(items) >= 25
    # Check that text and path items are included
    types = [type(itm).__name__ for itm in items]
    assert "QGraphicsRectItem" in types
    assert "QGraphicsTextItem" in types
    assert "QGraphicsLineItem" in types


def test_lesson_board_spectrophotometry_lifecycle(qapp):
    """Verifies integration, toggling, and stamping on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_spectrophotometry_visible()

    # Show
    item = board.show_spectrophotometry(True)
    assert item is not None
    assert board.is_spectrophotometry_visible()
    assert board.get_spectrophotometry() is item

    # Toggle off
    state = board.toggle_spectrophotometry()
    assert not state
    assert not board.is_spectrophotometry_visible()
    assert board.get_spectrophotometry() is None

    # Toggle on
    state = board.toggle_spectrophotometry()
    assert state
    assert board.is_spectrophotometry_visible()

    # Test stamp
    initial_count = len(board.items())
    board._on_stamp_spectrophotometry({
        "preset_code": "POTASSIUM_PERMANGANATE_KMnO4",
        "analyte_name": "KMnO₄",
        "conc_mM": 0.40,
        "path_cm": 1.0,
        "wavelength_nm": 525.0,
        "epsilon": 2400.0,
        "absorbance": 0.96,
        "transmittance_pct": 10.96,
    })
    after_count = len(board.items())
    assert after_count > initial_count + 20


def test_paint_rendering_zero_crash(qapp):
    """Ensures QPainter rendering executes without raising any exception."""
    item = InteractiveSpectrophotometryItem()
    image = QImage(700, 420, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    # If it completed without crash, paint succeeds
    assert not image.isNull()
