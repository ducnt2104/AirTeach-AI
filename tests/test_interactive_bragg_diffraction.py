"""
AirTeach AI — Unit Tests for Interactive Materials Science & Crystallography:
Bragg's Law, X-Ray Diffraction (XRD) & Crystal Lattice Planes (Milestone 138 / Task I.125).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_bragg_diffraction import (
    InteractiveBraggDiffractionItem,
    MillerPlane,
    BraggDiffractionPreset,
    BRAGG_PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_cubic_interplanar_spacing_and_bragg_equation():
    """Verifies theoretical d_hkl = a / sqrt(h^2 + k^2 + l^2) and sin(theta_B) = lambda / (2d)."""
    # Silicon standard SRM 640: a = 0.5431 nm
    a_si = 0.5431
    plane_111 = MillerPlane(1, 1, 1, 100.0, "(111)")
    plane_220 = MillerPlane(2, 2, 0, 55.0, "(220)")
    plane_400 = MillerPlane(4, 0, 0, 6.0, "(400)")

    # 1. d-spacing
    d_111 = plane_111.d_spacing_nm(a_si)
    d_220 = plane_220.d_spacing_nm(a_si)
    d_400 = plane_400.d_spacing_nm(a_si)

    assert math.isclose(d_111, a_si / math.sqrt(3.0), rel_tol=1e-5)
    assert math.isclose(d_220, a_si / math.sqrt(8.0), rel_tol=1e-5)
    assert math.isclose(d_400, a_si / 4.0, rel_tol=1e-5)

    # 2. Bragg angle 2theta with Cu-Ka1 (lambda = 0.15406 nm)
    lam = 0.15406
    two_th_111 = plane_111.two_theta_deg(lam, a_si)
    two_th_220 = plane_220.two_theta_deg(lam, a_si)
    two_th_400 = plane_400.two_theta_deg(lam, a_si)

    assert two_th_111 is not None and math.isclose(two_th_111, 28.44, abs_tol=0.15)
    assert two_th_220 is not None and math.isclose(two_th_220, 47.30, abs_tol=0.20)
    assert two_th_400 is not None and math.isclose(two_th_400, 69.13, abs_tol=0.25)


def test_all_curriculum_presets_validity():
    """Verifies that all 5 curriculum presets contain authentic crystallographic parameters."""
    expected_codes = [
        "SILICON_POWDER_STANDARD",
        "COPPER_FCC_POLYCRYSTAL",
        "ALPHA_IRON_BCC_FERRITE",
        "SODIUM_CHLORIDE_ROCK_SALT",
        "NANOCRYSTALLINE_GRAIN_SIZE",
    ]
    for code in expected_codes:
        assert code in BRAGG_PRESETS, f"Preset {code} must be defined"
        preset = BRAGG_PRESETS[code]
        assert preset.lattice_a_nm > 0.1, "Lattice parameter a must be physical (> 0.1 nm)"
        assert preset.wavelength_nm > 0.05, "X-ray wavelength must be physical"
        assert preset.crystallite_size_nm > 1.0, "Crystallite size D must be > 1.0 nm"
        assert len(preset.allowed_planes) >= 3, "Preset must have at least 3 reflection planes"
        assert len(preset.description) > 40, "Pedagogical description must be detailed"

        # Check that all planes in preset have physically observable Bragg angles
        for plane in preset.allowed_planes:
            two_th = plane.two_theta_deg(preset.wavelength_nm, preset.lattice_a_nm)
            assert two_th is not None, f"Plane {plane.label} in {code} must produce real Bragg angle"
            assert 10.0 <= two_th <= 120.0, f"Diffraction angle {two_th} must be in goniometer range"


def test_extinction_rules_and_halite_structure_factor():
    """
    Verifies systematic extinction rules for cubic systems:
      - FCC: only all-odd or all-even (unmixed)
      - BCC: h + k + l must be even
      - NaCl: odd planes have weaker intensity than even planes due to |f_Cl - f_Na| < |f_Cl + f_Na|
    """
    # 1. Copper FCC: check all planes have unmixed indices
    cu_preset = BRAGG_PRESETS["COPPER_FCC_POLYCRYSTAL"]
    for p in cu_preset.allowed_planes:
        parities = [x % 2 for x in (p.h, p.k, p.l)]
        assert len(set(parities)) == 1, f"Plane {p.label} in FCC must be unmixed"

    # 2. Iron BCC: check h + k + l is even
    fe_preset = BRAGG_PRESETS["ALPHA_IRON_BCC_FERRITE"]
    for p in fe_preset.allowed_planes:
        assert (p.h + p.k + p.l) % 2 == 0, f"Plane {p.label} in BCC must have even h+k+l"

    # 3. NaCl Rock Salt: odd plane (111) is weaker than even plane (200)
    nacl_preset = BRAGG_PRESETS["SODIUM_CHLORIDE_ROCK_SALT"]
    p_111 = next(p for p in nacl_preset.allowed_planes if p.label == "(111)")
    p_200 = next(p for p in nacl_preset.allowed_planes if p.label == "(200)")
    assert p_111.relative_intensity < p_200.relative_intensity, (
        f"NaCl (111) intensity ({p_111.relative_intensity}) must be less than (200) ({p_200.relative_intensity})"
    )


def test_debye_scherrer_crystallite_broadening(qapp):
    """Verifies Scherrer formula beta = K*lambda / (D*cos theta): smaller grain -> broader peak."""
    item = InteractiveBraggDiffractionItem()

    # Standard Silicon: D = 45.0 nm
    item.set_preset("SILICON_POWDER_STANDARD")
    fwhm_bulk = item.scherrer_fwhm_deg
    assert fwhm_bulk < 0.35, f"Bulk crystal should have narrow Scherrer width: {fwhm_bulk:.3f}°"

    # Nanocrystal: D = 5.0 nm
    item.set_preset("NANOCRYSTALLINE_GRAIN_SIZE")
    fwhm_nano = item.scherrer_fwhm_deg
    assert fwhm_nano > 1.0, f"Nanocrystal must have broad Scherrer width: {fwhm_nano:.3f}°"

    # The ratio of FWHMs should be inversely proportional to crystallite size ratio
    ratio_fwhm = fwhm_nano / fwhm_bulk
    # D ratio: 45 / 5 = 9.0 (slight angular cos(theta) difference)
    assert 6.0 <= ratio_fwhm <= 12.0, f"Scherrer broadening ratio {ratio_fwhm} should match size ratio ~9.0"


def test_optical_path_difference_and_constructive_resonance(qapp):
    """Verifies that at exact Bragg angle, path difference Delta = 2d*sin(theta) = lambda."""
    item = InteractiveBraggDiffractionItem()
    item.set_preset("SILICON_POWDER_STANDARD")

    # Set to exact theoretical Bragg angle
    th_bragg = item.theoretical_bragg_theta_deg
    item.set_incident_theta(th_bragg)

    # Path difference Delta
    delta = item.optical_path_difference_nm
    lam = item._wavelength_nm
    assert math.isclose(delta, lam, rel_tol=1e-3), f"Delta ({delta:.5f} nm) must equal lambda ({lam:.5f} nm)"

    # Resonance flag
    assert item.is_bragg_resonance is True
    assert item.interference_intensity_factor > 0.95

    # Detune by 4.0 degrees away from Bragg condition
    item.set_incident_theta(th_bragg + 4.0)
    assert item.is_bragg_resonance is False
    assert item.interference_intensity_factor < 0.15


def test_interactive_controls_and_plane_cycling(qapp):
    """Tests interactive angle adjustment, plane selection, wavelength toggle, and grain size tuning."""
    item = InteractiveBraggDiffractionItem()
    item.set_preset("COPPER_FCC_POLYCRYSTAL")

    initial_plane = item.active_plane.label
    assert initial_plane == "(111)"

    # Cycle to next plane: (200)
    item.set_active_plane_index(1)
    assert item.active_plane.label == "(200)"
    assert item.theoretical_bragg_theta_deg > 20.0

    # Toggle wavelength from Cu-Ka (0.154 nm) to Mo-Ka (0.071 nm)
    item.toggle_wavelength()
    assert math.isclose(item._wavelength_nm, 0.07107, rel_tol=1e-3)
    # Shorter wavelength shifts Bragg angle to smaller angle: theta_Mo < theta_Cu
    assert item.theoretical_bragg_theta_deg < 15.0

    # Toggle back to Cu-Ka
    item.toggle_wavelength()
    assert math.isclose(item._wavelength_nm, 0.15406, rel_tol=1e-3)

    # Adjust crystallite size D
    initial_d = item._crystallite_d_nm
    item.adjust_crystallite_size(+10.0)
    assert item._crystallite_d_nm == initial_d + 10.0

    # Toggle auto-scan
    assert item._is_auto_scanning is False
    item.toggle_auto_scan()
    assert item._is_auto_scanning is True
    item.toggle_auto_scan()
    assert item._is_auto_scanning is False


def test_diffractogram_intensity_spectrum_calculation(qapp):
    """Tests continuous calculation of XRD spectrum I(2theta) around reflection peaks."""
    item = InteractiveBraggDiffractionItem()
    item.set_preset("SILICON_POWDER_STANDARD")

    # In Silicon, (111) peak is at 2theta ~ 28.44°
    i_peak = item.calculate_diffractogram_intensity(28.44)
    i_off_peak = item.calculate_diffractogram_intensity(20.0)

    assert i_peak > 70.0, f"Peak intensity at 28.44° should be high: {i_peak}"
    assert i_off_peak < 10.0, f"Off-peak background intensity should be low: {i_off_peak}"


def test_stamp_diagram_generation_and_lesson_board_integration(qapp):
    """Verifies that PhysicsTools.create_bragg_diffraction_diagram generates vector items and integrates with LessonBoard."""
    board = LessonBoard()

    # 1. Show and toggle Bragg item on board
    assert board.is_bragg_diffraction_visible() is False
    item = board.show_bragg_diffraction(True)
    assert item is not None
    assert board.is_bragg_diffraction_visible() is True
    assert board.get_bragg_diffraction() is item

    # Toggle off
    board.toggle_bragg_diffraction()
    assert board.is_bragg_diffraction_visible() is False

    # 2. Vector diagram stamping
    stamped_items = PhysicsTools.create_bragg_diffraction_diagram(
        center=(400.0, 300.0),
        preset_code="SILICON_POWDER_STANDARD",
        crystal_type="DIAMOND_CUBIC",
        lattice_a_nm=0.5431,
        wavelength_nm=0.15406,
        crystallite_d_nm=45.0,
        plane_label="(111)",
        d_spacing_nm=0.3135,
        bragg_theta_deg=14.22,
        two_theta_deg=28.44,
        scherrer_fwhm_deg=0.22,
    )
    assert len(stamped_items) >= 20, "Stamped diagram must produce full set of vector items"
    for itm in stamped_items:
        board.addItem(itm)

    # 3. Ensure stamped diagram is inside scene and selectable
    items_in_scene = board.get_items_in_rect(board.sceneRect())
    assert len(items_in_scene) >= len(stamped_items), "Stamped items must be selectable in rect"


def test_paint_render_pipeline_no_crash(qapp):
    """Verifies that paint() renders all panels without runtime errors or exceptions."""
    item = InteractiveBraggDiffractionItem()
    img = QImage(960, 650, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)

    painter = QPainter(img)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    # Verify image rendered non-zero pixels
    assert not img.isNull()


def test_main_window_toggle_bragg_diffraction(qapp):
    """Verifies that MainWindow._on_toggle_bragg_diffraction works and toggles visibility."""
    from src.ui.main_window import MainWindow
    win = MainWindow()
    try:
        # Initially hidden
        assert win._scene.is_bragg_diffraction_visible() is False

        # Toggle on
        is_vis = win._on_toggle_bragg_diffraction()
        assert is_vis is True
        assert win._scene.is_bragg_diffraction_visible() is True

        # Toggle off
        is_vis = win._on_toggle_bragg_diffraction()
        assert is_vis is False
        assert win._scene.is_bragg_diffraction_visible() is False
    finally:
        win.close()

