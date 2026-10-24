"""
Unit tests for InteractiveThinLensSystemItem and PhysicsTools.create_thin_lens_system_diagram.
Verifies Gaussian thin lens formula 1/f = 1/d_o + 1/d_i, lateral magnification m = -d_i/d_o,
real and virtual image formation, 2x2 Ray Transfer ABCD Matrix invariants (det = 1, B = 0),
two-lens compound microscope and telescope systems, touchless air gestures, and lesson board integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_thin_lens_system import (
    InteractiveThinLensSystemItem,
    THIN_LENS_PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_thin_lens_initialization(qapp):
    item = InteractiveThinLensSystemItem()
    assert item.active_preset.code == "CONVEX_LENS_REAL_IMAGE"
    assert math.isclose(item._f1_mm, 100.0, rel_tol=1e-4)
    assert math.isclose(item.object_distance_d1_mm, 250.0, rel_tol=1e-4)
    assert item.is_image1_real is True
    assert item.magnification_m1 < 0.0  # Inverted
    assert item.boundingRect().width() == InteractiveThinLensSystemItem.CARD_W
    assert item.boundingRect().height() == InteractiveThinLensSystemItem.CARD_H


def test_gaussian_thin_lens_equation_exact_math(qapp):
    """
    Verifies Gaussian thin lens equation:
    1/f = 1/d_o + 1/d_i => d_i = (f * d_o) / (d_o - f)
    """
    item = InteractiveThinLensSystemItem()

    # 1. Real image with convex lens: f = 100, do = 250 -> di = 166.67 mm
    item.load_preset("CONVEX_LENS_REAL_IMAGE")
    expected_di = (100.0 * 250.0) / (250.0 - 100.0)
    assert math.isclose(item.image_distance_d1_mm, expected_di, rel_tol=1e-4)
    assert math.isclose(item.magnification_m1, -expected_di / 250.0, rel_tol=1e-4)

    # 2. Virtual image with magnifying glass: f = 100, do = 60 -> di = -150 mm
    item.load_preset("MAGNIFYING_GLASS_VIRTUAL_IMAGE")
    expected_mag_di = (100.0 * 60.0) / (60.0 - 100.0)
    assert math.isclose(item.image_distance_d1_mm, -150.0, rel_tol=1e-4)
    assert math.isclose(item.magnification_m1, 2.50, rel_tol=1e-4)
    assert item.is_image1_real is False

    # 3. Virtual image with concave lens: f = -120, do = 180 -> di = -72 mm
    item.load_preset("CONCAVE_DIVERGING_LENS")
    expected_concave_di = (-120.0 * 180.0) / (180.0 - (-120.0))
    assert math.isclose(item.image_distance_d1_mm, -72.0, rel_tol=1e-4)
    assert math.isclose(item.magnification_m1, 0.40, rel_tol=1e-4)
    assert item.is_image1_real is False


def test_lateral_magnification_scaling(qapp):
    """
    Verifies lateral magnification:
    - Inverted real images: m < 0
    - Upright virtual images: m > 0
    """
    item = InteractiveThinLensSystemItem()
    item.load_preset("CONVEX_LENS_REAL_IMAGE")
    assert item.magnification_m1 < 0.0
    assert item.image_height_h1_mm < 0.0

    item.load_preset("MAGNIFYING_GLASS_VIRTUAL_IMAGE")
    assert item.magnification_m1 > 1.0
    assert item.image_height_h1_mm > item._object_h_mm


def test_ray_transfer_abcd_matrix_invariants(qapp):
    """
    Verifies 2x2 Ray Transfer ABCD Matrix:
    - B = 0 mm on imaging conjugate plane (all rays from an object point converge at image point)
    - A = lateral magnification m
    - det(M) = AD - BC = 1.0 (Liouville's theorem / conservation of etendue)
    """
    item = InteractiveThinLensSystemItem()
    for preset_code in ["CONVEX_LENS_REAL_IMAGE", "MAGNIFYING_GLASS_VIRTUAL_IMAGE", "CONCAVE_DIVERGING_LENS"]:
        item.load_preset(preset_code)
        a, b, c, d = item.ray_transfer_matrix

        # Imaging condition: B == 0
        assert math.isclose(b, 0.0, abs_tol=1e-4)

        # Magnification: A == m
        assert math.isclose(a, item.magnification_m1, rel_tol=1e-4)

        # Determinant invariant: AD - BC == 1.0
        det_m = a * d - b * c
        assert math.isclose(det_m, 1.0, rel_tol=1e-4)


def test_compound_two_lens_microscope_optics(qapp):
    """
    Verifies compound two-lens microscope magnification: M_total = m1 * m2.
    """
    item = InteractiveThinLensSystemItem()
    item.load_preset("COMPOUND_MICROSCOPE_TWO_LENS")
    assert item._is_two_lens is True

    m_tot = item.total_magnification_m
    # Total magnification should be negative (inverted) and large (magnified)
    assert m_tot < -5.0


def test_all_five_presets_load(qapp):
    item = InteractiveThinLensSystemItem()
    for preset_code, preset in THIN_LENS_PRESETS.items():
        item.load_preset(preset_code)
        assert item.active_preset.code == preset_code
        assert item._f1_mm == preset.lens1.focal_length_mm
        assert item.object_distance_d1_mm > 0.0
        assert abs(item.total_magnification_m) > 0.0


def test_parameter_steppers_and_type_toggle(qapp):
    item = InteractiveThinLensSystemItem()
    init_do = item.object_distance_d1_mm
    item.step_object_distance(15.0)
    assert item.object_distance_d1_mm != init_do

    init_f = item._f1_mm
    item.step_focal_length(10.0)
    assert abs(item._f1_mm) == abs(init_f) + 10.0

    # Toggle lens type Convex <-> Concave
    item.toggle_lens_type()
    assert item._f1_mm < 0.0
    item.toggle_lens_type()
    assert item._f1_mm > 0.0

    item.toggle_animation()
    assert item._is_running is False
    item.toggle_animation()
    assert item._is_running is True


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractiveThinLensSystemItem()
    start_pos = item.pos()
    item.handle_air_drag(QPointF(20.0, 30.0))
    assert item.pos() == start_pos + QPointF(20.0, 30.0)

    # Air pinch on btn_lens_type
    rect_toggle = item._buttons["btn_lens_type"]
    scene_pt = item.mapToScene(rect_toggle.center())
    prev_f = item._f1_mm
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item._f1_mm == -prev_f


def test_vector_diagram_generation_and_stamping(qapp):
    item = InteractiveThinLensSystemItem()
    stamped = []
    item.stamp_requested.connect(lambda d: stamped.append(d))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert "f1_mm" in stamped[0]
    assert "di_mm" in stamped[0]
    assert "magnification" in stamped[0]

    items = PhysicsTools.create_thin_lens_system_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="CONVEX_LENS_REAL_IMAGE",
        f1_mm=100.0,
        do_mm=250.0,
        di_mm=166.7,
        magnification=-0.67,
        is_real=True,
        is_two_lens=False,
    )
    assert len(items) >= 15
    assert any("THẤU KÍNH" in getattr(it, "toPlainText", lambda: "")() for it in items)


def test_lesson_board_thin_lens_lifecycle(qapp):
    board = LessonBoard()
    assert board.is_thin_lens_visible() is False
    assert board.get_thin_lens() is None

    # Show
    item = board.show_thin_lens(True)
    assert item is not None
    assert board.is_thin_lens_visible() is True
    assert board.get_thin_lens() is item
    assert board.get_thin_lens_item() is item

    # Toggle to hide
    visible = board.toggle_thin_lens()
    assert visible is False
    assert board.is_thin_lens_visible() is False
    assert board.get_thin_lens() is None
    assert board.get_thin_lens_item() is not None

    # Toggle to show
    visible = board.toggle_thin_lens()
    assert visible is True
    assert board.is_thin_lens_visible() is True
