"""
Unit Tests for Interactive Special Relativity: Minkowski Spacetime Diagram & Lorentz Engine
==========================================================================================
Covers:
1. Authentic Lorentz boost transformation: ct' = γ(ct - βx), x' = γ(x - βct).
2. Lorentz factor γ = 1 / sqrt(1 - β^2) calculation and boundary clamping.
3. Invariant spacetime interval Δs^2 = c^2 Δt^2 - Δx^2 exact invariance between S and S'.
4. Interval classification: Timelike (Δs^2 > 0), Spacelike (Δs^2 < 0), Lightlike (Δs^2 = 0).
5. 5 Curriculum Presets (Lorentz Grid, Twin Paradox, Train Simultaneity, Muon Decay, Light Cone).
6. Spacetime-to-scene coordinate mapping and bidirectional roundtrip.
7. Feature toggles (moving frame S' axes, calibration hyperbolas, light cone).
8. Touchless webcam air gestures (air-tap on buttons and velocity slider).
9. High-DPI QPainter rendering without exceptions.
10. Canvas stamping and JSON serialization roundtrip (.airteach packaging).
11. LessonBoard lifecycle methods and MainWindow integration.
"""
import math
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent

from src.ui.interactive_special_relativity import (
    PRESETS,
    InteractiveSpecialRelativityItem,
    RelativityPreset,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_initialization(qapp):
    """Test default parameters, preset, dimensions, and flags."""
    item = InteractiveSpecialRelativityItem(pos=(140.0, 70.0))
    assert item.x() == 140.0
    assert item.y() == 70.0
    assert item._active_preset_code == "LORENTZ_GRID"
    assert abs(item._beta - 0.60) < 1e-4
    assert abs(item._gamma - 1.25) < 1e-4
    assert item._show_moving_frame is True
    assert item._show_hyperbolas is True
    assert item._show_light_cone is True
    assert item.boundingRect().width() == InteractiveSpecialRelativityItem.WIDTH
    assert item.boundingRect().height() == InteractiveSpecialRelativityItem.HEIGHT


def test_lorentz_transformation_and_inverse(qapp):
    """Verify Lorentz boost forward and inverse transformations match exact relativity mathematics."""
    item = InteractiveSpecialRelativityItem()
    item.set_beta(0.60)
    assert abs(item._gamma - 1.25) < 1e-4

    # Test stationary event (0, 0) remains (0, 0)
    xp, ctp = item.lorentz_transform(0.0, 0.0)
    assert abs(xp) < 1e-5 and abs(ctp) < 1e-5

    # Test event at (x=2.0, ct=1.0) with beta = 0.6:
    # ct' = 1.25 * (1.0 - 0.6 * 2.0) = 1.25 * (-0.2) = -0.25
    # x'  = 1.25 * (2.0 - 0.6 * 1.0) = 1.25 * (1.4)  = 1.75
    xp, ctp = item.lorentz_transform(2.0, 1.0)
    assert abs(xp - 1.75) < 1e-4
    assert abs(ctp - (-0.25)) < 1e-4

    # Test inverse boost transforms back to original coordinates
    x_orig, ct_orig = item.inverse_lorentz_transform(xp, ctp)
    assert abs(x_orig - 2.0) < 1e-4
    assert abs(ct_orig - 1.0) < 1e-4

    # At beta = 0.0, Lorentz boost is the identity transformation
    item.set_beta(0.0)
    assert abs(item._gamma - 1.0) < 1e-4
    xp_zero, ctp_zero = item.lorentz_transform(3.5, 4.2)
    assert abs(xp_zero - 3.5) < 1e-4
    assert abs(ctp_zero - 4.2) < 1e-4


def test_invariant_spacetime_interval_conservation(qapp):
    """Verify that Δs^2 is strictly invariant under any Lorentz boost across frames S and S'."""
    item = InteractiveSpecialRelativityItem()

    # Set arbitrary events E1 and E2
    item.set_event1_pos(-1.5, 0.5)
    item.set_event2_pos(2.0, 3.5)

    # In Frame S: dx = 3.5, dct = 3.0 -> s2 = 3.0^2 - 3.5^2 = 9.0 - 12.25 = -3.25 (Spacelike)
    s2, itype, proper = item.compute_spacetime_interval()
    assert abs(s2 - (-3.25)) < 1e-4
    assert "Spacelike" in itype
    assert abs(proper - math.sqrt(3.25)) < 1e-4

    # Now verify invariance across a range of relativistic velocities
    for beta_test in (-0.8, -0.5, 0.0, 0.3, 0.6, 0.85, 0.95):
        item.set_beta(beta_test)
        x1_p, ct1_p = item.lorentz_transform(item._e1[0], item._e1[1])
        x2_p, ct2_p = item.lorentz_transform(item._e2[0], item._e2[1])
        dx_p = x2_p - x1_p
        dct_p = ct2_p - ct1_p
        s2_prime = (dct_p * dct_p) - (dx_p * dx_p)
        assert abs(s2_prime - s2) < 1e-4, f"Failed at beta={beta_test}: {s2_prime} != {s2}"


def test_interval_classifications(qapp):
    """Test Timelike, Spacelike, and Lightlike interval classifications."""
    item = InteractiveSpecialRelativityItem()

    # 1. Timelike: dct = 4.0, dx = 1.0 -> s2 = 16 - 1 = 15 > 0
    item.set_event1_pos(0.0, 0.0)
    item.set_event2_pos(1.0, 4.0)
    s2, itype, proper = item.compute_spacetime_interval()
    assert s2 > 0.0
    assert "Timelike" in itype
    assert abs(proper - math.sqrt(15.0)) < 1e-4

    # 2. Spacelike: dct = 1.0, dx = 3.0 -> s2 = 1 - 9 = -8 < 0
    item.set_event1_pos(0.0, 0.0)
    item.set_event2_pos(3.0, 1.0)
    s2, itype, proper = item.compute_spacetime_interval()
    assert s2 < 0.0
    assert "Spacelike" in itype
    assert abs(proper - math.sqrt(8.0)) < 1e-4

    # 3. Lightlike: dct = 2.5, dx = 2.5 -> s2 = 0
    item.set_event1_pos(0.0, 0.0)
    item.set_event2_pos(2.5, 2.5)
    s2, itype, proper = item.compute_spacetime_interval()
    assert abs(s2) < 1e-4
    assert "Lightlike" in itype
    assert proper == 0.0


def test_curriculum_presets(qapp):
    """Test loading all 5 curriculum presets and signal emission."""
    item = InteractiveSpecialRelativityItem()
    emitted = []
    item.preset_changed.connect(emitted.append)

    for code in ("TWIN_PARADOX", "SIMULTANEITY_TRAIN", "MUON_DECAY", "LIGHT_CONE_CAUSALITY", "LORENTZ_GRID"):
        item.load_preset(code)
        assert item._active_preset_code == code
        preset = PRESETS[code]
        assert abs(item._beta - preset.beta) < 1e-4
        assert item._e1 == preset.e1_pos
        assert item._e2 == preset.e2_pos

    assert emitted == [
        "TWIN_PARADOX",
        "SIMULTANEITY_TRAIN",
        "MUON_DECAY",
        "LIGHT_CONE_CAUSALITY",
        "LORENTZ_GRID",
    ]


def test_coordinate_mapping_roundtrip(qapp):
    """Test bidirectional spacetime-to-scene coordinate mapping."""
    item = InteractiveSpecialRelativityItem()
    for x in (-3.0, 0.0, 2.5):
        for ct in (-2.0, 0.0, 4.0):
            scene_pt = item.spacetime_to_scene(x, ct)
            x_rec, ct_rec = item.scene_to_spacetime(scene_pt)
            assert abs(x_rec - x) < 1e-4
            assert abs(ct_rec - ct) < 1e-4


def test_toggles(qapp):
    """Test feature visibility toggles."""
    item = InteractiveSpecialRelativityItem()

    # Frame S' toggle
    assert item._show_moving_frame is True
    res = item.toggle_moving_frame()
    assert res is False and item._show_moving_frame is False
    res = item.toggle_moving_frame()
    assert res is True and item._show_moving_frame is True

    # Hyperbolas toggle
    assert item._show_hyperbolas is True
    res = item.toggle_hyperbolas()
    assert res is False and item._show_hyperbolas is False

    # Light cone toggle
    assert item._show_light_cone is True
    res = item.toggle_light_cone()
    assert res is False and item._show_light_cone is False


def test_touchless_air_tap(qapp):
    """Test touchless air gesture tapping on buttons and velocity slider."""
    item = InteractiveSpecialRelativityItem(pos=(100.0, 100.0))

    # 1. Tap outside card
    consumed = item.handle_air_tap(QPointF(0.0, 0.0))
    assert consumed is False

    # 2. Tap on quick beta button beta_0.8
    b8_rect = item._btn_rects["beta_0.8"]
    scene_pos = item.mapToScene(b8_rect.center())
    consumed = item.handle_air_tap(scene_pos)
    assert consumed is True
    assert abs(item._beta - 0.80) < 1e-3

    # 3. Tap on toggle_hyperbolas
    hyp_rect = item._btn_rects["toggle_hyperbolas"]
    scene_pos = item.mapToScene(hyp_rect.center())
    consumed = item.handle_air_tap(scene_pos)
    assert consumed is True
    assert item._show_hyperbolas is False

    # 4. Tap on slider center (beta = 0.0)
    slider_center = item.mapToScene(item._slider_rect.center())
    consumed = item.handle_air_tap(slider_center)
    assert consumed is True
    assert abs(item._beta) < 0.05


def test_qpainter_rendering(qapp):
    """Test QPainter rendering on QImage without crashes."""
    item = InteractiveSpecialRelativityItem()
    img = QImage(800, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()


def test_canvas_stamping(qapp):
    """Test stamping current relativity diagram onto canvas."""
    scene = QGraphicsScene()
    item = InteractiveSpecialRelativityItem(pos=(120.0, 60.0))
    item.load_preset("MUON_DECAY")
    item.set_beta(0.92)

    stamped = item.stamp_to_canvas(scene)
    assert len(stamped) == 1
    clone = stamped[0]
    assert isinstance(clone, InteractiveSpecialRelativityItem)
    assert clone._active_preset_code == "MUON_DECAY"
    assert abs(clone._beta - 0.92) < 1e-3


def test_serialization_roundtrip(qapp):
    """Test to_dict and from_dict state preservation."""
    item = InteractiveSpecialRelativityItem(pos=(160.0, 90.0))
    item.load_preset("TWIN_PARADOX")
    item.set_beta(0.85)
    item.set_event1_pos(-0.5, 0.2)
    item.set_event2_pos(1.8, 4.4)
    item.toggle_hyperbolas()

    data = item.to_dict()
    assert data["type"] == "interactive_special_relativity"
    assert data["pos"] == [160.0, 90.0]
    assert data["preset_code"] == "TWIN_PARADOX"
    assert abs(data["beta"] - 0.85) < 1e-4
    assert data["e1"] == [-0.5, 0.2]
    assert data["e2"] == [1.8, 4.4]
    assert data["show_hyperbolas"] is False

    restored = InteractiveSpecialRelativityItem.from_dict(data)
    assert restored.x() == 160.0
    assert restored.y() == 90.0
    assert restored._active_preset_code == "TWIN_PARADOX"
    assert abs(restored._beta - 0.85) < 1e-4
    assert restored._e1 == (-0.5, 0.2)
    assert restored._e2 == (1.8, 4.4)
    assert restored._show_hyperbolas is False


def test_lesson_board_relativity_integration(qapp):
    """Test LessonBoard lifecycle methods for Special Relativity visualizer."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert board.is_special_relativity_visible() is False
    assert board.get_special_relativity_item() is None

    # Show item
    item = board.show_special_relativity(True, pos=(180.0, 120.0))
    assert item is not None
    assert board.is_special_relativity_visible() is True
    assert board.get_special_relativity_item() is item
    assert item.pos() == QPointF(180.0, 120.0)

    # Toggle visibility
    is_vis = board.toggle_special_relativity()
    assert is_vis is False
    assert board.is_special_relativity_visible() is False

    is_vis = board.toggle_special_relativity()
    assert is_vis is True
    assert board.is_special_relativity_visible() is True


def test_main_window_relativity_toggle(qapp):
    """Test MainWindow toggle action and slot for Special Relativity visualizer."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_toggle_special_relativity")
        assert win._scene.is_special_relativity_visible() is False

        # Toggle on
        is_vis = win._on_toggle_special_relativity()
        assert is_vis is True
        assert win._scene.is_special_relativity_visible() is True

        # Toggle off
        is_vis = win._on_toggle_special_relativity()
        assert is_vis is False
        assert win._scene.is_special_relativity_visible() is False
    finally:
        win.close()

