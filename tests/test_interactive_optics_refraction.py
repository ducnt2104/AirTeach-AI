"""
Unit Test Suite for Interactive Optics Snell's Law & Refraction Simulator
=========================================================================
Tests:
  - Optical geometry initialization and boundary rects.
  - Mathematical accuracy of Snell's Law (n1 * sin(i) = n2 * sin(r)).
  - Law of reflection (i' = i).
  - Critical angle computation (i_cr = arcsin(n2 / n1)) and Total Internal Reflection (TIR).
  - Medium parameter updates, cycling presets, and media swapping.
  - Light speed calculations in optical media (v = c / n).
  - Mouse interaction (dragging S and I, button clicks).
  - Touchless air gesture interaction (air drag, air tap).
  - Stamping permanent vector diagrams to LessonBoard and MainWindow integration.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from src.ui.interactive_optics_refraction import InteractiveOpticsRefractionItem, OPTICAL_MEDIA
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication is running for Qt GUI objects."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_optics_initialization_and_geometry(qapp):
    """Test default values, bounding box, and geometry coordinates."""
    item = InteractiveOpticsRefractionItem(
        pos=(200.0, 150.0),
        width=600.0,
        height=400.0,
        n1=1.000,
        n2=1.333,
    )
    assert item.pos() == QPointF(200.0, 150.0)
    assert item.boundingRect().width() >= 600.0
    assert item.boundingRect().height() >= 400.0

    # Incidence point I on interface
    pt_i = item.incidence_point()
    assert pt_i.y() == pytest.approx(200.0, abs=1.0)

    # Source point S in Medium 1
    pt_s = item.source_point()
    assert pt_s.y() < pt_i.y()

    # Normal line NN'
    n_top, n_bot = item.normal_endpoints()
    assert n_top.x() == pytest.approx(pt_i.x(), abs=1e-3)
    assert n_bot.x() == pytest.approx(pt_i.x(), abs=1e-3)
    assert n_top.y() < pt_i.y()
    assert n_bot.y() > pt_i.y()


def test_snells_law_and_refraction_angle(qapp):
    """Verify Snell's Law calculation: n1*sin(i) = n2*sin(r)."""
    # Air (n1 = 1.0) into Glass (n2 = 1.5)
    item = InteractiveOpticsRefractionItem(n1=1.000, n2=1.500)
    item.set_incident_angle(30.0)

    i_deg = item.get_incident_angle()
    assert i_deg == pytest.approx(30.0, abs=0.5)

    # Law of reflection
    assert item.get_reflection_angle() == pytest.approx(i_deg, abs=1e-3)

    # Snell's Law: sin(r) = (1.0 / 1.5) * sin(30°) = (1.0 / 1.5) * 0.5 = 1/3
    # r = arcsin(1/3) = 19.471°
    expected_r = math.degrees(math.asin(1.0 / 3.0))
    r_deg = item.get_refraction_angle()
    assert r_deg is not None
    assert r_deg == pytest.approx(expected_r, abs=0.5)

    # Going from rare to dense: no critical angle, TIR is impossible
    assert item.get_critical_angle() is None
    assert not item.is_total_internal_reflection()

    # Endpoints exist
    pt_r = item.reflected_ray_endpoint()
    assert pt_r.y() < item.incidence_point().y()

    pt_k = item.refracted_ray_endpoint()
    assert pt_k is not None
    assert pt_k.y() > item.incidence_point().y()


def test_total_internal_reflection_and_critical_angle(qapp):
    """Verify Total Internal Reflection (TIR) and critical angle when n1 > n2."""
    # Glass (n1 = 1.50) into Air (n2 = 1.00)
    item = InteractiveOpticsRefractionItem(n1=1.500, n2=1.000)

    # Critical angle: sin(i_cr) = 1.0 / 1.5 = 0.6667 => i_cr = 41.81°
    i_crit = item.get_critical_angle()
    assert i_crit is not None
    assert i_crit == pytest.approx(41.81, abs=0.2)

    # 1. Below critical angle: i = 30° < 41.81°
    item.set_incident_angle(30.0)
    assert not item.is_total_internal_reflection()
    r_deg = item.get_refraction_angle()
    assert r_deg is not None
    # sin(r) = 1.5 * sin(30°) = 0.75 => r = 48.59°
    assert r_deg == pytest.approx(48.59, abs=0.5)
    assert item.refracted_ray_endpoint() is not None

    # 2. Above critical angle: i = 60° > 41.81° => Total Internal Reflection!
    item.set_incident_angle(60.0)
    assert item.is_total_internal_reflection()
    assert item.get_refraction_angle() is None
    assert item.refracted_ray_endpoint() is None

    # Reflected ray is still fully present
    pt_r = item.reflected_ray_endpoint()
    assert pt_r is not None
    assert pt_r.y() < item.incidence_point().y()


def test_media_presets_and_swapping(qapp):
    """Test cycling optical presets and swapping n1 <-> n2."""
    item = InteractiveOpticsRefractionItem(n1=1.000, n2=1.333)

    # Test swap
    item.swap_media()
    assert item._n1 == pytest.approx(1.333, abs=1e-3)
    assert item._n2 == pytest.approx(1.000, abs=1e-3)

    # Swap back
    item.swap_media()
    assert item._n1 == pytest.approx(1.000, abs=1e-3)
    assert item._n2 == pytest.approx(1.333, abs=1e-3)

    # Cycle presets
    old_n1 = item._n1
    item.cycle_medium_1()
    assert item._n1 != old_n1

    old_n2 = item._n2
    item.cycle_medium_2()
    assert item._n2 != old_n2

    # Direct setter
    item.set_medium_1("Custom Medium", 1.62)
    assert item._n1 == pytest.approx(1.62, abs=1e-3)
    assert item._m1_name == "Custom Medium"


def test_speed_of_light_in_media(qapp):
    """Test light velocity v = c / n calculation."""
    item = InteractiveOpticsRefractionItem(n1=1.000, n2=1.500)
    v1, v2 = item.speed_of_light_in_media()
    assert v1 == pytest.approx(3.00, abs=1e-2)
    assert v2 == pytest.approx(2.00, abs=1e-2)


def test_touchless_air_gestures(qapp):
    """Test touchless air drag and air tap interactions."""
    scene = QGraphicsScene()
    item = InteractiveOpticsRefractionItem(pos=(100.0, 100.0))
    scene.addItem(item)

    # Hover update
    item.set_air_hover_pos(QPointF(150.0, 150.0))
    assert item._air_hover_pos is not None

    item.set_air_hover_pos(None)
    assert item._air_hover_pos is None

    # Air tap on swap button
    swap_center_local = item._rect_btn_swap.center()
    swap_center_scene = item.mapToScene(swap_center_local)
    old_n1 = item._n1
    old_n2 = item._n2

    tapped = item.handle_air_tap(swap_center_scene)
    assert tapped is True
    assert item._n1 == old_n2
    assert item._n2 == old_n1

    # Air drag near handle S
    s_scene = item.mapToScene(item.source_point())
    dragged = item.handle_air_drag(s_scene)
    assert dragged is True

    # Drag to new position
    new_s_scene = QPointF(s_scene.x() - 30.0, s_scene.y() + 10.0)
    dragged_move = item.handle_air_drag(new_s_scene)
    assert dragged_move is True


def test_stamp_to_canvas_and_lessonboard(qapp):
    """Test stamping vector ray diagram to LessonBoard canvas."""
    board = LessonBoard()
    item = InteractiveOpticsRefractionItem(pos=(300.0, 300.0), n1=1.0, n2=1.333)
    board.addItem(item)

    stamped_items = item.stamp_to_canvas(board)
    assert len(stamped_items) >= 4  # interface, normal, incident, reflected, refracted, text

    # Verify all stamped items exist in board scene
    for itm in stamped_items:
        assert itm.scene() == board


def test_lessonboard_and_main_window_integration(qapp):
    """Test LessonBoard lifecycle methods and MainWindow action toggles."""
    win = MainWindow()

    # LessonBoard lifecycle
    board = win._scene
    assert not board.is_optics_refraction_visible()

    board.show_optics_refraction(True)
    assert board.is_optics_refraction_visible()
    optics_item = board.get_optics_refraction_item()
    assert optics_item is not None
    assert optics_item.isVisible()

    board.show_optics_refraction(False)
    assert not board.is_optics_refraction_visible()

    # Toggle
    is_vis = board.toggle_optics_refraction()
    assert is_vis is True
    assert board.is_optics_refraction_visible()

    is_vis = board.toggle_optics_refraction()
    assert is_vis is False
    assert not board.is_optics_refraction_visible()

    # MainWindow slot
    toggled = win._on_toggle_optics_refraction()
    assert toggled is True
    assert board.is_optics_refraction_visible()

    win.close()
