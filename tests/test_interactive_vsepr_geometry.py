"""
Unit Test Suite for Interactive Chemistry 3D VSEPR Molecular Geometry Inspector
===============================================================================
Tests:
  - VSEPR model initialization and default configuration (AX4 Tetrahedral).
  - All 8 standard VSEPR presets (AX2, AX3, AX2E, AX4, AX3E, AX2E2, AX5, AX6).
  - Chemical properties: hybridization, bond angles, lone pairs, and real molecule examples.
  - 3D Euler rotation (Pitch & Yaw) and perspective point projection.
  - Auto-spin animation timer progression.
  - Mouse dragging for 3D rotation and button click handling.
  - Touchless air gesture interaction (Air Drag & Air Tap).
  - Stamping permanent vector diagrams to LessonBoard and MainWindow integration.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_vsepr_geometry import InteractiveVseprGeometryItem
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication is running for Qt GUI objects."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_vsepr_initialization_and_models(qapp):
    """Test default parameters, bounding box, and presets presence."""
    item = InteractiveVseprGeometryItem(pos=(200.0, 150.0), model_key="AX4")
    assert item.pos() == QPointF(200.0, 150.0)
    assert item.boundingRect().width() >= 640.0
    assert item.boundingRect().height() >= 420.0
    assert item.current_key() == "AX4"

    model = item.current_model()
    assert model.type_code == "AX₄"
    assert "Tứ diện" in model.name_vi
    assert model.hybridization == "sp³"
    assert model.bond_angle_str == "109.5°"
    assert model.example_formula == "CH₄"
    assert len(model.ligands) == 4
    assert len(model.lone_pairs) == 0

    item.close()


def test_vsepr_model_selection_and_properties(qapp):
    """Test switching between all 8 standard VSEPR configurations."""
    item = InteractiveVseprGeometryItem()

    # 1. AX2: Linear (CO2)
    item.set_model("AX2")
    m = item.current_model()
    assert m.type_code == "AX₂"
    assert m.hybridization == "sp"
    assert m.bond_angle_str == "180°"
    assert len(m.ligands) == 2
    assert len(m.lone_pairs) == 0

    # 2. AX3: Trigonal Planar (BF3)
    item.set_model("AX3")
    m = item.current_model()
    assert m.type_code == "AX₃"
    assert m.hybridization == "sp²"
    assert m.bond_angle_str == "120°"
    assert len(m.ligands) == 3

    # 3. AX2E: Bent (SO2)
    item.set_model("AX2E")
    m = item.current_model()
    assert m.type_code == "AX₂E"
    assert m.hybridization == "sp²"
    assert len(m.ligands) == 2
    assert len(m.lone_pairs) == 1

    # 4. AX3E: Trigonal Pyramidal (NH3)
    item.set_model("AX3E")
    m = item.current_model()
    assert m.type_code == "AX₃E"
    assert m.hybridization == "sp³"
    assert len(m.ligands) == 3
    assert len(m.lone_pairs) == 1
    assert m.example_formula == "NH₃"

    # 5. AX2E2: Bent (H2O)
    item.set_model("AX2E2")
    m = item.current_model()
    assert m.type_code == "AX₂E₂"
    assert m.hybridization == "sp³"
    assert len(m.ligands) == 2
    assert len(m.lone_pairs) == 2
    assert m.example_formula == "H₂O"

    # 6. AX5: Trigonal Bipyramidal (PCl5)
    item.set_model("AX5")
    m = item.current_model()
    assert m.type_code == "AX₅"
    assert m.hybridization == "sp³d"
    assert len(m.ligands) == 5

    # 7. AX6: Octahedral (SF6)
    item.set_model("AX6")
    m = item.current_model()
    assert m.type_code == "AX₆"
    assert m.hybridization == "sp³d²"
    assert m.bond_angle_str == "90°"
    assert len(m.ligands) == 6

    item.close()


def test_3d_rotation_and_projection(qapp):
    """Test 3D Euler rotation and pseudo-perspective projection."""
    item = InteractiveVseprGeometryItem()

    # Zero rotation projection
    item.set_rotation(0.0, 0.0)
    sx, sy, sz = item.project_point(0.0, 0.0, 0.0)
    # Origin should project exactly to projection center
    assert sx == pytest.approx(item._proj_cx, abs=1e-3)
    assert sy == pytest.approx(item._proj_cy, abs=1e-3)
    assert sz == pytest.approx(0.0, abs=1e-3)

    # 90° Yaw rotation: (100, 0, 0) rotated by 90° around Y => (0, 0, -100)
    item.set_rotation(0.0, 90.0)
    rx, ry, rz = item._rotate_3d(100.0, 0.0, 0.0)
    assert rx == pytest.approx(0.0, abs=1e-3)
    assert ry == pytest.approx(0.0, abs=1e-3)
    assert rz == pytest.approx(-100.0, abs=1e-3)

    # Angle wrapping
    item.set_rotation(370.0, -400.0)
    assert -180.0 <= item._pitch <= 180.0
    assert -180.0 <= item._yaw <= 180.0

    item.close()


def test_auto_spin_animation(qapp):
    """Test 3D auto-spin animation toggle and timer callback."""
    item = InteractiveVseprGeometryItem()
    assert not item.is_spinning()

    # Toggle spin
    spinning = item.toggle_spin()
    assert spinning is True
    assert item.is_spinning() is True

    # Advance tick
    old_yaw = item._yaw
    item._on_spin_tick()
    assert item._yaw != old_yaw

    # Stop spin
    spinning = item.toggle_spin()
    assert spinning is False
    assert item.is_spinning() is False

    item.close()


def test_touchless_air_gestures(qapp):
    """Test touchless air drag (3D rotation) and air tap (model selector)."""
    scene = QGraphicsScene()
    item = InteractiveVseprGeometryItem(pos=(100.0, 100.0), model_key="AX4")
    scene.addItem(item)

    # Hover
    item.set_air_hover_pos(QPointF(150.0, 150.0))
    assert item._air_hover_pos is not None
    item.set_air_hover_pos(None)
    assert item._air_hover_pos is None

    # Air tap on AX2 model button (first button in top row)
    rect_ax2 = item._rect_btn_models[0]
    btn_ax2_scene = item.mapToScene(rect_ax2.center())
    tapped = item.handle_air_tap(btn_ax2_scene)
    assert tapped is True
    assert item.current_key() == "AX2"

    # Air tap on auto-spin button
    spin_btn_scene = item.mapToScene(item._rect_btn_spin.center())
    tapped_spin = item.handle_air_tap(spin_btn_scene)
    assert tapped_spin is True
    assert item.is_spinning() is True

    # Air drag to rotate 3D model
    center_scene = item.mapToScene(QPointF(item._proj_cx, item._proj_cy))
    item.handle_air_drag(center_scene)

    drag_right_scene = QPointF(center_scene.x() + 40.0, center_scene.y() - 20.0)
    old_yaw = item._yaw
    item.handle_air_drag(drag_right_scene)
    assert item._yaw != old_yaw

    item.close()


def test_stamp_to_canvas_and_lessonboard(qapp):
    """Test stamping vector 3D molecular diagram to LessonBoard."""
    board = LessonBoard()
    item = InteractiveVseprGeometryItem(pos=(300.0, 300.0), model_key="AX4")
    board.addItem(item)

    stamped_items = item.stamp_to_canvas(board)
    assert len(stamped_items) >= 5  # central atom, 4 ligands, bonds, title banner
    for itm in stamped_items:
        assert itm.scene() == board

    item.close()


def test_lessonboard_and_main_window_integration(qapp):
    """Test LessonBoard lifecycle methods and MainWindow action toggles."""
    win = MainWindow()
    board = win._scene
    assert not board.is_vsepr_geometry_visible()

    # Show item
    board.show_vsepr_geometry(True, model_key="AX3E")
    assert board.is_vsepr_geometry_visible()
    item = board.get_vsepr_geometry_item()
    assert item is not None
    assert item.isVisible()
    assert item.current_key() == "AX3E"

    # Hide and Toggle
    board.show_vsepr_geometry(False)
    assert not board.is_vsepr_geometry_visible()

    is_vis = board.toggle_vsepr_geometry()
    assert is_vis is True
    assert board.is_vsepr_geometry_visible()

    # MainWindow slot
    win._on_toggle_vsepr_geometry()
    assert not board.is_vsepr_geometry_visible()

    item.close()
    win.close()
