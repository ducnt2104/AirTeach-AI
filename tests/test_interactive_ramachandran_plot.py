"""
Tests for InteractiveRamachandranPlotItem (Biochemistry & Protein Folding)
Milestone 211
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
)

from src.ui.interactive_ramachandran_plot import (
    InteractiveRamachandranPlotItem,
    PRESETS,
    RamachandranPreset,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def rama_item(qapp):
    scene = QGraphicsScene()
    item = InteractiveRamachandranPlotItem()
    scene.addItem(item)
    item._scene_ref = scene
    return item


def test_init_default_values(rama_item):
    """Test initial attributes and physical invariants."""
    assert rama_item.WIDTH == 920.0
    assert rama_item.HEIGHT == 600.0
    assert rama_item.current_preset.key == "alpha_helix"
    assert not rama_item._is_paused
    assert rama_item._sim_time == 0.0
    assert rama_item._phi_deg == -57.0
    assert rama_item._psi_deg == -47.0


def test_presets_cycling(rama_item):
    """Test switching through all 5 educational presets."""
    preset_signals = []
    rama_item.preset_changed.connect(preset_signals.append)

    for i, p in enumerate(PRESETS):
        rama_item.set_preset(i)
        assert rama_item.current_preset.key == p.key
        assert rama_item._preset_index == i
        eval_data = rama_item.evaluate_conformation()
        assert eval_data["region"] in ("core", "allowed", "generously_allowed", "disallowed")

    assert len(preset_signals) == len(PRESETS)
    assert preset_signals[-1] == PRESETS[-1].key


def test_evaluate_conformation_details(rama_item):
    """Test secondary structure classification and steric clash physics."""
    # 1. Alpha-helix
    rama_item.set_preset(0)
    data_a = rama_item.evaluate_conformation()
    assert data_a["region"] == "core"
    assert data_a["h_bond"] is True
    assert data_a["is_clash"] is False

    # 2. Beta-sheet
    rama_item.set_preset(1)
    data_b = rama_item.evaluate_conformation()
    assert data_b["region"] == "core"
    assert data_b["is_clash"] is False

    # 3. Steric clash (eclipsed 0, 0)
    rama_item.set_preset(4)
    data_clash = rama_item.evaluate_conformation()
    assert data_clash["is_clash"] is True
    assert data_clash["region"] == "disallowed"
    assert data_clash["clash_severity"] > 0.5


def test_residue_specific_constraints(rama_item):
    """Test Glycine flexibility and Proline conformational locking."""
    # Proline locks phi in [-80, -50]
    rama_item.set_residue_type("proline")
    rama_item.set_phi(-120.0)
    assert rama_item._phi_deg == -80.0
    rama_item.set_phi(20.0)
    assert rama_item._phi_deg == -50.0

    # Glycine allows positive phi region
    rama_item.set_residue_type("glycine")
    rama_item.set_phi(57.0)
    rama_item.set_psi(47.0)
    data_gly = rama_item.evaluate_conformation()
    assert data_gly["region"] == "core"


def test_step_simulation(rama_item):
    """Test time stepping and pause toggle."""
    rama_item.step_simulation(0.2)
    assert rama_item._sim_time == 0.2

    rama_item.toggle_pause()
    assert rama_item._is_paused
    rama_item.step_simulation(0.5)
    assert rama_item._sim_time == 0.2

    rama_item.toggle_pause()
    assert not rama_item._is_paused


def test_property_setters(rama_item):
    """Test dihedral angle setters and bounds clamping."""
    prop_changes = []
    rama_item.property_changed.connect(lambda k, v: prop_changes.append((k, v)))
    rama_item.set_residue_type("general")

    # Phi in [-180, +180]
    rama_item.set_phi(45.0)
    assert rama_item._phi_deg == 45.0
    rama_item.set_phi(999.0)
    assert rama_item._phi_deg == 180.0
    rama_item.set_phi(-999.0)
    assert rama_item._phi_deg == -180.0

    # Psi in [-180, +180]
    rama_item.set_psi(-60.0)
    assert rama_item._psi_deg == -60.0
    rama_item.set_psi(500.0)
    assert rama_item._psi_deg == 180.0

    assert len(prop_changes) > 0


def test_serialization(rama_item):
    """Test to_dict and from_dict serialization."""
    rama_item.set_preset(2)
    rama_item.set_phi(-80.0)
    rama_item.set_psi(150.0)
    rama_item.setPos(130.0, 60.0)

    data = rama_item.to_dict()
    assert data["type"] == "InteractiveRamachandranPlotItem"
    assert data["preset_key"] == "collagen_pii"
    assert data["phi_deg"] == -80.0
    assert data["psi_deg"] == 150.0
    assert data["pos"] == [130.0, 60.0]

    # Reconstruct
    new_item = InteractiveRamachandranPlotItem.from_dict(data)
    assert new_item.current_preset.key == "collagen_pii"
    assert new_item._phi_deg == -80.0
    assert new_item._psi_deg == 150.0
    assert new_item.pos() == QPointF(130.0, 60.0)


def test_mouse_events_preset_buttons(rama_item):
    """Test clicking preset buttons."""
    btn_rects = rama_item._preset_button_rects()
    target_rect, target_preset = btn_rects[1]

    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(target_rect.center())
    event.setButton(Qt.LeftButton)

    rama_item.mousePressEvent(event)
    assert rama_item._preset_index == 1
    assert rama_item.current_preset.key == target_preset.key


def test_mouse_events_plot_drag(rama_item):
    """Test dragging on the 2D Ramachandran contour plot."""
    p_rect = rama_item._plot_rect()
    center_pt = p_rect.center()

    press_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    press_event.setPos(center_pt)
    press_event.setButton(Qt.LeftButton)
    rama_item.mousePressEvent(press_event)
    assert rama_item._dragging_probe is True

    # Move to top-left quadrant (negative phi, positive psi)
    move_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseMove)
    move_event.setPos(QPointF(p_rect.left() + 50.0, p_rect.top() + 50.0))
    rama_item.mouseMoveEvent(move_event)
    assert rama_item._phi_deg < 0.0
    assert rama_item._psi_deg > 0.0

    release_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseRelease)
    release_event.setPos(QPointF(p_rect.left() + 50.0, p_rect.top() + 50.0))
    rama_item.mouseReleaseEvent(release_event)
    assert rama_item._dragging_probe is False


def test_mouse_slider_drag(rama_item):
    """Test dragging bottom slider."""
    ctrl_rect = rama_item._controls_rect()
    col_w = (ctrl_rect.width() - 50.0) / 2.0
    cx = ctrl_rect.left() + 20.0
    track_y = ctrl_rect.top() + 36.0

    press_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    press_event.setPos(QPointF(cx + 20.0, track_y))
    press_event.setButton(Qt.LeftButton)
    rama_item.mousePressEvent(press_event)
    assert rama_item._dragging_slider == "phi"

    move_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseMove)
    move_event.setPos(QPointF(cx + col_w * 0.8, track_y))
    rama_item.mouseMoveEvent(move_event)
    assert rama_item._phi_deg > 0.0

    release_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseRelease)
    release_event.setPos(QPointF(cx + col_w * 0.8, track_y))
    rama_item.mouseReleaseEvent(release_event)
    assert rama_item._dragging_slider is None


def test_paint_rendering(rama_item):
    """Test QPainter rendering for full coverage with no exceptions."""
    img = QImage(920, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    opt = QStyleOptionGraphicsItem()

    # Paint for each preset (including steric clash and glycine)
    for preset_idx in range(len(PRESETS)):
        rama_item.set_preset(preset_idx)
        rama_item.paint(painter, opt)

    painter.end()

    assert not img.isNull()
    nonzero = 0
    for y in (100, 250, 400, 520):
        for x in (100, 300, 500, 700):
            if img.pixelColor(x, y).alpha() > 0:
                nonzero += 1
    assert nonzero > 0


def test_lesson_board_integration(qapp):
    """Test show_ramachandran_plot, toggle_ramachandran_plot, is_ramachandran_plot_visible on LessonBoard."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert not board.is_ramachandran_plot_visible()
    assert board.get_ramachandran_plot_item() is None

    # Show item
    item = board.show_ramachandran_plot(True)
    assert item is not None
    assert board.is_ramachandran_plot_visible()
    assert board.get_ramachandran_plot_item() is item

    # Toggle hide
    res = board.toggle_ramachandran_plot()
    assert res is False
    assert not board.is_ramachandran_plot_visible()

    # Toggle show
    res = board.toggle_ramachandran_plot()
    assert res is True
    assert board.is_ramachandran_plot_visible()

