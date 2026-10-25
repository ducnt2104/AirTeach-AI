"""
Tests for InteractiveLavalNozzleItem (Compressible Flow & de Laval Nozzle Simulator)
Milestone 210
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

from src.ui.interactive_laval_nozzle import (
    InteractiveLavalNozzleItem,
    PRESETS,
    LavalPreset,
    GAMMA,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def nozzle_item(qapp):
    scene = QGraphicsScene()
    item = InteractiveLavalNozzleItem()
    scene.addItem(item)
    item._scene_ref = scene
    return item


def test_init_default_values(nozzle_item):
    """Test initial attributes and physical invariants."""
    assert nozzle_item.WIDTH == 920.0
    assert nozzle_item.HEIGHT == 600.0
    assert nozzle_item.current_preset.key == "ideally_expanded"
    assert not nozzle_item._is_paused
    assert nozzle_item._sim_time == 0.0
    assert nozzle_item._area_ratio > 1.0
    assert nozzle_item._p0_bar > 0.0


def test_presets_cycling(nozzle_item):
    """Test switching through all 5 educational presets."""
    preset_signals = []
    nozzle_item.preset_changed.connect(preset_signals.append)

    for i, p in enumerate(PRESETS):
        nozzle_item.set_preset(i)
        assert nozzle_item.current_preset.key == p.key
        assert nozzle_item._preset_index == i
        state = nozzle_item.compute_flow_state()
        assert state["regime"] is not None
        assert not math.isnan(state["thrust_n"])
        assert not math.isnan(state["mach_exit"])

    assert len(preset_signals) == len(PRESETS)
    assert preset_signals[-1] == PRESETS[-1].key


def test_isentropic_solver_mach_area(nozzle_item):
    """Test Newton-Raphson isentropic Mach-Area solver."""
    # At throat A/A* = 1.0 => Mach = 1.0
    m_throat = nozzle_item.solve_mach_from_area(1.0, supersonic=False)
    assert abs(m_throat - 1.0) < 1e-3

    # For A/A* = 2.0:
    # Subsonic solution should be around 0.3
    m_sub = nozzle_item.solve_mach_from_area(2.0, supersonic=False)
    assert 0.1 < m_sub < 0.5
    # Supersonic solution should be around 2.2
    m_sup = nozzle_item.solve_mach_from_area(2.0, supersonic=True)
    assert 1.8 < m_sup < 2.5


def test_flow_regimes_evaluation(nozzle_item):
    """Test flow regime diagnosis and rocket thrust calculations."""
    # 1. Ideally expanded
    nozzle_item.set_preset(0)
    state_ideal = nozzle_item.compute_flow_state()
    assert state_ideal["regime"] == "ideal"
    assert state_ideal["mach_exit"] > 2.5
    assert state_ideal["thrust_n"] > 1000.0
    assert state_ideal["isp_s"] > 180.0

    # 2. Shock diamonds (overexpanded)
    nozzle_item.set_preset(1)
    state_diamonds = nozzle_item.compute_flow_state()
    assert state_diamonds["regime"] == "overexpanded"
    assert state_diamonds["pe_bar"] < nozzle_item._pb_bar

    # 3. Normal shock in nozzle
    nozzle_item.set_preset(2)
    state_shock = nozzle_item.compute_flow_state()
    assert state_shock["regime"] == "normal_shock"
    assert state_shock["normal_shock_x"] is not None
    assert 0.5 <= state_shock["normal_shock_x"] <= 1.0

    # 4. Underexpanded
    nozzle_item.set_preset(3)
    state_under = nozzle_item.compute_flow_state()
    assert state_under["regime"] == "underexpanded"
    assert state_under["pe_bar"] > nozzle_item._pb_bar

    # 5. Subsonic venturi
    nozzle_item.set_preset(4)
    state_sub = nozzle_item.compute_flow_state()
    assert state_sub["regime"] == "subsonic"
    assert state_sub["mach_exit"] < 1.0


def test_step_simulation_animation(nozzle_item):
    """Test time stepping and pause toggle."""
    nozzle_item.step_simulation(0.2)
    assert nozzle_item._sim_time == 0.2

    nozzle_item.toggle_pause()
    assert nozzle_item._is_paused
    nozzle_item.step_simulation(0.5)
    assert nozzle_item._sim_time == 0.2

    nozzle_item.toggle_pause()
    assert not nozzle_item._is_paused


def test_property_setters(nozzle_item):
    """Test physical parameter setters and bounds clamping."""
    prop_changes = []
    nozzle_item.property_changed.connect(lambda k, v: prop_changes.append((k, v)))

    # Chamber pressure P0
    nozzle_item.set_chamber_pressure(45.0)
    assert nozzle_item._p0_bar == 45.0
    nozzle_item.set_chamber_pressure(999.0)
    assert nozzle_item._p0_bar == 100.0
    nozzle_item.set_chamber_pressure(-5.0)
    assert nozzle_item._p0_bar == 0.5

    # Back pressure Pb
    nozzle_item.set_back_pressure(0.5)
    assert nozzle_item._pb_bar == 0.5
    nozzle_item.set_back_pressure(10.0)
    assert nozzle_item._pb_bar == 5.0

    # Area ratio
    nozzle_item.set_area_ratio(5.5)
    assert nozzle_item._area_ratio == 5.5

    # Chamber temp T0
    nozzle_item.set_chamber_temp(3200.0)
    assert nozzle_item._t0_k == 3200.0

    assert len(prop_changes) > 0


def test_serialization(nozzle_item):
    """Test to_dict and from_dict serialization."""
    nozzle_item.set_preset(1)
    nozzle_item.set_chamber_pressure(25.0)
    nozzle_item.set_back_pressure(0.8)
    nozzle_item.set_area_ratio(3.8)
    nozzle_item.set_chamber_temp(2600.0)
    nozzle_item.setPos(140.0, 75.0)

    data = nozzle_item.to_dict()
    assert data["type"] == "InteractiveLavalNozzleItem"
    assert data["preset_key"] == "shock_diamonds"
    assert data["p0_bar"] == 25.0
    assert data["pb_bar"] == 0.8
    assert data["area_ratio"] == 3.8
    assert data["t0_k"] == 2600.0
    assert data["pos"] == [140.0, 75.0]

    # Reconstruct
    new_item = InteractiveLavalNozzleItem.from_dict(data)
    assert new_item.current_preset.key == "shock_diamonds"
    assert new_item._p0_bar == 25.0
    assert new_item._pb_bar == 0.8
    assert new_item._area_ratio == 3.8
    assert new_item._t0_k == 2600.0
    assert new_item.pos() == QPointF(140.0, 75.0)


def test_mouse_events_preset_buttons(nozzle_item):
    """Test clicking preset buttons."""
    btn_rects = nozzle_item._preset_button_rects()
    target_rect, target_preset = btn_rects[3]

    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(target_rect.center())
    event.setButton(Qt.LeftButton)

    nozzle_item.mousePressEvent(event)
    assert nozzle_item._preset_index == 3
    assert nozzle_item.current_preset.key == target_preset.key


def test_mouse_events_pause_badge(nozzle_item):
    """Test clicking pause status badge."""
    badge_rect = QRectF(nozzle_item.WIDTH - 150.0, 22.0, 122.0, 26.0)
    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(badge_rect.center())
    event.setButton(Qt.LeftButton)

    nozzle_item.mousePressEvent(event)
    assert nozzle_item._is_paused


def test_mouse_slider_drag(nozzle_item):
    """Test dragging bottom slider."""
    ctrl_rect = nozzle_item._controls_rect()
    col_w = (ctrl_rect.width() - 40.0) / 4.0
    cx = ctrl_rect.left() + 20.0
    track_y = ctrl_rect.top() + 36.0

    press_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    press_event.setPos(QPointF(cx + 10.0, track_y))
    press_event.setButton(Qt.LeftButton)
    nozzle_item.mousePressEvent(press_event)
    assert nozzle_item._dragging_slider == "p0"

    move_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseMove)
    move_event.setPos(QPointF(cx + col_w * 0.75, track_y))
    nozzle_item.mouseMoveEvent(move_event)
    assert nozzle_item._p0_bar > 10.0

    release_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseRelease)
    release_event.setPos(QPointF(cx + col_w * 0.75, track_y))
    nozzle_item.mouseReleaseEvent(release_event)
    assert nozzle_item._dragging_slider is None


def test_paint_rendering(nozzle_item):
    """Test QPainter rendering for full coverage with no exceptions."""
    img = QImage(920, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    opt = QStyleOptionGraphicsItem()

    # Paint for each regime
    for preset_idx in range(len(PRESETS)):
        nozzle_item.set_preset(preset_idx)
        nozzle_item.paint(painter, opt)

    painter.end()

    assert not img.isNull()
    nonzero = 0
    for y in (100, 250, 400, 520):
        for x in (100, 300, 500, 700):
            if img.pixelColor(x, y).alpha() > 0:
                nonzero += 1
    assert nonzero > 0


def test_lesson_board_integration(qapp):
    """Test show_laval_nozzle, toggle_laval_nozzle, is_laval_nozzle_visible on LessonBoard."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert not board.is_laval_nozzle_visible()
    assert board.get_laval_nozzle_item() is None

    # Show item
    item = board.show_laval_nozzle(True)
    assert item is not None
    assert board.is_laval_nozzle_visible()
    assert board.get_laval_nozzle_item() is item

    # Toggle hide
    res = board.toggle_laval_nozzle()
    assert res is False
    assert not board.is_laval_nozzle_visible()

    # Toggle show
    res = board.toggle_laval_nozzle()
    assert res is True
    assert board.is_laval_nozzle_visible()

