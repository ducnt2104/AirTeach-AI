"""
Tests for InteractiveHeatTransferItem (Thermodynamics & Heat Transfer Simulator)
Milestone 208
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
)

from src.ui.interactive_heat_transfer import (
    InteractiveHeatTransferItem,
    PRESETS,
    HeatTransferPreset,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def heat_item(qapp):
    scene = QGraphicsScene()
    item = InteractiveHeatTransferItem()
    scene.addItem(item)
    item._scene_ref = scene
    return item


def test_init_default_values(heat_item):
    """Test initial attributes and physical dimension invariants."""
    assert heat_item.WIDTH == 900.0
    assert heat_item.HEIGHT == 600.0
    assert heat_item.NX == 60
    assert heat_item.NY == 40
    assert heat_item.current_preset.key == "rod_conduction"
    assert not heat_item._is_paused
    assert heat_item._sim_time == 0.0

    # Temperature field initialized
    assert heat_item._temp.shape == (40, 60)
    assert np.all(heat_item._temp >= heat_item._t_ambient)
    assert not np.isnan(heat_item._temp).any()


def test_presets_cycling(heat_item):
    """Test switching through all 5 educational presets."""
    preset_signals = []
    heat_item.preset_changed.connect(preset_signals.append)

    for i, p in enumerate(PRESETS):
        heat_item.set_preset(i)
        assert heat_item.current_preset.key == p.key
        assert heat_item._preset_index == i
        assert not np.isnan(heat_item._temp).any()
        assert not np.isnan(heat_item._k_map).any()

    assert len(preset_signals) == len(PRESETS)
    assert preset_signals[-1] == PRESETS[-1].key


def test_property_setters(heat_item):
    """Test thermal parameter setters and boundary clamping."""
    prop_changes = []
    heat_item.property_changed.connect(lambda k, v: prop_changes.append((k, v)))

    # Thermal conductivity
    heat_item.set_thermal_conductivity(250.0)
    assert heat_item._k == 250.0
    heat_item.set_thermal_conductivity(9999.0)
    assert heat_item._k == 500.0
    heat_item.set_thermal_conductivity(-10.0)
    assert heat_item._k == 0.01

    # Convection coefficient
    heat_item.set_convection_coeff(50.0)
    assert heat_item._h == 50.0
    heat_item.set_convection_coeff(999.0)
    assert heat_item._h == 200.0
    heat_item.set_convection_coeff(-5.0)
    assert heat_item._h == 1.0

    # Heat source power
    heat_item.set_heat_source_power(80.0)
    assert heat_item._heat_source_w == 80.0
    heat_item.set_heat_source_power(-20.0)
    assert heat_item._heat_source_w == 0.0

    # Ambient temperature
    heat_item.set_ambient_temp(30.0)
    assert heat_item._t_ambient == 30.0
    heat_item.set_ambient_temp(100.0)
    assert heat_item._t_ambient == 60.0

    assert len(prop_changes) > 0


def test_ftcs_step_simulation(heat_item):
    """Test FTCS 2D transient heat diffusion forward step."""
    heat_item.set_preset(0)  # rod conduction
    t_start = heat_item._temp.copy()

    # Step simulation forward by 0.5s
    heat_item.step_simulation(0.5)
    assert heat_item._sim_time == 0.5
    assert not np.isnan(heat_item._temp).any()
    assert not np.isinf(heat_item._temp).any()

    # Center temperature should increase due to diffusion from left boundary
    mid_x = heat_item.NX // 4
    mid_y = heat_item.NY // 2
    assert heat_item._temp[mid_y, mid_x] >= t_start[mid_y, mid_x]


def test_cpu_fins_heat_generation(heat_item):
    """Test volumetric heat generation in CPU fins preset."""
    heat_item.set_preset(1)  # cpu_fins
    assert heat_item.current_preset.key == "cpu_fins"
    assert np.any(heat_item._q_dot > 0)

    # Step forward and verify hotspot temperature rises
    initial_max = float(np.max(heat_item._temp))
    heat_item.step_simulation(0.2)
    new_max = float(np.max(heat_item._temp))
    assert new_max >= initial_max


def test_pause_and_reset(heat_item):
    """Test pause toggle and simulation reset."""
    heat_item.set_preset(0)
    heat_item.step_simulation(0.3)
    time_before = heat_item._sim_time

    heat_item.toggle_pause()
    assert heat_item._is_paused

    # Stepping while paused should not advance time or state
    heat_item.step_simulation(0.5)
    assert heat_item._sim_time == time_before

    heat_item.toggle_pause()
    assert not heat_item._is_paused

    # Reset
    heat_item.reset_simulation()
    assert heat_item._sim_time == 0.0


def test_probe_reading(heat_item):
    """Test virtual temperature probe reading within and outside canvas."""
    c_rect = heat_item._canvas_rect()
    heat_item._probe_pos = QPointF(c_rect.center().x(), c_rect.center().y())

    t_c, grad_m, flux_w, biot = heat_item.get_probe_reading()
    assert not math.isnan(t_c)
    assert not math.isnan(grad_m)
    assert not math.isnan(flux_w)
    assert not math.isnan(biot)
    assert t_c >= 0.0
    assert grad_m >= 0.0
    assert flux_w >= 0.0

    # Probe outside canvas returns default ambient
    heat_item._probe_pos = QPointF(-100.0, -100.0)
    t_out, grad_out, flux_out, biot_out = heat_item.get_probe_reading()
    assert t_out == heat_item._t_ambient
    assert grad_out == 0.0


def test_serialization(heat_item):
    """Test serialization to_dict and restoration from_dict."""
    heat_item.set_preset(2)
    heat_item.set_thermal_conductivity(85.0)
    heat_item.set_convection_coeff(42.0)
    heat_item.set_heat_source_power(55.0)
    heat_item.setPos(120.0, 80.0)
    heat_item._probe_active = True
    heat_item._probe_pos = QPointF(200.0, 150.0)

    data = heat_item.to_dict()
    assert data["type"] == "InteractiveHeatTransferItem"
    assert data["preset_key"] == "convection_plume"
    assert data["k"] == 85.0
    assert data["h"] == 42.0
    assert data["heat_source_w"] == 55.0
    assert data["pos"] == [120.0, 80.0]
    assert data["probe_active"] is True
    assert data["probe_pos"] == [200.0, 150.0]

    # Reconstruct
    new_item = InteractiveHeatTransferItem.from_dict(data)
    assert new_item.current_preset.key == "convection_plume"
    assert new_item._k == 85.0
    assert new_item._h == 42.0
    assert new_item._heat_source_w == 55.0
    assert new_item.pos() == QPointF(120.0, 80.0)
    assert new_item._probe_active is True
    assert new_item._probe_pos == QPointF(200.0, 150.0)


def test_mouse_events_preset_buttons(heat_item):
    """Test mouse clicking on preset tabs."""
    btn_rects = heat_item._preset_button_rects()
    target_rect, target_preset = btn_rects[3]

    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(target_rect.center())
    event.setButton(Qt.LeftButton)

    heat_item.mousePressEvent(event)
    assert heat_item._preset_index == 3
    assert heat_item.current_preset.key == target_preset.key


def test_mouse_events_actions_and_badge(heat_item):
    """Test clicking pause badge, probe button, and flux vector button."""
    # Click pause badge
    badge_rect = QRectF(heat_item.WIDTH - 150.0, 22.0, 122.0, 26.0)
    event_pause = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event_pause.setPos(badge_rect.center())
    event_pause.setButton(Qt.LeftButton)
    heat_item.mousePressEvent(event_pause)
    assert heat_item._is_paused

    # Click probe toggle button
    ctrl_rect = heat_item._controls_rect()
    btn_probe = QRectF(ctrl_rect.left() + 20, ctrl_rect.top() + 62.0, 160, 24)
    event_probe = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event_probe.setPos(btn_probe.center())
    event_probe.setButton(Qt.LeftButton)
    heat_item.mousePressEvent(event_probe)
    assert heat_item._probe_active

    # Click flux vector toggle button
    btn_flux = QRectF(ctrl_rect.left() + 190, ctrl_rect.top() + 62.0, 160, 24)
    init_flux = heat_item._show_flux_vectors
    event_flux = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event_flux.setPos(btn_flux.center())
    event_flux.setButton(Qt.LeftButton)
    heat_item.mousePressEvent(event_flux)
    assert heat_item._show_flux_vectors != init_flux


def test_mouse_slider_drag(heat_item):
    """Test dragging bottom slider track."""
    ctrl_rect = heat_item._controls_rect()
    col_w = (ctrl_rect.width() - 40.0) / 4.0
    cx = ctrl_rect.left() + 20.0
    track_y = ctrl_rect.top() + 36.0

    press_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    press_event.setPos(QPointF(cx + 20.0, track_y))
    press_event.setButton(Qt.LeftButton)
    heat_item.mousePressEvent(press_event)
    assert heat_item._dragging_slider == "k"

    move_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseMove)
    move_event.setPos(QPointF(cx + col_w * 0.75, track_y))
    heat_item.mouseMoveEvent(move_event)
    assert heat_item._k > 100.0

    release_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseRelease)
    release_event.setPos(QPointF(cx + col_w * 0.75, track_y))
    heat_item.mouseReleaseEvent(release_event)
    assert heat_item._dragging_slider is None


def test_paint_rendering(heat_item):
    """Test QPainter rendering for full coverage with no exceptions."""
    img = QImage(900, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    opt = QStyleOptionGraphicsItem()

    # Step simulation so isotherms and flux vectors are active
    heat_item.set_preset(1)
    heat_item._probe_active = True
    heat_item.step_simulation(0.2)

    painter.paint(painter, opt) if hasattr(painter, 'paint') else heat_item.paint(painter, opt)
    painter.end()

    assert not img.isNull()
    # Check that pixel values were written
    nonzero = 0
    for y in (100, 250, 400, 520):
        for x in (100, 300, 500, 700):
            if img.pixelColor(x, y).alpha() > 0:
                nonzero += 1
    assert nonzero > 0


def test_lesson_board_integration(qapp):
    """Test show_heat_transfer, toggle_heat_transfer, is_heat_transfer_visible on LessonBoard."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert not board.is_heat_transfer_visible()
    assert board.get_heat_transfer_item() is None

    # Show item
    item = board.show_heat_transfer(True)
    assert item is not None
    assert board.is_heat_transfer_visible()
    assert board.get_heat_transfer_item() is item

    # Toggle hide
    res = board.toggle_heat_transfer()
    assert res is False
    assert not board.is_heat_transfer_visible()

    # Toggle show
    res = board.toggle_heat_transfer()
    assert res is True
    assert board.is_heat_transfer_visible()

