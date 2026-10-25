"""
Tests for InteractiveQuantumDiffractionItem (Quantum Double-Slit & de Broglie Matter Waves)
Milestone 209
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

from src.ui.interactive_quantum_diffraction import (
    InteractiveQuantumDiffractionItem,
    PRESETS,
    QuantumPreset,
    PLANCK_H,
    SPEED_C,
    M_ELECTRON,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def quantum_item(qapp):
    scene = QGraphicsScene()
    item = InteractiveQuantumDiffractionItem()
    scene.addItem(item)
    item._scene_ref = scene
    return item


def test_init_default_values(quantum_item):
    """Test initial attributes and physical invariants."""
    assert quantum_item.WIDTH == 920.0
    assert quantum_item.HEIGHT == 600.0
    assert quantum_item.current_preset.key == "electron_hitachi"
    assert not quantum_item._is_paused
    assert quantum_item._sim_time == 0.0
    assert len(quantum_item._particle_hits) == 0
    assert quantum_item._wavelength > 0.0
    assert quantum_item._wavelength < 1e-10  # pm range for 50 keV electron


def test_presets_cycling(quantum_item):
    """Test switching through all 5 educational presets."""
    preset_signals = []
    quantum_item.preset_changed.connect(preset_signals.append)

    for i, p in enumerate(PRESETS):
        quantum_item.set_preset(i)
        assert quantum_item.current_preset.key == p.key
        assert quantum_item._preset_index == i
        assert quantum_item._particle_type == p.particle_type
        assert quantum_item._wavelength > 0.0
        assert not math.isnan(quantum_item._wavelength)

    assert len(preset_signals) == len(PRESETS)
    assert preset_signals[-1] == PRESETS[-1].key


def test_de_broglie_wavelength_calc(quantum_item):
    """Test relativistic and non-relativistic de Broglie wavelength calculations."""
    # 1. 50 keV electron (relativistic)
    quantum_item.set_preset(0)
    assert abs(quantum_item._wavelength - 5.355e-12) < 0.1e-12

    # 2. 532 nm optical photon
    quantum_item.set_preset(1)
    assert abs(quantum_item._wavelength - 532e-9) < 1e-9

    # 3. Thermal neutron (0.025 eV)
    quantum_item.set_preset(3)
    assert abs(quantum_item._wavelength - 1.81e-10) < 0.1e-10


def test_intensity_computation_and_decoherence(quantum_item):
    """Test quantum Born probability density with and without which-way detector."""
    quantum_item.set_preset(0)  # Hitachi electron without which-way
    quantum_item.set_decoherence(0.0)

    # Center should be an interference maximum
    i_center = quantum_item.compute_intensity_at(0.0)
    assert i_center > 0.8

    # Check that intensity is bounded in [0, 1]
    for y in (-0.8, -0.4, 0.0, 0.4, 0.8):
        val = quantum_item.compute_intensity_at(y)
        assert 0.0 <= val <= 1.05

    # Switch on full which-way decoherence
    quantum_item.set_decoherence(1.0)
    assert quantum_item._which_way is True
    i_center_collapsed = quantum_item.compute_intensity_at(0.0)
    assert not math.isnan(i_center_collapsed)


def test_which_way_toggle(quantum_item):
    """Test toggling which-way detector."""
    quantum_item.set_decoherence(0.0)
    assert not quantum_item._which_way

    quantum_item.toggle_which_way()
    assert quantum_item._which_way
    assert quantum_item._decoherence == 1.0

    quantum_item.toggle_which_way()
    assert not quantum_item._which_way
    assert quantum_item._decoherence == 0.0


def test_particle_arrival_sampling(quantum_item):
    """Test stochastic Monte Carlo particle arrival."""
    quantum_item.set_preset(0)
    for _ in range(20):
        y = quantum_item.sample_particle_arrival()
        assert not math.isnan(y)
        assert -1.5 <= y <= 1.5


def test_step_simulation_particle_accumulation(quantum_item):
    """Test particle accumulation on phosphor screen over time."""
    quantum_item.clear_screen()
    assert len(quantum_item._particle_hits) == 0

    quantum_item.step_simulation(0.2)
    assert quantum_item._sim_time == 0.2
    assert len(quantum_item._particle_hits) > 0

    # Paused simulation should not accumulate
    quantum_item.toggle_pause()
    count_before = len(quantum_item._particle_hits)
    quantum_item.step_simulation(0.5)
    assert len(quantum_item._particle_hits) == count_before

    quantum_item.toggle_pause()
    quantum_item.clear_screen()
    assert len(quantum_item._particle_hits) == 0


def test_property_setters(quantum_item):
    """Test setter methods and clamping."""
    prop_changes = []
    quantum_item.property_changed.connect(lambda k, v: prop_changes.append((k, v)))

    # Slit separation d
    quantum_item.set_slit_separation(15.0)
    assert quantum_item._slit_d_um == 15.0
    quantum_item.set_slit_separation(9999.0)
    assert quantum_item._slit_d_um == 500.0
    quantum_item.set_slit_separation(20.0)
    assert quantum_item._slit_d_um == 20.0

    # Slit width a (clamped to d * 0.95)
    quantum_item.set_slit_width(2.5)
    assert quantum_item._slit_a_um == 2.5
    quantum_item.set_slit_width(50.0)
    assert quantum_item._slit_a_um == 20.0 * 0.95

    # Decoherence
    quantum_item.set_decoherence(0.45)
    assert quantum_item._decoherence == 0.45

    assert len(prop_changes) > 0


def test_serialization(quantum_item):
    """Test to_dict and from_dict serialization."""
    quantum_item.set_preset(3)  # thermal neutrons
    quantum_item.set_slit_separation(30.0)
    quantum_item.set_slit_width(4.0)
    quantum_item.set_decoherence(0.7)
    quantum_item.setPos(150.0, 90.0)

    data = quantum_item.to_dict()
    assert data["type"] == "InteractiveQuantumDiffractionItem"
    assert data["preset_key"] == "thermal_neutrons"
    assert data["slit_d_um"] == 30.0
    assert data["slit_a_um"] == 4.0
    assert data["decoherence"] == 0.7
    assert data["pos"] == [150.0, 90.0]

    # Reconstruct
    new_item = InteractiveQuantumDiffractionItem.from_dict(data)
    assert new_item.current_preset.key == "thermal_neutrons"
    assert new_item._slit_d_um == 30.0
    assert new_item._slit_a_um == 4.0
    assert new_item._decoherence == 0.7
    assert new_item.pos() == QPointF(150.0, 90.0)


def test_mouse_events_preset_buttons(quantum_item):
    """Test clicking preset buttons."""
    btn_rects = quantum_item._preset_button_rects()
    target_rect, target_preset = btn_rects[2]

    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(target_rect.center())
    event.setButton(Qt.LeftButton)

    quantum_item.mousePressEvent(event)
    assert quantum_item._preset_index == 2
    assert quantum_item.current_preset.key == target_preset.key


def test_mouse_events_actions(quantum_item):
    """Test clicking which-way button and clear screen button."""
    ctrl_rect = quantum_item._controls_rect()
    action_y = ctrl_rect.top() + 62.0

    # Toggle which-way button
    btn_which = QRectF(ctrl_rect.left() + 20, action_y, 180, 24)
    init_state = quantum_item._which_way
    event_which = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event_which.setPos(btn_which.center())
    event_which.setButton(Qt.LeftButton)
    quantum_item.mousePressEvent(event_which)
    assert quantum_item._which_way != init_state

    # Clear screen button
    quantum_item._particle_hits.append(0.5)
    btn_clear = QRectF(ctrl_rect.left() + 215, action_y, 160, 24)
    event_clear = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event_clear.setPos(btn_clear.center())
    event_clear.setButton(Qt.LeftButton)
    quantum_item.mousePressEvent(event_clear)
    assert len(quantum_item._particle_hits) == 0


def test_mouse_slider_drag(quantum_item):
    """Test dragging bottom slider."""
    ctrl_rect = quantum_item._controls_rect()
    col_w = (ctrl_rect.width() - 40.0) / 4.0
    cx = ctrl_rect.left() + 20.0
    track_y = ctrl_rect.top() + 36.0

    press_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    press_event.setPos(QPointF(cx + 10.0, track_y))
    press_event.setButton(Qt.LeftButton)
    quantum_item.mousePressEvent(press_event)
    assert quantum_item._dragging_slider == "slit_d"

    move_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseMove)
    move_event.setPos(QPointF(cx + col_w * 0.8, track_y))
    quantum_item.mouseMoveEvent(move_event)
    assert quantum_item._slit_d_um > 1.0

    release_event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMouseRelease)
    release_event.setPos(QPointF(cx + col_w * 0.8, track_y))
    quantum_item.mouseReleaseEvent(release_event)
    assert quantum_item._dragging_slider is None


def test_paint_rendering(quantum_item):
    """Test QPainter rendering for full coverage with no exceptions."""
    img = QImage(920, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    opt = QStyleOptionGraphicsItem()

    # Step simulation so particles and wavefronts exist
    quantum_item.set_preset(0)
    quantum_item.step_simulation(0.2)

    quantum_item.paint(painter, opt)
    painter.end()

    assert not img.isNull()
    nonzero = 0
    for y in (100, 250, 400, 520):
        for x in (100, 300, 500, 700):
            if img.pixelColor(x, y).alpha() > 0:
                nonzero += 1
    assert nonzero > 0


def test_lesson_board_integration(qapp):
    """Test show_quantum_diffraction, toggle_quantum_diffraction, is_quantum_diffraction_visible on LessonBoard."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert not board.is_quantum_diffraction_visible()
    assert board.get_quantum_diffraction_item() is None

    # Show item
    item = board.show_quantum_diffraction(True)
    assert item is not None
    assert board.is_quantum_diffraction_visible()
    assert board.get_quantum_diffraction_item() is item

    # Toggle hide
    res = board.toggle_quantum_diffraction()
    assert res is False
    assert not board.is_quantum_diffraction_visible()

    # Toggle show
    res = board.toggle_quantum_diffraction()
    assert res is True
    assert board.is_quantum_diffraction_visible()

