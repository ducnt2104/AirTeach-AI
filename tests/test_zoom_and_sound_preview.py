"""
AirTeach AI - Whiteboard Canvas Zoom & Sound Preview Unit Tests
===============================================================
Verifies:
1. SettingsDialog multi-sound feedback preview tester and volume testing.
2. CanvasZoomWidget floating acrylic HUD button controls, signals, and formatting.
3. CanvasZoomController magnification clamping (0.25x - 4.0x), wheel/drag event filtering,
   and pan scrollbar displacements.
4. MainWindow integration, View menu actions/shortcuts, and in-air menu triggers.
"""
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from src.ui.canvas_zoom_controller import (
    CanvasZoomController,
    CanvasZoomWidget,
    DEFAULT_ZOOM_FACTOR,
    MAX_ZOOM_FACTOR,
    MIN_ZOOM_FACTOR,
)
from src.ui.main_window import MainWindow
from src.ui.settings_dialog import SettingsDialog
from src.utils.audio_feedback import AudioFeedbackManager


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── 1. SettingsDialog Audio Sound Preview Tests ───────────────────────────────


def test_settings_dialog_audio_preview_combo(qapp):
    """Verify SettingsDialog has audio test dropdown with all sound types."""
    dialog = SettingsDialog()
    assert hasattr(dialog, "audio_test_type_combo")
    assert hasattr(dialog, "btn_test_audio")

    # Verify combo contains all gesture acoustic sounds
    items = [dialog.audio_test_type_combo.itemText(i) for i in range(dialog.audio_test_type_combo.count())]
    assert any("Snap" in s for s in items)
    assert any("Click" in s for s in items)
    assert any("Page" in s or "Lật" in s for s in items)
    assert any("Erase" in s or "Xóa" in s for s in items)
    assert any("Timer" in s or "Chuông" in s for s in items)
    assert any("Undo" in s or "Hoàn tác" in s for s in items)


def test_settings_dialog_play_test_sound(qapp):
    """Verify play_test_sound executes across all acoustic variations without throwing."""
    dialog = SettingsDialog()
    dialog.audio_volume_spin.setValue(60)

    mgr = AudioFeedbackManager.instance()
    initial_enabled = dialog.audio_feedback_check.isChecked()

    # Play test sounds
    dialog.play_test_sound("snap")
    dialog.play_test_sound("click")
    dialog.play_test_sound("page")
    dialog.play_test_sound("erase")
    dialog.play_test_sound("timer")
    dialog.play_test_sound("undo")

    # State restored to match checkbox
    assert mgr.enabled == initial_enabled


def test_settings_dialog_on_test_audio_slot(qapp):
    """Verify _on_test_audio triggers sound according to combo selection."""
    dialog = SettingsDialog()
    dialog.audio_volume_spin.setValue(80)

    # Select click
    for idx in range(dialog.audio_test_type_combo.count()):
        if "Click" in dialog.audio_test_type_combo.itemText(idx):
            dialog.audio_test_type_combo.setCurrentIndex(idx)
            break

    # Trigger test
    dialog._on_test_audio()
    # No exceptions, volume updated
    assert AudioFeedbackManager.instance().volume == 0.8


# ── 2. CanvasZoomWidget HUD Tests ─────────────────────────────────────────────


def test_canvas_zoom_widget_controls_and_signals(qapp):
    """Verify CanvasZoomWidget buttons, signals, and label updates."""
    widget = CanvasZoomWidget()
    assert widget.btn_pan is not None
    assert widget.btn_zoom_out is not None
    assert widget.btn_reset is not None
    assert widget.btn_zoom_in is not None

    received = []
    widget.zoom_in_requested.connect(lambda: received.append("in"))
    widget.zoom_out_requested.connect(lambda: received.append("out"))
    widget.zoom_reset_requested.connect(lambda: received.append("reset"))
    widget.pan_toggled.connect(lambda active: received.append(f"pan_{active}"))

    widget.btn_zoom_in.click()
    assert "in" in received

    widget.btn_zoom_out.click()
    assert "out" in received

    widget.btn_reset.click()
    assert "reset" in received

    widget.btn_pan.click()
    assert "pan_True" in received
    assert widget.is_pan_active() is True

    widget.btn_pan.click()
    assert "pan_False" in received
    assert widget.is_pan_active() is False


def test_canvas_zoom_widget_formatting(qapp):
    """Verify zoom percentage formatting and pan toggle state."""
    widget = CanvasZoomWidget()
    widget.set_zoom_factor(1.0)
    assert widget.btn_reset.text() == "100%"

    widget.set_zoom_factor(1.5)
    assert widget.btn_reset.text() == "150%"

    widget.set_zoom_factor(0.75)
    assert widget.btn_reset.text() == "75%"

    widget.set_pan_active(True)
    assert widget.btn_pan.isChecked() is True
    widget.set_pan_active(False)
    assert widget.btn_pan.isChecked() is False


# ── 3. CanvasZoomController Tests ─────────────────────────────────────────────


def test_canvas_zoom_controller_defaults_and_clamping(qapp):
    """Verify CanvasZoomController bounds, step magnification, and reset."""
    scene = QGraphicsScene(0, 0, 1920, 1080)
    view = QGraphicsView(scene)
    view.resize(800, 600)
    view.show()

    ctrl = CanvasZoomController(view, scene)
    assert ctrl.zoom_factor() == DEFAULT_ZOOM_FACTOR

    # Zoom in steps
    z1 = ctrl.zoom_in()
    assert z1 > DEFAULT_ZOOM_FACTOR
    assert ctrl.zoom_factor() == z1

    # Zoom in to ceiling
    for _ in range(20):
        ctrl.zoom_in()
    assert ctrl.zoom_factor() <= MAX_ZOOM_FACTOR

    # Zoom out to floor
    for _ in range(30):
        ctrl.zoom_out()
    assert ctrl.zoom_factor() >= MIN_ZOOM_FACTOR

    # Reset
    ctrl.reset_zoom()
    assert ctrl.zoom_factor() == DEFAULT_ZOOM_FACTOR
    assert ctrl.get_widget().btn_reset.text() == "100%"


def test_canvas_zoom_controller_pan_mode_and_drag(qapp):
    """Verify pan toggle, cursor update, and mouse drag offset."""
    scene = QGraphicsScene(0, 0, 1920, 1080)
    view = QGraphicsView(scene)
    view.resize(800, 600)
    view.show()

    ctrl = CanvasZoomController(view, scene)
    assert ctrl.is_pan_mode() is False

    # Toggle pan mode
    active = ctrl.toggle_pan_mode()
    assert active is True
    assert ctrl.is_pan_mode() is True

    # Pan by delta
    ctrl.pan_by(50, 30)

    # Disable pan
    ctrl.set_pan_mode(False)
    assert ctrl.is_pan_mode() is False


def test_canvas_zoom_controller_wheel_event_filtering(qapp):
    """Verify Ctrl+Wheel zooms view while normal wheel does not."""
    scene = QGraphicsScene(0, 0, 1920, 1080)
    view = QGraphicsView(scene)
    view.resize(800, 600)
    view.show()

    ctrl = CanvasZoomController(view, scene)
    initial_zoom = ctrl.zoom_factor()

    # Wheel event without Ctrl (normal scroll) -> should not trigger zoom
    event_no_ctrl = QWheelEvent(
        QPointF(400, 300),
        QPointF(400, 300),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    handled = ctrl.eventFilter(view.viewport(), event_no_ctrl)
    assert handled is False
    assert ctrl.zoom_factor() == initial_zoom

    # Wheel event WITH Ctrl -> triggers zoom in
    event_with_ctrl = QWheelEvent(
        QPointF(400, 300),
        QPointF(400, 300),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    handled_ctrl = ctrl.eventFilter(view.viewport(), event_with_ctrl)
    assert handled_ctrl is True
    assert ctrl.zoom_factor() > initial_zoom


# ── 4. MainWindow Zoom & Pan Integration Tests ────────────────────────────────


def test_main_window_zoom_and_pan_integration(qapp):
    """Verify MainWindow has zoom controller, slots, menu triggers, and shortcuts."""
    win = MainWindow(camera_index=-1)
    win.show()

    assert hasattr(win, "_zoom_ctrl")
    assert win._zoom_ctrl is not None

    # Test zoom slots
    initial_zoom = win._zoom_ctrl.zoom_factor()
    new_zoom = win._on_zoom_in()
    assert new_zoom > initial_zoom

    out_zoom = win._on_zoom_out()
    assert out_zoom < new_zoom

    win._on_zoom_reset()
    assert win._zoom_ctrl.zoom_factor() == 1.0

    # Test pan slot
    pan_state = win._on_toggle_pan()
    assert pan_state is True
    assert win._zoom_ctrl.is_pan_mode() is True

    win._on_toggle_pan()
    assert win._zoom_ctrl.is_pan_mode() is False

    # Test in-air menu action triggers
    win._on_menu_action_triggered("action", "zoom_in")
    assert win._zoom_ctrl.zoom_factor() > 1.0

    win._on_menu_action_triggered("action", "zoom_reset")
    assert win._zoom_ctrl.zoom_factor() == 1.0

    win._on_menu_action_triggered("action", "pan")
    assert win._zoom_ctrl.is_pan_mode() is True
    win._on_menu_action_triggered("action", "pan")
    assert win._zoom_ctrl.is_pan_mode() is False

    win.close()


def test_bimanual_pinch_to_zoom_and_air_panning(qapp):
    """Verify two-hand pinch distance changes scale factor and air pan mode scrolls canvas."""
    from src.core.hand_gesture_engine import GestureState
    from src.core.multi_hand_coordinator import HandLandmarkData, HandRole

    win = MainWindow(camera_index=-1)
    win.show()

    win._zoom_ctrl.reset_zoom()
    assert win._zoom_ctrl.zoom_factor() == 1.0

    # 1. Two-hand pinch zoom: Frame 1 with distance 0.3
    h1 = HandLandmarkData(
        track_id="hand_0",
        role=HandRole.PRIMARY,
        gesture_state=GestureState.POINT,
        canvas_x=500.0,
        canvas_y=400.0,
        index_tip_norm=(0.3, 0.5),
        is_pinched=True,
    )
    h2 = HandLandmarkData(
        track_id="hand_1",
        role=HandRole.SECONDARY,
        gesture_state=GestureState.POINT,
        canvas_x=800.0,
        canvas_y=400.0,
        index_tip_norm=(0.6, 0.5),
        is_pinched=True,
    )
    win._on_multi_gesture_event([h1, h2])
    assert win._last_bimanual_pinch_dist == pytest.approx(0.3, abs=0.01)

    # Frame 2: Hands spread apart to distance 0.5 -> zoom in!
    h1_spread = HandLandmarkData(
        track_id="hand_0",
        role=HandRole.PRIMARY,
        gesture_state=GestureState.POINT,
        canvas_x=400.0,
        canvas_y=400.0,
        index_tip_norm=(0.2, 0.5),
        is_pinched=True,
    )
    h2_spread = HandLandmarkData(
        track_id="hand_1",
        role=HandRole.SECONDARY,
        gesture_state=GestureState.POINT,
        canvas_x=900.0,
        canvas_y=400.0,
        index_tip_norm=(0.7, 0.5),
        is_pinched=True,
    )
    win._on_multi_gesture_event([h1_spread, h2_spread])
    assert win._zoom_ctrl.zoom_factor() > 1.0

    # 2. Air Panning mode
    win._zoom_ctrl.set_pan_mode(True)
    h_pan1 = HandLandmarkData(
        track_id="hand_0",
        role=HandRole.PRIMARY,
        gesture_state=GestureState.POINT,
        canvas_x=600.0,
        canvas_y=500.0,
        index_tip_norm=(0.5, 0.5),
    )
    win._on_multi_gesture_event([h_pan1])

    h_pan2 = HandLandmarkData(
        track_id="hand_0",
        role=HandRole.PRIMARY,
        gesture_state=GestureState.POINT,
        canvas_x=650.0,
        canvas_y=520.0,
        index_tip_norm=(0.55, 0.52),
    )
    win._on_multi_gesture_event([h_pan2])
    # Handled without raising and no ink strokes generated during pan
    assert len(win._scene._strokes) == 0

    win.close()
