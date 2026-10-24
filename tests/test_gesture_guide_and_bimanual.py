"""
AirTeach AI - In-Air Gesture Guide HUD & Bimanual Tuning Tests
==============================================================
Verifies GestureGuideOverlay HUD card, live gesture active lighting,
dragging, and two-hand bimanual tuning for Spotlight aperture and Loupe magnification.
"""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.core.hand_gesture_engine import GestureState
from src.core.multi_hand_coordinator import HandLandmarkData, HandRole
from src.ui.gesture_guide_overlay import GestureCard, GestureGuideOverlay
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gesture_card_properties_and_states(qapp):
    """Verify GestureCard styling and state updates."""
    card = GestureCard("✌️", "Air Pen", "Draw in air", gesture_enum=GestureState.DRAW)
    assert card._is_active is False

    card.set_active(True)
    assert card._is_active is True
    assert "00ff88" in card.styleSheet()

    card.set_active(False)
    assert card._is_active is False


def test_gesture_guide_overlay_lifecycle_and_drag(qapp):
    """Verify GestureGuideOverlay cards initialization, dragging, and paintEvent."""
    overlay = GestureGuideOverlay()
    assert len(overlay._cards) == 6
    assert "DRAW" in overlay._cards
    assert "POINT" in overlay._cards
    assert "SELECT" in overlay._cards
    assert "PINCH" in overlay._cards
    assert "PAUSE" in overlay._cards
    assert "SWIPE" in overlay._cards

    # Test live active gesture lighting
    overlay.update_active_gesture(GestureState.DRAW)
    assert overlay._cards["DRAW"]._is_active is True
    assert overlay._cards["POINT"]._is_active is False

    overlay.update_active_gesture(GestureState.POINT)
    assert overlay._cards["DRAW"]._is_active is False
    assert overlay._cards["POINT"]._is_active is True

    overlay.update_active_gesture(GestureState.NONE, is_pinched=True)
    assert overlay._cards["PINCH"]._is_active is True

    # Paint event
    overlay.repaint()

    # Dragging simulation
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPoint(10, 10),
        QPoint(100, 100),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(press)
    assert overlay._drag_start is not None

    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPoint(25, 25),
        QPoint(115, 115),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseMoveEvent(move)

    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPoint(25, 25),
        QPoint(115, 115),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseReleaseEvent(release)
    assert overlay._drag_start is None

    # Toggle & close
    is_vis = overlay.toggle_guide()
    assert is_vis is True
    assert overlay.isVisible() is True

    is_vis2 = overlay.toggle_guide()
    assert is_vis2 is False
    assert overlay.isVisible() is False


def test_main_window_gesture_guide_and_bimanual_tuning(qapp):
    """Verify MainWindow integrates gesture guide HUD and two-hand bimanual spotlight/loupe tuning."""
    win = MainWindow(camera_index=-1)
    win.show()

    # 1. Toggle gesture guide HUD
    assert win._gesture_guide_overlay is None
    vis = win._on_toggle_gesture_guide()
    assert vis is True
    assert win._gesture_guide_overlay is not None
    assert win._gesture_guide_overlay.isVisible() is True

    # 2. Test Bimanual Spotlight aperture modulation
    win._on_toggle_spotlight()
    assert win._teaching_mode == "SPOTLIGHT"

    h_primary = HandLandmarkData(
        track_id="hand_0",
        role=HandRole.PRIMARY,
        gesture_state=GestureState.POINT,
        canvas_x=500.0,
        canvas_y=400.0,
        index_tip_norm=(0.3, 0.4),
    )
    h_secondary = HandLandmarkData(
        track_id="hand_1",
        role=HandRole.SECONDARY,
        gesture_state=GestureState.POINT,
        canvas_x=800.0,
        canvas_y=200.0,
        index_tip_norm=(0.7, 0.2),  # norm_y = 0.2 -> high hand raises radius
    )

    win._on_multi_gesture_event([h_primary, h_secondary])
    spot_item = win._scene._spotlight_item
    assert spot_item is not None
    # radius = 50.0 + (1.0 - 0.2) * 260.0 = 258.0 px
    assert spot_item.radius() > 200.0

    # 3. Test Bimanual Loupe magnification modulation
    win._on_toggle_loupe()
    assert win._teaching_mode == "LOUPE"

    h_sec_low = HandLandmarkData(
        track_id="hand_1",
        role=HandRole.SECONDARY,
        gesture_state=GestureState.POINT,
        canvas_x=800.0,
        canvas_y=900.0,
        index_tip_norm=(0.7, 0.9),  # norm_y = 0.9 -> low hand lowers zoom
    )
    win._on_multi_gesture_event([h_primary, h_sec_low])
    loupe_item = win._scene._loupe_item
    assert loupe_item is not None
    # zoom = 1.5 + (1.0 - 0.9) * 2.5 = 1.75x
    assert loupe_item.zoom_factor() < 2.0

    # Toggle guide off
    vis_off = win._on_toggle_gesture_guide()
    assert vis_off is False
    assert win._gesture_guide_overlay.isVisible() is False

    win.close()
