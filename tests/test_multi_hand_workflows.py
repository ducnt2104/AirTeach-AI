"""
AirTeach AI - Multi-Hand Workflows & Workstation Integration Tests
==================================================================
Tests:
  - Concurrent dual-hand drawing on LessonBoard
  - Asymmetric draw + erase interactions
  - In-air Floating Action Menu invocation via Air Tap / Pinch Click
  - Settings dialog multi-hand configuration persistence
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.core.hand_gesture_engine import GestureState
from src.core.multi_hand_coordinator import (
    HandLandmarkData,
    HandRole,
    MultiHandMode,
    AirClickType,
)
from src.ui.floating_action_menu import MenuLayoutMode
from src.ui.main_window import MainWindow
from src.ui.settings_dialog import SettingsDialog
from src.services.credential_service import TeacherCredentialService


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_multi_hand_initialization(qapp, tmp_path):
    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_floating_menu")
        assert win._floating_menu is not None
        assert win._floating_menu.scene() is win._scene
        assert win._floating_menu.zValue() == 50
        assert not win._floating_menu.isVisible()

        assert hasattr(win, "_multi_hand_mode")
        assert win._multi_hand_mode in ("draw_and_tool", "dual_draw", "draw_and_erase", "independent")
        assert hasattr(win, "_hand_strokes")
        assert isinstance(win._hand_strokes, dict)
    finally:
        win.close()


def test_concurrent_dual_hand_drawing(qapp):
    win = MainWindow(camera_index=-1)
    try:
        win._multi_hand_mode = "dual_draw"
        win._scene.clear_all_strokes()

        # Frame 1: Hand 0 and Hand 1 start drawing at different locations
        h0 = HandLandmarkData(
            track_id="hand_0",
            handedness="Right",
            canvas_x=384.0,
            canvas_y=216.0,
            gesture_state=GestureState.DRAW,
            role=HandRole.PRIMARY,
        )
        h1 = HandLandmarkData(
            track_id="hand_1",
            handedness="Left",
            canvas_x=1152.0,
            canvas_y=648.0,
            gesture_state=GestureState.DRAW,
            role=HandRole.SECONDARY,
        )

        win._on_multi_gesture_event([h0, h1])
        assert "hand_0" in win._hand_strokes
        assert "hand_1" in win._hand_strokes
        assert len(win._scene._live_previews) == 2

        # Frame 2: Both hands extend stroke
        h0_ext = HandLandmarkData(
            track_id="hand_0",
            handedness="Right",
            canvas_x=480.0,
            canvas_y=270.0,
            gesture_state=GestureState.DRAW,
            role=HandRole.PRIMARY,
        )
        h1_ext = HandLandmarkData(
            track_id="hand_1",
            handedness="Left",
            canvas_x=1248.0,
            canvas_y=702.0,
            gesture_state=GestureState.DRAW,
            role=HandRole.SECONDARY,
        )
        win._on_multi_gesture_event([h0_ext, h1_ext])
        assert len(win._hand_strokes["hand_0"].raw_points) == 2
        assert len(win._hand_strokes["hand_1"].raw_points) == 2

        # Frame 3: Both hands lift pen (PAUSE)
        h0_end = HandLandmarkData(
            track_id="hand_0",
            handedness="Right",
            canvas_x=480.0,
            canvas_y=270.0,
            gesture_state=GestureState.PAUSE,
            role=HandRole.PRIMARY,
        )
        h1_end = HandLandmarkData(
            track_id="hand_1",
            handedness="Left",
            canvas_x=1248.0,
            canvas_y=702.0,
            gesture_state=GestureState.PAUSE,
            role=HandRole.SECONDARY,
        )
        win._on_multi_gesture_event([h0_end, h1_end])

        # Both strokes must be committed to scene
        assert len(win._scene._strokes) == 2
        stroke_list = list(win._scene._strokes.values())
        hand_ids = {s.hand_id for s in stroke_list}
        assert "hand_0" in hand_ids
        assert "hand_1" in hand_ids
    finally:
        win.close()


def test_asymmetric_draw_and_erase(qapp):
    win = MainWindow(camera_index=-1)
    try:
        win._multi_hand_mode = "draw_and_erase"
        win._scene.clear_all_strokes()

        # Step 1: Draw a stroke with primary hand from (100, 100) to (300, 100)
        h_pri_start = HandLandmarkData(
            track_id="hand_pri",
            handedness="Right",
            canvas_x=100.0,
            canvas_y=100.0,
            gesture_state=GestureState.DRAW,
            role=HandRole.PRIMARY,
        )
        win._on_multi_gesture_event([h_pri_start])

        h_pri_end = HandLandmarkData(
            track_id="hand_pri",
            handedness="Right",
            canvas_x=300.0,
            canvas_y=100.0,
            gesture_state=GestureState.DRAW,
            role=HandRole.PRIMARY,
        )
        win._on_multi_gesture_event([h_pri_end])

        h_pri_pause = HandLandmarkData(
            track_id="hand_pri",
            handedness="Right",
            canvas_x=300.0,
            canvas_y=100.0,
            gesture_state=GestureState.PAUSE,
            role=HandRole.PRIMARY,
        )
        win._on_multi_gesture_event([h_pri_pause])
        assert len(win._scene._strokes) == 1

        # Step 2: Secondary hand acts as eraser at (200, 100)
        h_sec_erase = HandLandmarkData(
            track_id="hand_sec",
            handedness="Left",
            canvas_x=200.0,
            canvas_y=100.0,
            gesture_state=GestureState.DRAW,
            role=HandRole.SECONDARY,
            is_pinched=True,
        )
        win._on_multi_gesture_event([h_sec_erase])

        # Primary stroke must be erased by secondary hand
        assert len(win._scene._strokes) == 0
    finally:
        win.close()


def test_in_air_floating_menu_invocation_and_selection(qapp):
    win = MainWindow(camera_index=-1)
    try:
        win._floating_menu.close_menu()
        assert not win._floating_menu.isVisible()

        # Trigger menu open via in-air pinch/tap on empty space
        h_menu = HandLandmarkData(
            track_id="hand_sec",
            handedness="Left",
            canvas_x=960.0,
            canvas_y=540.0,
            gesture_state=GestureState.POINT,
            role=HandRole.SECONDARY,
            is_air_tap=True,
        )
        win._on_multi_gesture_event([h_menu])

        assert win._floating_menu.isVisible()
        assert win._floating_menu.pos().x() == 960.0
        assert win._floating_menu.pos().y() == 540.0

        # Toggle via shortcut / method
        win._toggle_floating_menu()
        assert not win._floating_menu.isVisible()
    finally:
        win.close()


def test_settings_dialog_multi_hand_persistence(qapp, tmp_path):
    db_file = tmp_path / "test_profiles.db"
    cred_svc = TeacherCredentialService(db_file)

    dlg = SettingsDialog(
        teacher_id="teacher_math_01",
        teacher_name="Thầy Nguyễn Toán",
        credential_service=cred_svc,
    )
    try:
        # Change settings
        idx_mode = dlg.multi_hand_mode_combo.findData("dual_draw")
        dlg.multi_hand_mode_combo.setCurrentIndex(idx_mode)

        idx_hand = dlg.dominant_hand_combo.findData("Left")
        dlg.dominant_hand_combo.setCurrentIndex(idx_hand)

        idx_click = dlg.air_click_combo.findData("hybrid")
        dlg.air_click_combo.setCurrentIndex(idx_click)

        dlg.eraser_radius_spin.setValue(48)

        dlg._on_save_all()

        # Verify saved settings in database
        saved = cred_svc.get_settings("teacher_math_01")
        assert saved.multi_hand_mode == "dual_draw"
        assert saved.dominant_hand == "Left"
        assert saved.air_click_type == "hybrid"
        assert saved.eraser_radius == 48.0
    finally:
        dlg.close()


def test_secondary_hand_page_swipe_navigation(qapp):
    """
    Verify either hand (including secondary assistant hand) can navigate pages
    via SWIPE_LEFT and SWIPE_RIGHT air gestures.
    """
    win = MainWindow(camera_index=-1)
    win.show()

    try:
        # Set up 3 dummy pages
        win._pages = [
            MagicMock(image_path="page0.png", text=""),
            MagicMock(image_path="page1.png", text=""),
            MagicMock(image_path="page2.png", text=""),
        ]
        win._current_page_index = 0

        # Secondary hand performs SWIPE_RIGHT -> advances to page 1
        sec_h1 = HandLandmarkData(
            track_id="hand_1",
            handedness="Left",
            gesture_state=GestureState.SWIPE_RIGHT,
            role=HandRole.SECONDARY,
            canvas_x=200.0,
            canvas_y=400.0,
        )
        win._on_multi_gesture_event([sec_h1])
        assert win._current_page_index == 1

        # Secondary hand performs SWIPE_RIGHT again -> advances to page 2
        sec_h2 = HandLandmarkData(
            track_id="hand_1",
            handedness="Left",
            gesture_state=GestureState.SWIPE_RIGHT,
            role=HandRole.SECONDARY,
            canvas_x=220.0,
            canvas_y=400.0,
        )
        win._on_multi_gesture_event([sec_h2])
        assert win._current_page_index == 2

        # Secondary hand performs SWIPE_LEFT -> returns to page 1
        sec_h3 = HandLandmarkData(
            track_id="hand_1",
            handedness="Left",
            gesture_state=GestureState.SWIPE_LEFT,
            role=HandRole.SECONDARY,
            canvas_x=200.0,
            canvas_y=400.0,
        )
        win._on_multi_gesture_event([sec_h3])
        assert win._current_page_index == 1

        # Primary hand performs SWIPE_LEFT -> returns to page 0
        prim_h = HandLandmarkData(
            track_id="hand_0",
            handedness="Right",
            gesture_state=GestureState.SWIPE_LEFT,
            role=HandRole.PRIMARY,
            canvas_x=600.0,
            canvas_y=400.0,
        )
        win._on_multi_gesture_event([prim_h])
        assert win._current_page_index == 0
    finally:
        win.close()

