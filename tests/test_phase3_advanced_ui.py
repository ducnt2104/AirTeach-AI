"""
AirTeach AI - Phase 3 Advanced UI Tests
=========================================
Unit & integration tests for:
  - BackgroundTemplateGenerator (White, Oli grid, Lined, Oxy)
  - NewLessonDialog
  - LibraryAndPrepWidget (Workspace scanning & template creation)
  - LessonBoardCanvas (3 Z-layers, Undo/Redo, Eraser, Selection, Snapshots)
  - ReplayController (Playback engine, variable speed, scrubbing)
  - SidePanelsWidget (AI Assistant tab & Camera Telemetry tab)
  - LoginWindow (Webcam preview, PIN keypad, demo mode)
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from src.core.hand_gesture_engine import GestureState
from src.core.stroke_beautifier import BezierSegment, ShapeType, SnappedShape
from src.services.document_manager import DocumentManager
from src.services.face_profile_service import TeacherProfileService
from src.services.gemini_assistant import GeminiAssistant
from src.ui.library_and_prep_widget import (
    BackgroundTemplateGenerator,
    LibraryAndPrepWidget,
    NewLessonDialog,
)
from src.ui.lesson_board_canvas import InkStroke, LessonBoardCanvas, BOARD_HEIGHT, BOARD_WIDTH
from src.ui.login_window import LoginWindow, TeacherEnrollDialog
from src.ui.replay_controller import ReplayController
from src.ui.side_panels import AiAssistantTab, CameraTelemetryTab, SidePanelsWidget
from src.utils.telemetry_logger import TelemetryLogger


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BackgroundTemplateGenerator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackgroundTemplates:
    def test_plain_white(self, qapp):
        img = BackgroundTemplateGenerator.render_plain_white(640, 360)
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8
        assert np.all(img == 255)

    def test_oli_grid(self, qapp):
        img = BackgroundTemplateGenerator.render_oli_grid(640, 360)
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8
        # Should not be plain white since grid lines were drawn
        assert not np.all(img == 255)

    def test_lined_paper(self, qapp):
        img = BackgroundTemplateGenerator.render_lined_paper(640, 360)
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8
        assert not np.all(img == 255)

    def test_oxy_coordinates(self, qapp):
        img = BackgroundTemplateGenerator.render_oxy_coordinates(640, 360)
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8
        # Axis lines exist
        assert not np.all(img == 255)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NewLessonDialog Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewLessonDialog:
    def test_dialog_defaults(self, qapp):
        dlg = NewLessonDialog()
        assert dlg.page_count >= 1
        assert dlg.template_name in (
            NewLessonDialog.TEMPLATE_PLAIN,
            NewLessonDialog.TEMPLATE_OLI,
            NewLessonDialog.TEMPLATE_LINED,
            NewLessonDialog.TEMPLATE_OXY,
        )

    def test_set_and_get_title_subject(self, qapp):
        dlg = NewLessonDialog()
        dlg._title_edit.setText("Bài 1")
        dlg._subject_edit.setText("Toán")
        assert dlg.lesson_title == "Bài 1"
        assert dlg.subject == "Toán"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LibraryAndPrepWidget Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLibraryAndPrepWidget:
    @pytest.fixture
    def workspace(self, tmp_path):
        ws = tmp_path / "workspace_test"
        ws.mkdir()
        return ws

    def test_instantiation_empty_state(self, qapp, workspace, tmp_path):
        dm = DocumentManager(tmp_path / "work")
        widget = LibraryAndPrepWidget(workspace, dm, "Thầy Nam")
        assert widget.workspace_dir == workspace
        assert not widget._empty_label.isHidden()

    def test_refresh_with_airteach_file(self, qapp, workspace, tmp_path):
        dm = DocumentManager(tmp_path / "work")
        # Create a fake airteach package in workspace
        from src.services.document_manager import AirTeachManifest
        from datetime import datetime, timezone
        pkg_path = workspace / "lesson_sample.airteach"
        manifest = AirTeachManifest("Mẫu 1", 1, datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat())

        page_png = tmp_path / "p1.png"
        import cv2
        cv2.imwrite(str(page_png), np.full((100, 100, 3), 255, dtype=np.uint8))

        dm.save_lesson(pkg_path, manifest, [page_png], [], [])

        widget = LibraryAndPrepWidget(workspace, dm, "Thầy Nam")
        widget.refresh_lesson_list()
        assert widget._table.rowCount() == 1
        assert not widget._empty_label.isVisible()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LessonBoardCanvas Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLessonBoardCanvas:
    @pytest.fixture
    def canvas(self, qapp):
        c = LessonBoardCanvas()
        return c

    def test_scene_dimensions(self, canvas):
        assert canvas.sceneRect().width() == BOARD_WIDTH
        assert canvas.sceneRect().height() == BOARD_HEIGHT

    def test_background_layer_lifecycle(self, canvas):
        img = np.full((360, 640, 3), 220, dtype=np.uint8)
        canvas.set_background_page(img)
        assert canvas._bg_item is not None
        assert canvas._bg_item.zValue() == canvas.Z_BACKGROUND

        canvas.clear_background()
        assert canvas._bg_item is None

    def test_stroke_drawing_and_undo_redo(self, canvas):
        # Draw a stroke
        canvas.start_stroke(100.0, 100.0, 0.0)
        canvas.update_stroke(120.0, 110.0, 16.0)
        canvas.update_stroke(140.0, 130.0, 32.0)
        stroke = canvas.finish_stroke()

        assert stroke is not None
        assert stroke.stroke_id in canvas._strokes
        assert len(canvas._undo_stack) == 1

        # Undo
        ok = canvas.undo()
        assert ok is True
        assert stroke.stroke_id not in canvas._strokes
        assert len(canvas._redo_stack) == 1

        # Redo
        ok2 = canvas.redo()
        assert ok2 is True
        assert stroke.stroke_id in canvas._strokes

    def test_eraser_hit_test(self, canvas):
        canvas.start_stroke(200.0, 200.0, 0.0)
        canvas.update_stroke(210.0, 210.0, 16.0)
        stroke = canvas.finish_stroke()
        assert stroke is not None

        # Erase near (205, 205)
        erased = canvas.erase_at(205.0, 205.0, radius=30.0)
        assert erased is True
        assert stroke.stroke_id not in canvas._strokes

    def test_clear_all_ink_preserves_background(self, canvas):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        canvas.set_background_page(img)

        canvas.start_stroke(50.0, 50.0, 0.0)
        canvas.update_stroke(60.0, 60.0, 16.0)
        canvas.finish_stroke()
        assert len(canvas._strokes) == 1

        canvas.clear_all_ink()
        assert len(canvas._strokes) == 0
        assert canvas._bg_item is not None  # Background layer untouched

    def test_laser_cursor_glow(self, canvas):
        canvas.set_laser_position(300.0, 400.0, visible=True)
        assert canvas._laser_item.isVisible()
        assert canvas._laser_glow.isVisible()
        assert canvas._laser_item.pos().x() == pytest.approx(300.0)
        assert canvas._laser_item.pos().y() == pytest.approx(400.0)

        canvas.hide_cursors()
        assert not canvas._laser_item.isVisible()
        assert not canvas._laser_glow.isVisible()

    def test_selection_region(self, canvas):
        canvas.update_selection(100.0, 100.0, 300.0, 250.0)
        rect = canvas.get_selected_rect()
        assert rect is not None
        assert rect.width() == pytest.approx(200.0)
        assert rect.height() == pytest.approx(150.0)

        canvas.hide_selection()
        assert canvas.get_selected_rect() is None

    def test_snapshot_pixmap(self, canvas):
        pix = canvas.snapshot_as_pixmap()
        assert not pix.isNull()
        assert pix.width() == BOARD_WIDTH
        assert pix.height() == BOARD_HEIGHT

    def test_export_strokes_format(self, canvas):
        canvas.start_stroke(10.0, 10.0, 0.0)
        canvas.update_stroke(20.0, 20.0, 16.0)
        canvas.finish_stroke()

        exported = canvas.get_strokes_for_export()
        assert len(exported) == 1
        assert "stroke_id" in exported[0]
        assert "segments" in exported[0]
        assert "widths" in exported[0]
        assert "color_hex" in exported[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ReplayController Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayController:
    def _create_mock_stroke(self, start_t: float, end_t: float) -> InkStroke:
        seg = BezierSegment(
            p0=np.array([10.0, 10.0]),
            p1=np.array([20.0, 15.0]),
            p2=np.array([30.0, 25.0]),
            p3=np.array([40.0, 40.0]),
        )
        return InkStroke(
            stroke_id=str(uuid.uuid4()),
            segments=[seg],
            widths=[2.5],
            timestamps_ms=[start_t, (start_t + end_t) / 2.0, end_t],
        )

    def test_load_strokes_sets_duration(self, qapp):
        ctrl = ReplayController()
        s1 = self._create_mock_stroke(1000.0, 2000.0)
        s2 = self._create_mock_stroke(2500.0, 4000.0)
        ctrl.load_strokes([s1, s2])

        assert ctrl._total_duration_ms == pytest.approx(3000.0)  # 4000 - 1000
        assert ctrl._start_time_ms == pytest.approx(1000.0)

    def test_play_pause_toggle(self, qapp):
        ctrl = ReplayController()
        s = self._create_mock_stroke(0.0, 1000.0)
        ctrl.load_strokes([s])

        assert not ctrl._is_playing
        ctrl.play()
        assert ctrl._is_playing
        ctrl.pause()
        assert not ctrl._is_playing

    def test_speed_multipliers(self, qapp):
        ctrl = ReplayController()
        ctrl._combo_speed.setCurrentIndex(0)
        assert ctrl._speed_multiplier == 1.0
        ctrl._combo_speed.setCurrentIndex(1)
        assert ctrl._speed_multiplier == 2.0
        ctrl._combo_speed.setCurrentIndex(2)
        assert ctrl._speed_multiplier == 4.0

    def test_timer_tick_advances_time(self, qapp):
        ctrl = ReplayController()
        s = self._create_mock_stroke(0.0, 1000.0)
        ctrl.load_strokes([s])
        ctrl._current_time_ms = 100.0
        ctrl._on_timer_tick()
        assert ctrl._current_time_ms > 100.0

    def test_reset_playback(self, qapp):
        ctrl = ReplayController()
        s = self._create_mock_stroke(0.0, 1000.0)
        ctrl.load_strokes([s])
        ctrl._current_time_ms = 500.0
        ctrl.reset_playback()
        assert ctrl._current_time_ms == 0.0
        assert not ctrl._is_playing


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SidePanelsWidget Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSidePanelsWidget:
    @pytest.fixture
    def panels(self, qapp, tmp_path):
        assistant = MagicMock(spec=GeminiAssistant)
        assistant.is_online = True
        assistant.api_key = "test_key"
        tel = TelemetryLogger(tmp_path / "telemetry.csv")
        widget = SidePanelsWidget(assistant, tel)
        return widget

    def test_tabs_count(self, panels):
        assert panels.count() == 2

    def test_ai_tab_actions(self, panels):
        tab = panels.tab_ai
        tab.set_current_page_text("Page text")
        assert tab._current_page_text == "Page text"
        tab._append_message("AI", "Xin chào", is_user=False)
        assert "Xin chào" in tab._chat_view.toPlainText()

    def test_camera_telemetry_tab(self, panels):
        tab = panels.tab_cam
        tab.update_fsm_state(GestureState.DRAW)
        assert "DRAW" in tab._fsm_badge.text()

        tab.update_telemetry(fps=30.5, stroke_count=12)
        assert "30.5" in tab._fps_lbl.text()
        assert "12" in tab._strokes_count_lbl.text()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LoginWindow Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoginWindow:
    @pytest.fixture
    def login_win(self, qapp):
        service = MagicMock(spec=TeacherProfileService)
        service.list_teachers.return_value = [
            {"id": "t1", "full_name": "Cô Lan", "subject": "Văn", "workspace_dir": "/tmp/t1", "created_at": "now"}
        ]
        # Avoid starting actual camera thread
        with patch.object(LoginWindow, "_start_camera"):
            win = LoginWindow(service=service, camera_index=99)
            yield win
            win.close()
            win.deleteLater()
            qapp.processEvents()

    def test_teacher_dropdown_populated(self, login_win):
        assert login_win._teacher_combo.count() == 1
        assert "Cô Lan" in login_win._teacher_combo.currentText()

    def test_pin_keypad_input(self, login_win):
        login_win._on_keypad_press("1")
        login_win._on_keypad_press("2")
        login_win._on_keypad_press("3")
        assert login_win._pin_buffer == "123"

        login_win._on_keypad_press("⌫")
        assert login_win._pin_buffer == "12"

        login_win._on_keypad_press("Xóa")
        assert login_win._pin_buffer == ""

    def test_demo_mode_signal_emitted(self, login_win):
        emitted = []
        login_win.demo_mode_requested.connect(lambda: emitted.append(True))
        login_win._on_demo_mode()
        assert len(emitted) == 1
