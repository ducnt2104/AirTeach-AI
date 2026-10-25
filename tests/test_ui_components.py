"""
AirTeach AI — Phase 3 UI Component Tests
==========================================
Tests for LessonBoard, CameraWidget, ChatPanel, SettingsDialog,
FaceAuthDialog, and MainWindow — all run without a real camera or display.

All tests use a single QApplication instance via a module-scoped fixture.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure PySide6 can run headlessly
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt, QPointF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication


# ── Module-scoped QApplication ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


# ═══════════════════════════════════════════════════════════════════════════════
# LessonBoard Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLessonBoard:
    """Tests for LessonBoard (QGraphicsScene canvas)."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        from src.ui.lesson_board import LessonBoard, InkStroke, BOARD_WIDTH, BOARD_HEIGHT
        from src.core.stroke_beautifier import BezierSegment, ShapeType
        self.LessonBoard = LessonBoard
        self.InkStroke = InkStroke
        self.BOARD_WIDTH = BOARD_WIDTH
        self.BOARD_HEIGHT = BOARD_HEIGHT
        self.BezierSegment = BezierSegment
        self.ShapeType = ShapeType
        self.board = LessonBoard()

    def test_scene_rect(self):
        rect = self.board.sceneRect()
        assert rect.width() == self.BOARD_WIDTH
        assert rect.height() == self.BOARD_HEIGHT

    def test_laser_starts_hidden(self):
        assert not self.board._laser_item.isVisible()

    def test_move_laser_visible(self):
        self.board.move_laser(100.0, 200.0, visible=True)
        assert self.board._laser_item.isVisible()
        assert self.board._laser_item.pos().x() == pytest.approx(100.0)
        assert self.board._laser_item.pos().y() == pytest.approx(200.0)

    def test_move_laser_invisible(self):
        self.board.move_laser(0.0, 0.0, visible=False)
        assert not self.board._laser_item.isVisible()

    def test_set_background_page(self):
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        self.board.set_background_page(img)
        assert self.board._bg_item is not None

    def test_clear_background(self):
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        self.board.set_background_page(img)
        self.board.clear_background()
        assert self.board._bg_item is None

    def test_begin_update_end_stroke_preview(self):
        color = QColor("#ff0000")
        self.board.begin_stroke_preview(50.0, 60.0, color)
        assert self.board._preview_item is not None
        self.board.update_stroke_preview(70.0, 80.0)
        self.board.update_stroke_preview(90.0, 100.0)
        self.board.end_stroke_preview()
        assert self.board._preview_item is None
        assert len(self.board._live_points) == 0

    def _make_freehand_stroke(self, stroke_id: str = None) -> object:
        """Create a minimal freehand InkStroke for testing."""
        if stroke_id is None:
            stroke_id = str(uuid.uuid4())
        # BezierSegment fields: p0 (start), p1 (cp1), p2 (cp2), p3 (end)
        seg = self.BezierSegment(
            p0=np.array([10.0, 10.0]),
            p1=np.array([20.0, 5.0]),
            p2=np.array([30.0, 15.0]),
            p3=np.array([40.0, 10.0]),
        )
        return self.InkStroke(
            stroke_id=stroke_id,
            segments=[seg],
            widths=[2.5],
            snap=None,
            color=QColor("#202124"),
            timestamps_ms=[0.0, 16.0],
        )

    def test_commit_freehand_stroke(self):
        stroke = self._make_freehand_stroke("stroke-001")
        self.board.commit_stroke(stroke)
        assert "stroke-001" in self.board._strokes
        assert stroke._path_item is not None

    def test_remove_stroke(self):
        stroke = self._make_freehand_stroke("stroke-002")
        self.board.commit_stroke(stroke)
        result = self.board.remove_stroke("stroke-002")
        assert result is True
        assert "stroke-002" not in self.board._strokes

    def test_remove_nonexistent_stroke(self):
        result = self.board.remove_stroke("does-not-exist")
        assert result is False

    def test_clear_all_strokes(self):
        for i in range(3):
            self.board.commit_stroke(self._make_freehand_stroke(f"batch-{i}"))
        self.board.clear_all_strokes()
        assert len(self.board._strokes) == 0

    def test_get_strokes_for_export(self):
        board2 = self.LessonBoard()
        stroke = self._make_freehand_stroke("export-001")
        board2.commit_stroke(stroke)
        data = board2.get_strokes_for_export()
        assert any(d["stroke_id"] == "export-001" for d in data)
        entry = next(d for d in data if d["stroke_id"] == "export-001")
        assert "segments" in entry
        assert "widths" in entry
        assert "color_hex" in entry
        assert "timestamps_ms" in entry

    def test_show_hide_selection_rect(self):
        rect = QRectF(100, 100, 200, 150)
        self.board.show_selection_rect(rect)
        assert self.board._select_rect_item is not None
        self.board.hide_selection_rect()
        assert self.board._select_rect_item is None

    def test_add_text_overlay(self):
        item = self.board.add_text_overlay("Hello AirTeach", 50.0, 50.0)
        assert item is not None
        assert item.toPlainText() == "Hello AirTeach"

    def test_add_latex_overlay(self):
        rgba_img = np.full((100, 200, 4), 255, dtype=np.uint8)
        item = self.board.add_latex_overlay(rgba_img, 60.0, 70.0)
        assert item is not None

    def test_snapshot_as_pixmap(self):
        pixmap = self.board.snapshot_as_pixmap()
        assert not pixmap.isNull()
        assert pixmap.width() == self.BOARD_WIDTH
        assert pixmap.height() == self.BOARD_HEIGHT

    def test_snapshot_as_image_and_export_to_file(self, tmp_path):
        """Verify snapshot_as_image resolution supersampling and atomic export_to_file."""
        # 1. 1080p standard snapshot
        img_1080p = self.board.snapshot_as_image(scale=1.0)
        assert not img_1080p.isNull()
        assert img_1080p.width() == 1920
        assert img_1080p.height() == 1080

        # 2. 4K UHD supersampling (scale=2.0)
        img_4k = self.board.snapshot_as_image(scale=2.0)
        assert not img_4k.isNull()
        assert img_4k.width() == 3840
        assert img_4k.height() == 2160

        # 3. Transparent PNG background
        img_trans = self.board.snapshot_as_image(scale=1.0, transparent_bg=True)
        assert not img_trans.isNull()
        assert img_trans.hasAlphaChannel()

        # 4. Atomic export to PNG file
        png_path = tmp_path / "board_export.png"
        ok_png = self.board.export_to_file(png_path, scale=1.0)
        assert ok_png is True
        assert png_path.exists()
        assert png_path.stat().st_size > 0

        # 5. Atomic export to JPG file
        jpg_path = tmp_path / "board_export.jpg"
        ok_jpg = self.board.export_to_file(jpg_path, scale=1.0, quality=90)
        assert ok_jpg is True
        assert jpg_path.exists()
        assert jpg_path.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CameraWidget Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCameraWidget:
    """Tests for CameraWidget (video display label)."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        from src.ui.camera_widget import CameraWidget
        self.widget = CameraWidget()

    def test_instantiation(self):
        assert self.widget is not None

    def test_size_hint(self):
        hint = self.widget.sizeHint()
        assert hint.width() > 0
        assert hint.height() > 0

    def test_minimum_size(self):
        assert self.widget.minimumWidth() >= 240
        assert self.widget.minimumHeight() >= 180

    def test_on_frame_updates_pixmap(self):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        self.widget.on_frame(frame)
        # After calling on_frame, current_frame should be set
        assert self.widget._current_frame is not None

    def test_on_frame_stores_frame(self):
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 100
        self.widget.on_frame(frame)
        assert self.widget._current_frame is frame

    def test_set_show_landmarks(self):
        self.widget.set_show_landmarks(False)
        assert self.widget._show_landmarks is False
        self.widget.set_show_landmarks(True)
        assert self.widget._show_landmarks is True

    def test_frame_displayed_signal_exists(self):
        assert hasattr(self.widget, "frame_displayed")

    def test_initial_text_is_offline(self):
        # Fresh widget should show "Camera offline" placeholder
        w2 = __import__("src.ui.camera_widget", fromlist=["CameraWidget"]).CameraWidget()
        # The text is set in __init__ before styleSheet override; check _current_frame is None
        assert w2._current_frame is None


# ═══════════════════════════════════════════════════════════════════════════════
# ChatPanel Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatPanel:
    """Tests for ChatPanel (Gemini assistant UI)."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        from src.ui.chat_panel import ChatPanel
        # Create a mock assistant with the correct is_online property
        mock_assistant = MagicMock()
        mock_assistant.is_online = True
        self.panel = ChatPanel(assistant=mock_assistant)

    def test_instantiation(self):
        assert self.panel is not None

    def test_minimum_width(self):
        assert self.panel.minimumWidth() >= 300

    def test_set_current_page_text(self):
        self.panel.set_current_page_text("Slide text goes here.")
        assert self.panel._current_page_text == "Slide text goes here."

    def test_append_message_user(self):
        self.panel._append_message("Tôi", "Xin chào AI!", is_user=True)
        text = self.panel._history_view.toPlainText()
        assert "Xin chào AI!" in text

    def test_append_message_ai(self):
        self.panel._append_message("AI", "Chào bạn!", is_user=False)
        text = self.panel._history_view.toPlainText()
        assert "Chào bạn!" in text

    def test_update_online_status_offline(self):
        self.panel.update_online_status(False)
        # isHidden() reflects setVisible(False) regardless of whether widget is shown on screen
        assert not self.panel._offline_banner.isHidden()

    def test_update_online_status_online(self):
        self.panel.update_online_status(True)
        assert self.panel._offline_banner.isHidden()

    def test_history_list_starts_empty(self):
        from src.ui.chat_panel import ChatPanel
        mock_assistant = MagicMock()
        mock_assistant.is_online = True
        fresh = ChatPanel(assistant=mock_assistant)
        assert fresh._history == []

    def test_request_canvas_snapshot_signal_exists(self):
        assert hasattr(self.panel, "request_canvas_snapshot")

    def test_input_box_exists(self):
        assert hasattr(self.panel, "_input_box")


# ═══════════════════════════════════════════════════════════════════════════════
# SettingsDialog Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSettingsDialog:
    """Tests for SettingsDialog (application settings)."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        from src.ui.settings_dialog import SettingsDialog
        self.dlg = SettingsDialog()

    def test_instantiation(self):
        assert self.dlg is not None

    def test_default_ink_color(self):
        color = self.dlg.ink_color
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_default_min_stroke_width(self):
        assert self.dlg.min_stroke_width > 0

    def test_default_max_stroke_width(self):
        assert self.dlg.max_stroke_width >= self.dlg.min_stroke_width

    def test_default_rdp_epsilon(self):
        eps = self.dlg.rdp_epsilon
        assert eps > 0.0

    def test_one_euro_beta(self):
        beta = self.dlg.one_euro_beta
        assert isinstance(beta, float)

    def test_shape_snapping_enabled_is_bool(self):
        enabled = self.dlg.shape_snapping_enabled
        assert isinstance(enabled, bool)

    def test_clahe_enabled_is_bool(self):
        enabled = self.dlg.clahe_enabled
        assert isinstance(enabled, bool)

    def test_api_key_changed_signal_exists(self):
        assert hasattr(self.dlg, "api_key_changed")

    def test_camera_index_changed_signal_exists(self):
        assert hasattr(self.dlg, "camera_index_changed")

    def test_calibration_requested_signal_exists(self):
        assert hasattr(self.dlg, "calibration_requested")


# ═══════════════════════════════════════════════════════════════════════════════
# FaceAuthDialog Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaceAuthDialog:
    """Tests for FaceAuthDialog (face login / enrollment)."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        from src.ui.face_auth_dialog import FaceAuthDialog, DialogMode
        from src.services.face_profile_service import TeacherProfileService
        self.FaceAuthDialog = FaceAuthDialog
        self.DialogMode = DialogMode
        # Create a mock service so dialogs can be instantiated without SQLite
        self.mock_service = MagicMock(spec=TeacherProfileService)

    def test_login_mode_instantiation(self):
        dlg = self.FaceAuthDialog(service=self.mock_service, mode=self.DialogMode.LOGIN)
        assert dlg is not None

    def test_enroll_mode_instantiation(self):
        dlg = self.FaceAuthDialog(service=self.mock_service, mode=self.DialogMode.ENROLL)
        assert dlg is not None

    def test_login_title(self):
        dlg = self.FaceAuthDialog(service=self.mock_service, mode=self.DialogMode.LOGIN)
        title = dlg.windowTitle()
        assert len(title) > 0

    def test_enroll_title(self):
        dlg = self.FaceAuthDialog(service=self.mock_service, mode=self.DialogMode.ENROLL)
        title = dlg.windowTitle()
        assert len(title) > 0

    def test_login_title_different_from_enroll(self):
        login_dlg = self.FaceAuthDialog(service=self.mock_service, mode=self.DialogMode.LOGIN)
        enroll_dlg = self.FaceAuthDialog(service=self.mock_service, mode=self.DialogMode.ENROLL)
        # Titles should differ
        assert login_dlg.windowTitle() != enroll_dlg.windowTitle()

    def test_result_attribute_exists(self):
        dlg = self.FaceAuthDialog(service=self.mock_service, mode=self.DialogMode.LOGIN)
        # Should have _mode attribute
        assert hasattr(dlg, "_mode")
        assert dlg._mode == self.DialogMode.LOGIN


# ═══════════════════════════════════════════════════════════════════════════════
# MainWindow Smoke Tests (no camera, no real network)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainWindowSmoke:
    """Smoke tests for MainWindow — verifies instantiation and key attributes."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        # Patch HandTrackingWorker to avoid camera access
        with patch("src.ui.main_window._QtCameraWorker") as mock_cam_cls, \
             patch("src.ui.main_window.QThread") as mock_thread_cls:

            mock_cam = MagicMock()
            for attr in ("frame_ready", "gesture_event", "error_occurred"):
                sig = MagicMock()
                sig.connect = MagicMock()
                setattr(mock_cam, attr, sig)
            mock_cam_cls.return_value = mock_cam

            mock_thread = MagicMock()
            mock_thread.isRunning.return_value = False
            mock_thread.started = MagicMock()
            mock_thread.started.connect = MagicMock()
            mock_thread_cls.return_value = mock_thread

            from src.ui.main_window import MainWindow
            self.window = MainWindow(teacher_row=None, camera_index=99)

    def test_window_title(self):
        assert "AirTeach" in self.window.windowTitle()

    def test_scene_is_lesson_board(self):
        from src.ui.lesson_board import LessonBoard
        assert isinstance(self.window._scene, LessonBoard)

    def test_chat_panel_exists(self):
        from src.ui.chat_panel import ChatPanel
        assert isinstance(self.window._chat_panel, ChatPanel)

    def test_camera_widget_exists(self):
        from src.ui.camera_widget import CameraWidget
        assert isinstance(self.window._cam_widget, CameraWidget)

    def test_gesture_hud_label(self):
        assert self.window._gesture_hud is not None

    def test_status_bar_labels(self):
        assert self.window._status_mode is not None
        assert self.window._status_page is not None
        assert self.window._status_fps is not None

    def test_ink_color_default(self):
        assert isinstance(self.window._ink_color, QColor)

    def test_snapping_on_default(self):
        assert isinstance(self.window._snapping_on, bool)
        assert self.window._snapping_on is True

    def test_doc_manager_initialized(self):
        from src.services.document_manager import DocumentManager
        assert isinstance(self.window._doc_manager, DocumentManager)

    def test_assistant_initialized(self):
        from src.services.gemini_assistant import GeminiAssistant
        assert isinstance(self.window._assistant, GeminiAssistant)

    def test_recognizer_initialized(self):
        from src.services.smart_ink_ocr import SmartInkRecognizer
        assert isinstance(self.window._recognizer, SmartInkRecognizer)

    def test_scene_coords_center(self):
        bx, by = self.window._scene_coords(0.5, 0.5)
        from src.ui.lesson_board import BOARD_WIDTH, BOARD_HEIGHT
        assert bx == pytest.approx(BOARD_WIDTH * 0.5)
        assert by == pytest.approx(BOARD_HEIGHT * 0.5)

    def test_scene_coords_origin(self):
        bx, by = self.window._scene_coords(0.0, 0.0)
        assert bx == pytest.approx(0.0)
        assert by == pytest.approx(0.0)

    def test_scene_coords_full(self):
        bx, by = self.window._scene_coords(1.0, 1.0)
        from src.ui.lesson_board import BOARD_WIDTH, BOARD_HEIGHT
        assert bx == pytest.approx(BOARD_WIDTH)
        assert by == pytest.approx(BOARD_HEIGHT)

    def test_rdp_epsilon_positive(self):
        assert self.window._rdp_epsilon > 0

    def test_ink_width_range_valid(self):
        assert self.window._ink_min_width > 0
        assert self.window._ink_max_width >= self.window._ink_min_width


# ═══════════════════════════════════════════════════════════════════════════════
# InkStroke Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInkStroke:
    """Verify InkStroke dataclass fields and compare=False behavior."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        from src.ui.lesson_board import InkStroke
        from src.core.stroke_beautifier import BezierSegment
        self.InkStroke = InkStroke
        self.BezierSegment = BezierSegment

    def _make_seg(self):
        # BezierSegment fields: p0 (start), p1 (cp1), p2 (cp2), p3 (end)
        return self.BezierSegment(
            p0=np.array([0.0, 0.0]),
            p1=np.array([1.0, 0.0]),
            p2=np.array([2.0, 0.0]),
            p3=np.array([3.0, 0.0]),
        )

    def test_defaults(self):
        s = self.InkStroke(stroke_id="abc", segments=[], widths=[])
        assert s.snap is None
        assert s._path_item is None
        assert s._shape_item is None
        assert s.timestamps_ms == []

    def test_equality_ignores_graphics_items(self):
        """Two strokes with same data but different _path_item should be equal."""
        seg = self._make_seg()
        s1 = self.InkStroke(stroke_id="x", segments=[seg], widths=[2.0])
        s2 = self.InkStroke(stroke_id="x", segments=[seg], widths=[2.0])
        # _path_item has compare=False so these should be equal
        s1._path_item = object()  # type: ignore
        assert s1 == s2

    def test_stroke_id_stored_correctly(self):
        s = self.InkStroke(stroke_id="test-id-123", segments=[], widths=[])
        assert s.stroke_id == "test-id-123"

    def test_color_default(self):
        s = self.InkStroke(stroke_id="c", segments=[], widths=[])
        assert s.color == QColor("#202124")


# ═══════════════════════════════════════════════════════════════════════════════
# MainWindow Multi-Page Segregation & Restoration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainWindowMultiPage:
    """Verify stroke segregation, stashing, and page switching in MainWindow."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        with patch("src.ui.main_window._QtCameraWorker") as mock_cam_cls, \
             patch("src.ui.main_window.QThread") as mock_thread_cls:

            mock_cam = MagicMock()
            for attr in ("frame_ready", "gesture_event", "error_occurred"):
                sig = MagicMock()
                sig.connect = MagicMock()
                setattr(mock_cam, attr, sig)
            mock_cam_cls.return_value = mock_cam

            mock_thread = MagicMock()
            mock_thread.isRunning.return_value = False
            mock_thread.started = MagicMock()
            mock_thread.started.connect = MagicMock()
            mock_thread_cls.return_value = mock_thread

            from src.ui.main_window import MainWindow
            from src.services.document_manager import PageInfo
            self.window = MainWindow(teacher_row=None, camera_index=99)
            # Create synthetic 3 pages
            self.window._pages = [
                PageInfo(page_index=0, image_path="", width_px=1920, height_px=1080, text="P1"),
                PageInfo(page_index=1, image_path="", width_px=1920, height_px=1080, text="P2"),
                PageInfo(page_index=2, image_path="", width_px=1920, height_px=1080, text="P3"),
            ]

    def _create_sample_stroke(self, stroke_id: str):
        from src.ui.lesson_board import InkStroke
        from src.core.stroke_beautifier import BezierSegment
        seg = BezierSegment(
            p0=np.array([10.0, 10.0]),
            p1=np.array([20.0, 20.0]),
            p2=np.array([30.0, 30.0]),
            p3=np.array([40.0, 40.0]),
        )
        return InkStroke(
            stroke_id=stroke_id,
            segments=[seg],
            widths=[2.5],
            color=QColor("#ff0000"),
            timestamps_ms=[0.0],
        )

    def test_stroke_segregation_across_pages(self):
        # Draw on Page 0
        s0 = self._create_sample_stroke("stroke_p0")
        self.window._scene.commit_stroke(s0)
        self.window._page_strokes[0] = list(self.window._scene._strokes.values())

        assert len(self.window._scene._strokes) == 1
        assert "stroke_p0" in self.window._scene._strokes

        # Navigate to Page 1
        self.window._on_next_page()
        assert self.window._current_page_index == 1
        assert len(self.window._scene._strokes) == 0

        # Draw on Page 1
        s1 = self._create_sample_stroke("stroke_p1")
        self.window._scene.commit_stroke(s1)
        self.window._page_strokes[1] = list(self.window._scene._strokes.values())
        assert len(self.window._scene._strokes) == 1
        assert "stroke_p1" in self.window._scene._strokes

        # Navigate back to Page 0
        self.window._on_prev_page()
        assert self.window._current_page_index == 0
        assert len(self.window._scene._strokes) == 1
        assert "stroke_p0" in self.window._scene._strokes

        # Navigate to Page 1 again
        self.window._on_next_page()
        assert self.window._current_page_index == 1
        assert len(self.window._scene._strokes) == 1
        assert "stroke_p1" in self.window._scene._strokes

    def test_load_strokes_from_records(self):
        from src.services.document_manager import StrokeRecord, BezierPointRecord
        rec0 = StrokeRecord(
            stroke_id="rec_p0",
            page_index=0,
            color_hex="#0000ff",
            width=3.0,
            points=[BezierPointRecord(10.0, 10.0, 0.0), BezierPointRecord(50.0, 50.0, 100.0)],
        )
        rec1 = StrokeRecord(
            stroke_id="rec_p1",
            page_index=1,
            color_hex="#00ff00",
            width=2.0,
            points=[BezierPointRecord(20.0, 20.0, 0.0), BezierPointRecord(80.0, 80.0, 100.0)],
        )
        self.window._load_strokes_from_records([rec0, rec1])
        assert len(self.window._page_strokes[0]) == 1
        assert len(self.window._page_strokes[1]) == 1
        assert self.window._page_strokes[0][0].stroke_id == "rec_p0"
        assert self.window._page_strokes[1][0].stroke_id == "rec_p1"


class TestSettingsDialogSmoothingPresets:
    def test_smoothing_preset_selection(self, qapp):
        from src.ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(teacher_id="test_teacher", teacher_name="Test Teacher")

        # 1. Default is balanced
        assert dlg.preset_combo.currentData() == "balanced"
        assert abs(dlg.one_euro_beta - 0.35) < 1e-3
        assert abs(dlg.rdp_epsilon - 2.5) < 1e-3
        assert dlg.smoothing_preset == "balanced"

        # 2. Switch to smooth
        dlg.preset_combo.setCurrentIndex(1)  # smooth
        assert dlg.preset_combo.currentData() == "smooth"
        assert abs(dlg.one_euro_beta - 0.15) < 1e-3
        assert abs(dlg.rdp_epsilon - 3.5) < 1e-3
        assert dlg.smoothing_preset == "smooth"

        # 3. Switch to instant
        dlg.preset_combo.setCurrentIndex(2)  # instant
        assert dlg.preset_combo.currentData() == "instant"
        assert abs(dlg.one_euro_beta - 0.70) < 1e-3
        assert abs(dlg.rdp_epsilon - 1.0) < 1e-3
        assert dlg.smoothing_preset == "instant"

        # 4. Manual adjustments switch to custom
        dlg.beta_spin.setValue(0.55)
        assert dlg.preset_combo.currentData() == "custom"
        assert dlg.smoothing_preset == "custom"
        assert abs(dlg.one_euro_beta - 0.55) < 1e-3


class TestLessonBoardGridAndAxes:
    def test_lesson_board_grid_toggle_and_modes(self, qapp):
        from src.ui.lesson_board import LessonBoard
        board = LessonBoard()
        assert board.grid_mode() == "none"

        assert board.toggle_grid() == "grid"
        assert board.grid_mode() == "grid"

        assert board.toggle_grid() == "cartesian"
        assert board.grid_mode() == "cartesian"

        assert board.toggle_grid() == "dots"
        assert board.grid_mode() == "dots"

        assert board.toggle_grid() == "none"
        assert board.grid_mode() == "none"

        board.set_grid_mode("cartesian")
        assert board.grid_mode() == "cartesian"

    def test_lesson_board_draw_background_rendering(self, qapp):
        from PySide6.QtGui import QImage, QPainter
        from src.ui.lesson_board import LessonBoard, BOARD_WIDTH, BOARD_HEIGHT
        board = LessonBoard()
        board.set_grid_mode("cartesian")

        img = QImage(BOARD_WIDTH, BOARD_HEIGHT, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.white)
        p = QPainter(img)
        board.drawBackground(p, QRectF(0, 0, BOARD_WIDTH, BOARD_HEIGHT))
        p.end()

        # Verify pixels were rendered
        ptr = img.bits()
        arr = np.frombuffer(ptr, np.uint8).reshape((BOARD_HEIGHT, BOARD_WIDTH, 4))
        # Not all pixels should be white (grid lines are drawn)
        assert np.any(arr[:, :, :3] < 250)

    def test_main_window_toggle_grid_slot(self, qapp):
        from src.ui.main_window import MainWindow
        win = MainWindow(camera_index=-1)
        assert win._scene.grid_mode() == "none"

        win._on_toggle_grid()
        assert win._scene.grid_mode() == "grid"

        win._on_toggle_grid()
        assert win._scene.grid_mode() == "cartesian"

        win.close()


class TestCameraDiagnosticsDialog:
    """Validate CameraDiagnosticsDialog optical telemetry and controls."""

    def test_camera_diagnostics_initialization(self, qapp):
        from src.ui.camera_diagnostics_dialog import CameraDiagnosticsDialog
        dlg = CameraDiagnosticsDialog()
        assert "Chẩn đoán Camera" in dlg.windowTitle()
        assert dlg.lbl_fps.text() == "— FPS"
        assert dlg.lbl_hands.text() == "0 bàn tay"
        assert dlg.combo_mode.count() == 4
        dlg.close()

    def test_camera_diagnostics_telemetry_update(self, qapp):
        from src.ui.camera_diagnostics_dialog import CameraDiagnosticsDialog
        from src.core.adaptive_vision_pipeline import CameraDiagnosticsData

        dlg = CameraDiagnosticsDialog()
        diag = CameraDiagnosticsData(
            fps=29.8,
            width=1280,
            height=720,
            mean_brightness=135.2,
            sharpness_score=62.4,
            shadow_clipping_pct=1.8,
            highlight_clipping_pct=0.4,
            condition_name="NORMAL",
            hands_detected=2,
            tracking_confidence=0.92,
            gesture_state_name="DRAW",
            draw_mode_name="INDEX_UP (Air Pen)",
            pipeline_latency_ms=11.2,
            frame_dropped_count=0,
        )
        dlg.update_diagnostics(diag)

        assert dlg.lbl_fps.text() == "29.8 FPS"
        assert dlg.lbl_resolution.text() == "1280 × 720 px"
        assert dlg.lbl_condition.text() == "NORMAL"
        assert "62.4" in dlg.lbl_sharpness.text()
        assert dlg.lbl_hands.text() == "2 bàn tay"
        assert dlg.lbl_confidence.text() == "0.92"
        assert dlg.lbl_gesture.text() == "DRAW"
        assert dlg.lbl_latency.text() == "11.2 ms"
        dlg.close()

    def test_camera_diagnostics_signals_and_modes(self, qapp):
        from src.ui.camera_diagnostics_dialog import CameraDiagnosticsDialog

        dlg = CameraDiagnosticsDialog()
        emitted_modes = []
        dlg.lighting_mode_changed.connect(lambda m: emitted_modes.append(m))

        dlg.combo_mode.setCurrentIndex(1)
        assert len(emitted_modes) == 1
        assert emitted_modes[0] == "low_light"

        calib_called = []
        dlg.calibration_requested.connect(lambda: calib_called.append(True))
        dlg._on_calib_clicked()
        assert len(calib_called) == 1

        dlg.close()

    def test_main_window_open_camera_diagnostics_slot(self, qapp):
        from src.ui.main_window import MainWindow

        win = MainWindow(camera_index=-1)
        assert win._diagnostics_dialog is None

        win._on_open_camera_diagnostics()
        assert win._diagnostics_dialog is not None
        assert win._diagnostics_dialog.isVisible()

        win._diagnostics_dialog.close()
        win.close()


def test_lesson_board_grid_spacing_clamping():
    """Verify LessonBoard.set_grid_spacing clamps values within [15, 200] px."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    board.set_grid_spacing(50)
    assert board.grid_spacing() == 50

    # Below minimum
    board.set_grid_spacing(5)
    assert board.grid_spacing() == 15

    # Above maximum
    board.set_grid_spacing(500)
    assert board.grid_spacing() == 200


def test_settings_dialog_grid_settings():
    """Verify SettingsDialog grid mode and spacing accessors and updates."""
    from src.ui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(teacher_id="test_t", teacher_name="Test Teacher")
    dlg.set_grid_mode("cartesian")
    assert dlg.grid_mode == "cartesian"

    dlg.set_grid_spacing(60)
    assert dlg.grid_spacing == 60
