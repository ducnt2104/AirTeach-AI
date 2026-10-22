"""
AirTeach AI — SVG Assets, Gesture Trainer & Smart Ink Review Tests
===================================================================
Tests for:
  1. All 33 SVG vector assets (icons, gesture cards, diagrams, illustrations).
  2. GestureTrainerDialog UI, skeleton overlay, telemetry HUD, and calibration.
  3. SmartInkReviewDialog text/formula preview, Matplotlib MathText rendering, and confirmation.
  4. MainWindow teaching toolbar, mode switching, palette, width selection, and Undo/Redo.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure PySide6 can run headlessly in test runner
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from src.core.hand_gesture_engine import GestureState, LM
from src.core.stroke_beautifier import BezierSegment
from src.services.smart_ink_ocr import OcrResult
from src.ui.gesture_trainer_dialog import GestureTrainerDialog
from src.ui.lesson_board import InkStroke, LessonBoard
from src.ui.main_window import MainWindow
from src.ui.smart_ink_review_dialog import SmartInkReviewDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SVG Assets Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestSvgAssets:
    """Verify all custom SVG assets exist, parse cleanly, and render to QIcon."""

    _ROOT = Path(__file__).resolve().parent.parent
    _ASSETS_DIR = _ROOT / "assets"

    def test_all_svg_assets_exist_and_render(self, qapp):
        svg_files = list(self._ASSETS_DIR.glob("**/*.svg"))
        assert len(svg_files) >= 33, f"Expected at least 33 SVGs, found {len(svg_files)}"

        for svg_path in svg_files:
            renderer = QSvgRenderer(str(svg_path))
            assert renderer.isValid(), f"Invalid SVG file: {svg_path}"
            size = renderer.defaultSize()
            assert size.width() > 0, f"SVG width <= 0 in {svg_path}"
            assert size.height() > 0, f"SVG height <= 0 in {svg_path}"

            # Verify QIcon can load it and produce a valid pixmap
            icon = QIcon(str(svg_path))
            pixmap = icon.pixmap(32, 32)
            assert not pixmap.isNull(), f"QIcon failed to render pixmap for {svg_path}"

    def test_gesture_cards_count(self):
        gesture_dir = self._ASSETS_DIR / "gestures"
        assert gesture_dir.is_dir()
        gestures = list(gesture_dir.glob("*.svg"))
        assert len(gestures) == 7, f"Expected 7 gesture SVG cards, found {len(gestures)}"

    def test_icon_assets_count(self):
        icons_dir = self._ASSETS_DIR / "icons"
        assert icons_dir.is_dir()
        icons = list(icons_dir.glob("*.svg"))
        assert len(icons) >= 24, f"Expected at least 24 icons, found {len(icons)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Gesture Trainer & Calibration Dialog Tests
# ═══════════════════════════════════════════════════════════════════════════════

class _DummyLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestGestureTrainerDialog:
    """Verify GestureTrainerDialog interactions, live skeleton feed, HUD, and calibration."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        self.dialog = GestureTrainerDialog()

    def _make_dummy_landmarks(self) -> list[_DummyLandmark]:
        # Generate 21 dummy landmarks inside [0, 1]
        landmarks = []
        for i in range(21):
            landmarks.append(_DummyLandmark(x=0.1 + (i % 5) * 0.18, y=0.1 + (i // 5) * 0.2))
        return landmarks

    def test_initial_state(self):
        assert len(self.dialog.GESTURE_INFO) == 7
        assert len(self.dialog._cards) == 7
        assert len(self.dialog._mastered_gestures) == 0
        assert self.dialog._hud_gesture.text() == "NONE"

    def test_update_frame_with_landmarks(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = self._make_dummy_landmarks()

        # Update frame with DRAW gesture
        self.dialog.update_frame(frame, landmarks, GestureState.DRAW, 0.5, 0.5)

        assert self.dialog._hud_gesture.text() == "DRAW"
        assert GestureState.DRAW in self.dialog._mastered_gestures
        assert len(self.dialog._mastered_gestures) == 1
        assert "1 / 7" in self.dialog._progress_lbl.text()

        # Update frame with POINT gesture
        self.dialog.update_frame(frame, landmarks, GestureState.POINT, 0.4, 0.4)
        assert self.dialog._hud_gesture.text() == "POINT"
        assert GestureState.POINT in self.dialog._mastered_gestures
        assert len(self.dialog._mastered_gestures) == 2
        assert "2 / 7" in self.dialog._progress_lbl.text()

    def test_reset_calibration_signal(self):
        emitted_corners = []
        self.dialog.calibration_updated.connect(lambda c: emitted_corners.append(c))

        self.dialog._on_reset_calibration()

        assert len(emitted_corners) == 1
        corners = emitted_corners[0]
        assert isinstance(corners, np.ndarray)
        assert corners.shape == (4, 2)
        assert np.allclose(corners[0], [0.0, 0.0])
        assert np.allclose(corners[2], [1.0, 1.0])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Smart Ink Review Dialog Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmartInkReviewDialog:
    """Verify SmartInkReviewDialog text/math formula rendering, editing, and confirmation."""

    def test_handwriting_text_mode(self, qapp):
        ocr_res = OcrResult(
            text="Hàm số đồng biến trên R",
            latex="",
            is_formula=False,
            confidence=0.96,
        )
        crop_img = np.zeros((100, 200, 3), dtype=np.uint8)

        dialog = SmartInkReviewDialog(ocr_result=ocr_res, crop_image_bgr=crop_img)

        assert dialog._text_edit.text() == "Hàm số đồng biến trên R"
        assert dialog._conf_bar.value() == 96
        assert dialog._radio_replace.isChecked()

        # Test confirmation emission
        received = []
        dialog.result_confirmed.connect(lambda txt, rep: received.append((txt, rep)))

        dialog._text_edit.setText("Hàm số nghịch biến trên R")
        dialog._radio_insert.setChecked(True)
        dialog._on_accept()

        assert len(received) == 1
        assert received[0] == ("Hàm số nghịch biến trên R", False)

    def test_math_formula_mode(self, qapp):
        ocr_res = OcrResult(
            text="",
            latex=r"f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}",
            is_formula=True,
            confidence=0.91,
        )

        dialog = SmartInkReviewDialog(ocr_result=ocr_res, crop_image_bgr=None)

        assert r"\frac" in dialog._text_edit.text()
        assert dialog._conf_bar.value() == 91

        # Check preview rendering was invoked
        pixmap = dialog._preview_lbl.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()

        # Confirm replacement
        received = []
        dialog.result_confirmed.connect(lambda txt, rep: received.append((txt, rep)))
        dialog._on_accept()

        assert len(received) == 1
        assert received[0][1] is True  # replace_ink was True by default


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MainWindow Teaching Toolbar & Mode Switcher Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainWindowTeachingToolbarAndModes:
    """Verify MainWindow teaching toolbar modes, colors, stroke width, and Undo/Redo."""

    @pytest.fixture(autouse=True)
    def setup(self, qapp, tmp_path):
        with patch("src.ui.main_window.TelemetryLogger"):
            # Launch MainWindow with offline camera (camera_index=-1)
            self.window = MainWindow(teacher_row=None, camera_index=-1)

    def _create_sample_stroke(self, stroke_id: str) -> InkStroke:
        seg = BezierSegment(
            p0=np.array([100.0, 100.0]),
            p1=np.array([120.0, 120.0]),
            p2=np.array([140.0, 140.0]),
            p3=np.array([160.0, 160.0]),
        )
        return InkStroke(
            stroke_id=stroke_id,
            segments=[seg],
            widths=[2.5],
            color=QColor("#1a73e8"),
            timestamps_ms=[0.0],
        )

    def test_teaching_mode_switching(self):
        modes = ["PEN", "LASER", "ERASER", "SELECT"]
        for mode in modes:
            self.window._set_teaching_mode(mode)
            assert self.window._teaching_mode == mode
            assert mode in self.window._status_mode.text()

    def test_ink_color_selection(self):
        self.window._set_ink_color("#d93025")
        assert self.window._ink_color == QColor("#d93025")

        self.window._set_ink_color("#137333")
        assert self.window._ink_color == QColor("#137333")

    def test_stroke_width_selection(self):
        # Index 0: 1.5px
        self.window._width_combo.setCurrentIndex(0)
        assert self.window._ink_min_width == 1.5

        # Index 2: 4.5px
        self.window._width_combo.setCurrentIndex(2)
        assert self.window._ink_min_width == 4.5

        # Index 3: 8.0px
        self.window._width_combo.setCurrentIndex(3)
        assert self.window._ink_min_width == 8.0

    def test_shape_snapping_toggle(self):
        self.window._toggle_shape_snapping(False)
        assert self.window._snapping_on is False

        self.window._toggle_shape_snapping(True)
        assert self.window._snapping_on is True

    def test_undo_redo_stack(self):
        stroke = self._create_sample_stroke("test_stroke_1")
        self.window._scene.commit_stroke(stroke)
        assert len(self.window._scene._strokes) == 1
        assert len(self.window._redo_stack) == 0

        # Undo
        self.window._on_undo()
        assert len(self.window._scene._strokes) == 0
        assert len(self.window._redo_stack) == 1
        assert self.window._redo_stack[0].stroke_id == "test_stroke_1"

        # Redo
        self.window._on_redo()
        assert len(self.window._scene._strokes) == 1
        assert len(self.window._redo_stack) == 0
        assert "test_stroke_1" in self.window._scene._strokes

    def test_on_ocr_done_with_smart_ink_review(self):
        ocr_res = OcrResult(
            text="E = mc^2",
            latex=r"E = mc^2",
            is_formula=True,
            confidence=0.98,
        )
        ocr_res._board_x = 200.0  # type: ignore[attr-defined]
        ocr_res._board_y = 300.0  # type: ignore[attr-defined]

        # Patch SmartInkReviewDialog.exec so it does not block the test runner
        with patch.object(SmartInkReviewDialog, "exec") as mock_exec:
            mock_exec.return_value = 1
            self.window._on_ocr_done(ocr_res)

        assert mock_exec.called
