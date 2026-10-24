"""
AirTeach AI - PerspectiveMapper Hardening, ReplayController & Lesson Package Tests
==================================================================================
Validates:
  1. PerspectiveMapper valid quadrilateral calibration and coordinate mapping.
  2. PerspectiveMapper immunity against collinear, zero-area, NaN, and degenerate corners.
  3. PerspectiveMapper NaN/inf coordinate mapping safety (returns (0.0, 0.0) without throwing).
  4. PerspectiveMapper reset_calibration restoring full-frame identity.
  5. MultiHandCoordinator perspective calibration and inheritance for dynamically spawned FSMs.
  6. HandTrackingWorker unified perspective calibration and set_air_click_type delegation.
  7. ReplayController 0.5x slow-motion, 1.5x, and programmatic set_speed() controls.
  8. Full .airteach lesson package roundtrip: serialization, deserialization, and canvas stroke reconstruction.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath
from PySide6.QtWidgets import QApplication

from src.core.hand_gesture_engine import (
    HandGestureFSM,
    HandTrackingWorker,
    PerspectiveMapper,
)
from src.core.multi_hand_coordinator import (
    AirClickType,
    HandRole,
    MultiHandCoordinator,
    MultiHandMode,
)
from src.core.stroke_beautifier import BezierSegment
from src.services.document_manager import (
    AirTeachManifest,
    DocumentManager,
    StrokeRecord,
    BezierPointRecord,
)
from src.ui.lesson_board_canvas import InkStroke, LessonBoardCanvas
from src.ui.main_window import MainWindow, _QtCameraWorker
from src.ui.replay_controller import ReplayController


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# =============================================================================
# 1. PerspectiveMapper Unit & Hardening Tests
# =============================================================================

class TestPerspectiveMapperHardening:
    def test_default_identity_mapping(self):
        """Default perspective mapper maps (0,0) -> (0,0) and (1,1) -> (W-1, H-1)."""
        mapper = PerspectiveMapper(1920, 1080)
        x0, y0 = mapper.map(0.0, 0.0)
        assert abs(x0 - 0.0) < 1e-3
        assert abs(y0 - 0.0) < 1e-3

        x1, y1 = mapper.map(1.0, 1.0)
        assert abs(x1 - 1919.0) < 1e-2
        assert abs(y1 - 1079.0) < 1e-2

        x_mid, y_mid = mapper.map(0.5, 0.5)
        assert abs(x_mid - 959.5) < 1e-2
        assert abs(y_mid - 539.5) < 1e-2

    def test_valid_quadrilateral_calibration(self):
        """Calibrating to a sub-rectangle scales points appropriately."""
        mapper = PerspectiveMapper(1920, 1080)
        # Inner zone: [0.2, 0.2] to [0.8, 0.8]
        sub_corners = np.array(
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            dtype=np.float32,
        )
        ok = mapper.calibrate(sub_corners)
        assert ok is True

        # Now (0.2, 0.2) maps to top-left (0, 0) of canvas
        tl_x, tl_y = mapper.map(0.2, 0.2)
        assert abs(tl_x - 0.0) < 1.0
        assert abs(tl_y - 0.0) < 1.0

        # And (0.8, 0.8) maps to bottom-right of canvas
        br_x, br_y = mapper.map(0.8, 0.8)
        assert abs(br_x - 1919.0) < 1.0
        assert abs(br_y - 1079.0) < 1.0

    def test_reset_calibration(self):
        """reset_calibration restores original identity mapping."""
        mapper = PerspectiveMapper(1920, 1080)
        sub_corners = np.array(
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            dtype=np.float32,
        )
        mapper.calibrate(sub_corners)
        mapper.reset_calibration()

        x1, y1 = mapper.map(1.0, 1.0)
        assert abs(x1 - 1919.0) < 1e-2
        assert abs(y1 - 1079.0) < 1e-2

    def test_adversarial_collinear_and_zero_inputs(self):
        """Degenerate inputs return False without throwing exceptions."""
        mapper = PerspectiveMapper(1920, 1080)

        # All 4 points collinear on a line
        collinear = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype=np.float32)
        assert mapper.calibrate(collinear) is False

        # All zeros
        zeros = np.zeros((4, 2), dtype=np.float32)
        assert mapper.calibrate(zeros) is False

        # Invalid shapes
        assert mapper.calibrate(np.zeros((3, 2), dtype=np.float32)) is False
        assert mapper.calibrate(np.zeros((5, 2), dtype=np.float32)) is False
        assert mapper.calibrate(np.zeros((4, 3), dtype=np.float32)) is False

        # Non-finite values
        with_nan = np.array([[np.nan, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        assert mapper.calibrate(with_nan) is False

        with_inf = np.array([[0.0, 0.0], [np.inf, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        assert mapper.calibrate(with_inf) is False

    def test_map_non_finite_coordinates_safety(self):
        """map() with NaN or Inf returns safe (0.0, 0.0) without crashing."""
        mapper = PerspectiveMapper(1920, 1080)
        assert mapper.map(float("nan"), 0.5) == (0.0, 0.0)
        assert mapper.map(0.5, float("inf")) == (0.0, 0.0)
        assert mapper.map(float("-inf"), float("nan")) == (0.0, 0.0)


# =============================================================================
# 2. Multi-Hand & Tracking Worker Perspective Delegation Tests
# =============================================================================

class TestMultiHandPerspectiveDelegation:
    def test_coordinator_perspective_calibration_and_fsm_inheritance(self):
        """Coordinator updates its perspective and newly spawned hand FSMs inherit calibrated corners."""
        coord = MultiHandCoordinator(canvas_width=1920, canvas_height=1080)
        corners = np.array(
            [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            dtype=np.float32,
        )
        assert coord.calibrate_perspective(corners) is True
        np.testing.assert_allclose(coord.perspective._src_pts, corners)

        # Spawning a new hand FSM dynamically must inherit the calibrated perspective
        fsm1 = coord._get_or_create_fsm("track_01")
        np.testing.assert_allclose(fsm1.perspective._src_pts, corners)

        # Resetting coordinator perspective resets both coordinator and existing FSMs
        coord.reset_perspective()
        identity = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(coord.perspective._src_pts, identity)
        np.testing.assert_allclose(fsm1.perspective._src_pts, identity)

    def test_hand_tracking_worker_perspective_delegation(self):
        """HandTrackingWorker calibrates and resets both FSM and MultiHandCoordinator."""
        worker = HandTrackingWorker(canvas_width=1920, canvas_height=1080)
        corners = np.array(
            [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]],
            dtype=np.float32,
        )
        ok = worker.calibrate_perspective(corners)
        assert ok is True
        np.testing.assert_allclose(worker.fsm.perspective._src_pts, corners)
        np.testing.assert_allclose(worker.coordinator.perspective._src_pts, corners)

        worker.reset_perspective()
        identity = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(worker.fsm.perspective._src_pts, identity)
        np.testing.assert_allclose(worker.coordinator.perspective._src_pts, identity)

    def test_hand_tracking_worker_set_air_click_type(self):
        """HandTrackingWorker delegates set_air_click_type to coordinator."""
        worker = HandTrackingWorker()
        worker.set_air_click_type(AirClickType.PINCH_CLICK)
        assert worker.coordinator.click_detector.click_type == AirClickType.PINCH_CLICK

        worker.set_air_click_type("dwell")
        assert worker.coordinator.click_detector.click_type == AirClickType.DWELL


# =============================================================================
# 3. ReplayController Multi-Speed & Programmatic Controls Tests
# =============================================================================

class TestReplayControllerExtendedSpeeds:
    def test_speed_combo_options_and_indices(self, qapp):
        """ReplayController offers 1.0x, 2.0x, 4.0x, 0.5x, and 1.5x options."""
        ctrl = ReplayController()
        items = [ctrl._combo_speed.itemText(i) for i in range(ctrl._combo_speed.count())]
        assert items == ["1.0x", "2.0x", "4.0x", "0.5x", "1.5x"]

        # Backwards compatible index order
        ctrl._combo_speed.setCurrentIndex(0)
        assert ctrl._speed_multiplier == 1.0
        ctrl._combo_speed.setCurrentIndex(1)
        assert ctrl._speed_multiplier == 2.0
        ctrl._combo_speed.setCurrentIndex(2)
        assert ctrl._speed_multiplier == 4.0

        # Slow motion
        ctrl._combo_speed.setCurrentIndex(3)
        assert ctrl._speed_multiplier == 0.5

        # 1.5x
        ctrl._combo_speed.setCurrentIndex(4)
        assert ctrl._speed_multiplier == 1.5

    def test_programmatic_set_speed(self, qapp):
        """set_speed programmatic method correctly selects combo text or custom multiplier."""
        ctrl = ReplayController()
        ctrl.set_speed(0.5)
        assert ctrl._speed_multiplier == 0.5
        assert ctrl._combo_speed.currentText() == "0.5x"

        ctrl.set_speed(2.0)
        assert ctrl._speed_multiplier == 2.0
        assert ctrl._combo_speed.currentText() == "2.0x"

        # Arbitrary custom speed clamped between 0.1 and 10.0
        ctrl.set_speed(0.75)
        assert ctrl._speed_multiplier == 0.75

        ctrl.set_speed(15.0)
        assert ctrl._speed_multiplier == 10.0

    def test_timer_tick_with_slow_motion(self, qapp):
        """At 0.5x speed, timer tick advances by TIMER_INTERVAL_MS * 0.5."""
        ctrl = ReplayController()
        ctrl.set_speed(0.5)
        ctrl._total_duration_ms = 5000.0
        ctrl._current_time_ms = 100.0

        ctrl._on_timer_tick()
        expected = 100.0 + (ctrl.TIMER_INTERVAL_MS * 0.5)
        assert abs(ctrl._current_time_ms - expected) < 1e-3


# =============================================================================
# 4. .airteach Lesson Package Stroke Restoration Roundtrip
# =============================================================================

class TestAirTeachLessonPackageRoundtrip:
    def test_save_and_reconstruct_lesson_package(self, qapp, tmp_path):
        """Strokes serialized into an .airteach package can be loaded and accurately reconstructed."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        doc_mgr = DocumentManager(work_root=work_dir)

        # 1. Create a dummy background page PNG
        import cv2
        page_img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        page_file = tmp_path / "page_0.png"
        cv2.imwrite(str(page_file), page_img)

        # 2. Build test InkStrokes with Bezier segments and timestamps
        p0 = np.array([100.0, 100.0])
        p1 = np.array([150.0, 120.0])
        p2 = np.array([200.0, 80.0])
        p3 = np.array([250.0, 100.0])
        seg = BezierSegment(p0, p1, p2, p3, width_start=3.0, width_end=4.0)

        record = StrokeRecord(
            stroke_id="test_stroke_42",
            page_index=0,
            color_hex="#1a73e8",
            width=3.5,
            points=[
                BezierPointRecord(x=100.0, y=100.0, t_ms=500.0),
                BezierPointRecord(x=250.0, y=100.0, t_ms=800.0),
            ],
            bezier_segments=[seg.to_dict()],
            timestamps_ms=[500.0, 650.0, 800.0],
            shape_type="FREEFORM",
        )

        manifest = AirTeachManifest(
            title="Kiểm thử bài giảng",
            page_count=1,
            created_at="2026-10-23T20:00:00Z",
            modified_at="2026-10-23T20:00:00Z",
        )

        # 3. Save package
        pkg_path = tmp_path / "test_lesson.airteach"
        doc_mgr.save_lesson(
            manifest=manifest,
            page_image_paths=[page_file],
            strokes=[record],
            recognized=[],
            output_path=pkg_path,
        )
        assert pkg_path.exists()

        # 4. Open package
        out_manifest, out_pages, out_strokes, out_rec = doc_mgr.open_lesson(pkg_path)
        assert out_manifest.title == "Kiểm thử bài giảng"
        assert len(out_pages) == 1
        assert len(out_strokes) == 1

        rec = out_strokes[0]
        assert rec.stroke_id == "test_stroke_42"
        assert rec.color_hex == "#1a73e8"
        assert rec.timestamps_ms == [500.0, 650.0, 800.0]

        # 5. Verify reconstruction into LessonBoardCanvas
        canvas = LessonBoardCanvas()
        # Reconstruct InkStroke
        reconstructed_seg = BezierSegment(
            np.array(rec.bezier_segments[0]["p0"]),
            np.array(rec.bezier_segments[0]["p1"]),
            np.array(rec.bezier_segments[0]["p2"]),
            np.array(rec.bezier_segments[0]["p3"]),
            width_start=rec.bezier_segments[0]["width_start"],
            width_end=rec.bezier_segments[0]["width_end"],
        )
        ink = InkStroke(
            stroke_id=rec.stroke_id,
            segments=[reconstructed_seg],
            widths=[rec.width],
            color=QColor(rec.color_hex),
            timestamps_ms=rec.timestamps_ms,
        )
        canvas._render_and_commit(ink)

        assert rec.stroke_id in canvas._strokes
        committed_stroke = canvas._strokes[rec.stroke_id]
        assert committed_stroke._path_item is not None
        assert committed_stroke._path_item.isVisible()
        assert not committed_stroke._path_item.path().isEmpty()

    def test_main_window_load_airteach_package_full_reconstruction(self, tmp_path, qapp):
        """Verify MainWindow._load_airteach_package restores strokes, recognized LaTeX and text overlays."""
        from src.services.document_manager import RecognizedItem
        from PySide6.QtGui import QImage, QPainter

        work_dir = tmp_path / 'work_dir'
        work_dir.mkdir(exist_ok=True)
        doc_mgr = DocumentManager(work_root=work_dir)

        # 1. Page image
        page_file = tmp_path / "page_001.png"
        img = QImage(1920, 1080, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.white)
        img.save(str(page_file))

        # 2. Stroke
        seg = BezierSegment(
            np.array([100.0, 100.0]),
            np.array([150.0, 80.0]),
            np.array([200.0, 120.0]),
            np.array([250.0, 100.0]),
            width_start=3.0,
            width_end=3.0,
        )
        record = StrokeRecord(
            stroke_id="stroke_rec_1",
            page_index=0,
            color_hex="#1a73e8",
            width=3.0,
            points=[
                BezierPointRecord(x=100.0, y=100.0, t_ms=100.0),
                BezierPointRecord(x=250.0, y=100.0, t_ms=300.0),
            ],
            bezier_segments=[seg.to_dict()],
            timestamps_ms=[100.0, 200.0, 300.0],
            shape_type="FREEFORM",
        )

        # 3. Recognized LaTeX formula and text label
        rec_formula = RecognizedItem(
            item_id="rec_item_1",
            page_index=0,
            text="E = mc^2",
            latex="E = mc^2",
            confidence=0.98,
            bounding_box=[200.0, 300.0, 350.0, 360.0],
            is_formula=True,
        )
        rec_text = RecognizedItem(
            item_id="rec_item_2",
            page_index=0,
            text="Vật lý 12",
            latex="",
            confidence=0.95,
            bounding_box=[400.0, 300.0, 500.0, 340.0],
            is_formula=False,
        )

        manifest = AirTeachManifest(
            title="Gói bài giảng thử nghiệm",
            page_count=1,
            created_at="2026-10-23T20:00:00Z",
            modified_at="2026-10-23T20:00:00Z",
        )

        pkg_path = tmp_path / "complete_lesson.airteach"
        doc_mgr.save_lesson(
            manifest=manifest,
            page_image_paths=[page_file],
            strokes=[record],
            recognized=[rec_formula, rec_text],
            output_path=pkg_path,
        )
        assert pkg_path.exists()

        # 4. Load into MainWindow
        win = MainWindow(camera_index=-1)
        win.show()

        win._load_airteach_package(pkg_path)

        # Verify strokes restored
        assert len(win._page_strokes.get(0, [])) == 1
        stroke = win._page_strokes[0][0]
        assert stroke.stroke_id == "stroke_rec_1"
        assert stroke.color.name().lower() == "#1a73e8"

        # Verify recognized items restored
        stem_items = win._page_stem_items.get(0, [])
        assert len(stem_items) == 2, "Both LaTeX formula and text overlay must be restored into _page_stem_items"

        # Check metadata
        data_0 = stem_items[0].data(1)
        data_1 = stem_items[1].data(1)
        assert data_0 is not None and data_1 is not None

        formulas = [d for d in (data_0, data_1) if d.get("is_formula")]
        texts = [d for d in (data_0, data_1) if not d.get("is_formula")]

        assert len(formulas) == 1
        assert formulas[0]["latex"] == "E = mc^2"
        assert len(texts) == 1
        assert texts[0]["text"] == "Vật lý 12"

        # Both items are visible on page 0
        assert stem_items[0].isVisible()
        assert stem_items[1].isVisible()

        win.close()
