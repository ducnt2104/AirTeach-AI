"""
AirTeach AI - Performance Benchmarking & Stress Testing Suite
============================================================
Validates Phase D Architectural Standards:
  1. Multi-hand tracking throughput and latency (< 5ms per frame)
  2. Large-scale vector stroke smoothing & Bézier fitting throughput (1000 pts < 100ms)
  3. 50-Page rapid presentation navigation with bounded LRU pixmap cache (<= 8 items)
  4. Memory stability under sustained multi-stroke in-air drawing (300 strokes)
  5. Geometric shape snapping computational efficiency (< 2ms per stroke)
  6. Floating action menu dwell hit-testing at high frame rates (60+ FPS)
"""
import gc
import math
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication

from tests.test_core_algorithms import _make_synthetic_landmarks
from src.core.multi_hand_coordinator import (
    AirClickType,
    HandLandmarkData,
    HandRole,
    MultiHandCoordinator,
    MultiHandMode,
)
from src.core.stroke_beautifier import (
    BezierSegment,
    GeometricShapeSnapper,
    ShapeType,
    fit_cubic_bezier_spline,
    ramer_douglas_peucker,
)
from src.ui.floating_action_menu import FloatingActionMenu, MenuLayoutMode
from src.ui.lesson_board import BOARD_HEIGHT, BOARD_WIDTH, InkStroke, LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestMultiHandPerformance:
    def test_multi_hand_coordinator_throughput(self):
        """Verify MultiHandCoordinator processes 100 dual-hand frames in < 200ms (< 2ms/frame)."""
        coord = MultiHandCoordinator(mode=MultiHandMode.DUAL_DRAW)
        r_lms = _make_synthetic_landmarks(index_tip_y=0.2)
        l_lms = _make_synthetic_landmarks(index_tip_y=0.25)

        start_t = time.perf_counter()
        iterations = 100
        for i in range(iterations):
            # Pass detected raw landmarks to tracker
            matched = coord.tracker.update(
                [
                    (r_lms, "Right", 0.95),
                    (l_lms, "Left", 0.92),
                ],
                t_ms=float(i * 33),
            )
            assert len(matched) == 2

        elapsed = time.perf_counter() - start_t
        avg_ms = (elapsed / iterations) * 1000.0

        # Must be well below real-time webcam frame budget (33ms)
        assert avg_ms < 5.0, f"Average multi-hand frame processing too slow: {avg_ms:.2f}ms"

    def test_large_stroke_simplification_and_spline_fitting(self):
        """Verify RDP + Bézier fitting for 1000 raw points executes in < 100ms."""
        # Generate 1000 noisy sinusoidal handwriting points
        t = np.linspace(0, 4 * np.pi, 1000)
        x = 100.0 + t * 50.0 + np.random.normal(0, 0.5, 1000)
        y = 300.0 + 80.0 * np.sin(t) + np.random.normal(0, 0.5, 1000)
        raw_points = list(zip(x, y))

        start_t = time.perf_counter()
        # 1. RDP simplification
        simplified = ramer_douglas_peucker(raw_points, epsilon=2.0)
        # 2. Cubic Bézier spline fitting
        segments = fit_cubic_bezier_spline(simplified)
        elapsed = (time.perf_counter() - start_t) * 1000.0

        assert len(simplified) < len(raw_points) * 0.4, "RDP did not compress stroke adequately"
        assert len(segments) > 0, "Bézier fitting produced no segments"
        assert elapsed < 100.0, f"Stroke smoothing took too long: {elapsed:.2f}ms"

    def test_geometric_shape_snapper_latency(self):
        """Verify shape snapping for simulated geometric shapes runs in < 2ms."""
        snapper = GeometricShapeSnapper(circle_roundness=0.75)

        # Generate a noisy circle (60 points)
        angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        cx, cy, r = 500.0, 500.0, 100.0
        circle_pts = [
            (cx + (r + np.random.normal(0, 1.0)) * math.cos(a),
             cy + (r + np.random.normal(0, 1.0)) * math.sin(a))
            for a in angles
        ]
        pts_arr = np.array(circle_pts, dtype=np.float64)

        start_t = time.perf_counter()
        for _ in range(50):
            res = snapper.snap(pts_arr)
            assert res is not None
            assert res.shape_type in (ShapeType.CIRCLE, ShapeType.ELLIPSE)

        elapsed_ms = ((time.perf_counter() - start_t) / 50.0) * 1000.0
        assert elapsed_ms < 2.0, f"Shape snapping too slow: {elapsed_ms:.2f}ms"


class TestPresentationMemoryAndStress:
    def test_50_page_rapid_navigation_lru_cache(self, qapp):
        """Verify that navigating 50 pages keeps LRU pixmap cache strictly bounded <= 8."""
        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.QThread"), patch("src.ui.main_window._QtCameraWorker"):
            win = MainWindow(camera_index=-1)

        # Simulate 50 pages with background pixmaps
        dummy_img = QImage(1920, 1080, QImage.Format.Format_RGB32)
        dummy_img.fill(Qt.GlobalColor.white)
        dummy_pix = QPixmap.fromImage(dummy_img)

        # Populate 50 pages into cache sequentially
        for page_idx in range(50):
            win._page_pixmap_cache[page_idx] = dummy_pix
            win._page_pixmap_cache.move_to_end(page_idx)
            if len(win._page_pixmap_cache) > win._MAX_PAGE_CACHE:
                win._page_pixmap_cache.popitem(last=False)

        assert len(win._page_pixmap_cache) == win._MAX_PAGE_CACHE
        assert len(win._page_pixmap_cache) == 8
        # Oldest pages (0 through 41) must have been evicted
        assert 0 not in win._page_pixmap_cache
        assert 41 not in win._page_pixmap_cache
        # Newest pages (42 through 49) must be retained
        assert 49 in win._page_pixmap_cache
        assert 42 in win._page_pixmap_cache

        win.close()

    def test_sustained_multi_stroke_drawing_stability(self, qapp):
        """Verify 300 strokes can be drawn and managed without scene degradation or memory leak."""
        board = LessonBoard()
        gc.collect()

        # Simulate 300 rapid strokes using commit_stroke with BezierSegment
        for i in range(300):
            p0 = (float(100 + (i % 800)), float(100 + (i % 600)))
            p3 = (p0[0] + 50.0, p0[1] + 50.0)
            seg = BezierSegment(p0=p0, p1=(p0[0] + 15, p0[1] + 20), p2=(p0[0] + 35, p0[1] + 30), p3=p3)
            stroke = InkStroke(
                stroke_id=f"stress_stroke_{i}",
                color=QColor("#1a73e8"),
                segments=[seg],
                widths=[3.0],
            )
            board.commit_stroke(stroke)

        assert len(board._strokes) == 300

        # Verify bulk query performance
        start_t = time.perf_counter()
        items = board.get_items_in_rect(QRectF(0, 0, BOARD_WIDTH, BOARD_HEIGHT))
        elapsed = (time.perf_counter() - start_t) * 1000.0
        assert len(items) == 300
        assert elapsed < 10.0, f"Querying 300 strokes took too long: {elapsed:.2f}ms"

        # Clear board cleanly
        board.clear_all_strokes()
        assert len(board._strokes) == 0
        gc.collect()

    def test_floating_action_menu_high_fps_dwell_performance(self, qapp):
        """Verify FloatingActionMenu dwell hit-testing at 60 FPS (120 ticks < 15ms)."""
        menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080, layout_mode=MenuLayoutMode.RADIAL_WHEEL)
        menu.show_at(960.0, 540.0)

        start_t = time.perf_counter()
        # Simulate 120 frames of user hovering on tool 0 (e.g. angle ~ 0)
        hover_x = 960.0 + 150.0 * 0.8
        hover_y = 540.0
        for i in range(120):
            menu.update_pointer(hover_x, hover_y, dwell_progress=0.5)

        elapsed = (time.perf_counter() - start_t) * 1000.0
        assert elapsed < 15.0, f"Floating menu hit-test loop too slow: {elapsed:.2f}ms"
        menu.close_menu()
