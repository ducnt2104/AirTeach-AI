"""
Unit tests for Dynamic Fading Laser Trail and Gesture Calibration Wizard.

Validates:
- LaserTrailItem point buffering, lifetime pruning, neon rendering, and idle timer stopping
- LessonBoard laser trail lifecycle, multi-hand independent trails, and enable/disable controls
- GestureCalibrationDialog slider adjustments, default resets, and target click testing
- MainWindow calibration integration
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.gesture_calibration_dialog import GestureCalibrationDialog
from src.ui.laser_trail_item import LaserTrailItem
from src.ui.lesson_board import BOARD_HEIGHT, BOARD_WIDTH, LessonBoard


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── 1. LaserTrailItem Tests ───────────────────────────────────────────────────

def test_laser_trail_initialization(qapp):
    """Test LaserTrailItem default properties."""
    trail = LaserTrailItem(color="#ff1744", lifetime_sec=1.5, base_width=6.0)
    assert trail.color() == QColor("#ff1744")
    assert trail.lifetime() == 1.5
    assert trail.zValue() == 39
    assert len(trail._points) == 0
    assert not trail._timer.isActive()


def test_laser_trail_add_point_and_timer_activation(qapp):
    """Test adding points starts the timer and updates bounding rect."""
    trail = LaserTrailItem(lifetime_sec=1.0)
    trail.add_point(100.0, 100.0)
    trail.add_point(150.0, 150.0)

    assert len(trail._points) == 2
    assert trail._timer.isActive()
    bounds = trail.boundingRect()
    assert bounds.contains(100.0, 100.0)
    assert bounds.contains(150.0, 150.0)


def test_laser_trail_jitter_filtering(qapp):
    """Test micro-movements < 2px are filtered to prevent point bloat."""
    trail = LaserTrailItem()
    trail.add_point(200.0, 200.0)
    trail.add_point(200.5, 200.5)  # distance < 1px -> should be ignored
    trail.add_point(201.0, 201.0)  # distance < 2px -> should be ignored

    assert len(trail._points) == 1


def test_laser_trail_pruning_and_timer_stop(qapp):
    """Test points older than lifetime are pruned and timer stops when empty."""
    trail = LaserTrailItem(lifetime_sec=0.2)
    t_past = time.perf_counter() - 0.5  # 500ms ago (> 200ms lifetime)

    trail.add_point(50.0, 50.0, timestamp=t_past)
    assert len(trail._points) == 1

    # Simulate timer tick
    trail._on_timer_tick()
    assert len(trail._points) == 0
    assert not trail._timer.isActive()


def test_laser_trail_clear(qapp):
    """Test clear resets all points and stops timer immediately."""
    trail = LaserTrailItem()
    trail.add_point(100.0, 100.0)
    trail.add_point(200.0, 200.0)
    assert trail._timer.isActive()

    trail.clear()
    assert len(trail._points) == 0
    assert not trail._timer.isActive()


def test_laser_trail_paint_execution(qapp):
    """Ensure paint executes smoothly with neon glow and hot core."""
    trail = LaserTrailItem(color="#00e676", lifetime_sec=2.0)
    now = time.perf_counter()
    trail.add_point(100.0, 100.0, timestamp=now - 0.2)
    trail.add_point(150.0, 120.0, timestamp=now - 0.1)
    trail.add_point(200.0, 180.0, timestamp=now)

    img = QImage(300, 300, QImage.Format.Format_ARGB32)
    p = QPainter(img)
    trail.paint(p, None)
    p.end()


# ── 2. LessonBoard Laser Trail Integration Tests ──────────────────────────────

def test_lesson_board_laser_trail_lifecycle(qapp):
    """Test LessonBoard automatically creates and feeds laser trail on move_laser."""
    board = LessonBoard()
    assert board.is_laser_trail_enabled() is True
    assert board.laser_trail_lifetime() == 1.5

    # Move primary laser
    board.move_laser(500.0, 500.0, visible=True, hand_id="primary", smooth=False)
    board.move_laser(520.0, 520.0, visible=True, hand_id="primary", smooth=False)

    assert "primary" in board._laser_trails
    trail = board._laser_trails["primary"]
    assert isinstance(trail, LaserTrailItem)
    assert len(trail._points) == 2

    # Move secondary laser independently
    board.move_laser(800.0, 800.0, visible=True, hand_id="secondary")
    assert "secondary" in board._laser_trails
    assert board._laser_trails["secondary"] is not trail


def test_lesson_board_disable_laser_trail(qapp):
    """Test disabling laser trail clears existing trails."""
    board = LessonBoard()
    board.move_laser(300.0, 300.0, visible=True)
    board.move_laser(350.0, 350.0, visible=True)
    assert len(board._laser_trails["primary"]._points) > 0

    board.set_laser_trail_enabled(False)
    assert board.is_laser_trail_enabled() is False
    assert len(board._laser_trails["primary"]._points) == 0


def test_lesson_board_set_laser_trail_lifetime(qapp):
    """Test modifying trail lifetime updates active trails."""
    board = LessonBoard()
    board.move_laser(100.0, 100.0, visible=True)
    board.set_laser_trail_lifetime(2.5)
    assert board.laser_trail_lifetime() == 2.5
    assert board._laser_trails["primary"].lifetime() == 2.5


# ── 3. GestureCalibrationDialog Tests ─────────────────────────────────────────

def test_gesture_calibration_dialog_initialization(qapp):
    """Test dialog controls build with expected initial values."""
    dlg = GestureCalibrationDialog()
    assert dlg.slider_sens.value() == 10
    assert dlg.slider_dwell.value() == 600
    assert dlg.radio_right.isChecked()
    assert dlg.chk_laser_trail.isChecked()
    assert dlg.slider_trail_time.value() == 15


def test_gesture_calibration_dialog_slider_changes(qapp):
    """Test slider value changes update corresponding label displays."""
    dlg = GestureCalibrationDialog()

    dlg.slider_sens.setValue(18)
    assert dlg.lbl_sens_val.text() == "1.8x"

    dlg.slider_dwell.setValue(750)
    assert dlg.lbl_dwell_val.text() == "750 ms"

    dlg.slider_trail_time.setValue(22)
    assert dlg.lbl_trail_time.text() == "2.2 giây"


def test_gesture_calibration_dialog_reset_defaults(qapp):
    """Test reset button restores standard defaults."""
    dlg = GestureCalibrationDialog()
    dlg.slider_sens.setValue(20)
    dlg.slider_dwell.setValue(1000)
    dlg.radio_left.setChecked(True)
    dlg.chk_laser_trail.setChecked(False)

    dlg._on_reset_defaults()
    assert dlg.slider_sens.value() == 10
    assert dlg.slider_dwell.value() == 600
    assert dlg.radio_right.isChecked()
    assert dlg.chk_laser_trail.isChecked()
    assert dlg.slider_trail_time.value() == 15


def test_gesture_calibration_dialog_target_practice(qapp):
    """Test clicking practice target increments test counter and updates label."""
    dlg = GestureCalibrationDialog()
    assert dlg._click_test_count == 0

    dlg.btn_target.click()
    assert dlg._click_test_count == 1
    assert "Số lần bấm thành công: 1 lần" in dlg.lbl_test_stats.text()

    dlg.btn_target.click()
    assert dlg._click_test_count == 2
    assert "Số lần bấm thành công: 2 lần" in dlg.lbl_test_stats.text()


def test_gesture_calibration_dialog_apply_signal(qapp):
    """Test applying calibration emits signal with collected settings."""
    dlg = GestureCalibrationDialog()
    dlg.slider_sens.setValue(15)
    dlg.slider_dwell.setValue(800)
    dlg.radio_left.setChecked(True)
    dlg.chk_laser_trail.setChecked(True)
    dlg.slider_trail_time.setValue(20)

    applied_data = []
    dlg.calibration_applied.connect(applied_data.append)

    dlg._on_save_and_apply()
    assert len(applied_data) == 1
    d = applied_data[0]
    assert d["sensitivity"] == 1.5
    assert d["dwell_ms"] == 800
    assert d["dominant_hand"] == "left"
    assert d["laser_trail_enabled"] is True
    assert d["laser_trail_lifetime"] == 2.0


# ── 4. MainWindow Integration Tests ───────────────────────────────────────────

def test_main_window_calibration_dialog_integration(qapp):
    """Test MainWindow opens calibration dialog and applies laser trail updates."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_open_gesture_calibration")

        # Create dialog with win as main_window
        dlg = GestureCalibrationDialog(main_window=win)
        dlg.chk_laser_trail.setChecked(False)
        dlg.slider_trail_time.setValue(25)
        dlg._on_save_and_apply()

        # Verify board settings updated
        assert win._scene.is_laser_trail_enabled() is False
        assert win._scene.laser_trail_lifetime() == 2.5
    finally:
        win.close()
