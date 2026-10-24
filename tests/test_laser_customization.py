"""
Tests for Presentation Laser Pointer Color & Size Customization Suite (Task I.44)
================================================================================
Covers:
  - LessonBoard laser color, radius, and trail customization defaults.
  - Setting laser color and propagating to active QGraphicsEllipseItem and LaserTrailItem.
  - Setting laser radius with boundary clamping (4.0 - 36.0 px) and rect adjustment.
  - Laser color preset cycling (Ruby -> Emerald -> Cyan -> Amber -> Violet).
  - Automatic application in move_laser() tracking pipeline.
  - SettingsDialog laser controls (combos, checkbox, properties).
  - MainWindow integration: _on_cycle_laser_color, shortcut Ctrl+L, and settings application.
"""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import QApplication

from src.ui.lesson_board import LessonBoard
from src.ui.settings_dialog import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestLessonBoardLaserCustomization:
    def test_laser_customization_defaults(self, qapp):
        board = LessonBoard()
        assert board.laser_color() == "#ef4444"
        assert board.laser_radius() == 12.0
        assert board.is_laser_trail_enabled() is True

    def test_set_laser_color_updates_items_and_trails(self, qapp):
        board = LessonBoard()
        board.move_laser(100, 100, visible=True, hand_id="primary")
        item = board._laser_items["primary"]
        assert item is not None

        # Change color to Emerald
        board.set_laser_color("#10b981")
        assert board.laser_color() == "#10b981"
        assert item.pen().color().name().lower() == "#10b981"

        # If trail exists, verify trail color updated
        if "primary" in board._laser_trails:
            trail = board._laser_trails["primary"]
            assert trail.color().name().lower() == "#10b981"

    def test_set_laser_radius_clamping_and_rect(self, qapp):
        board = LessonBoard()
        board.move_laser(200, 200, visible=True, hand_id="primary")
        item = board._laser_items["primary"]

        # Valid radius
        board.set_laser_radius(18.0)
        assert board.laser_radius() == 18.0
        assert item.rect().width() == 36.0
        assert item.rect().height() == 36.0

        # Underflow clamp (min 4.0)
        board.set_laser_radius(1.0)
        assert board.laser_radius() == 4.0
        assert item.rect().width() == 8.0

        # Overflow clamp (max 36.0)
        board.set_laser_radius(100.0)
        assert board.laser_radius() == 36.0
        assert item.rect().width() == 72.0

    def test_cycle_laser_color_presets(self, qapp):
        board = LessonBoard()
        # Default is Ruby (#ef4444, index 0)
        # Cycle 1 -> Emerald (#10b981)
        hex_c, name, emoji = board.cycle_laser_color()
        assert hex_c == "#10b981"
        assert "Lục Bảo" in name
        assert emoji == "🟢"
        assert board.laser_color() == "#10b981"

        # Cycle 2 -> Cyan (#06b6d4)
        hex_c, name, emoji = board.cycle_laser_color()
        assert hex_c == "#06b6d4"
        assert "Cyan" in name
        assert emoji == "🔵"

        # Cycle 3 -> Amber (#f59e0b)
        hex_c, name, emoji = board.cycle_laser_color()
        assert hex_c == "#f59e0b"
        assert "Hổ Phách" in name
        assert emoji == "🟡"

        # Cycle 4 -> Violet (#8b5cf6)
        hex_c, name, emoji = board.cycle_laser_color()
        assert hex_c == "#8b5cf6"
        assert "Tím" in name
        assert emoji == "🟣"

        # Cycle 5 -> Wrap around to Ruby (#ef4444)
        hex_c, name, emoji = board.cycle_laser_color()
        assert hex_c == "#ef4444"
        assert "Ruby" in name
        assert emoji == "🔴"

    def test_move_laser_adopts_custom_settings(self, qapp):
        board = LessonBoard()
        board.set_laser_color("#06b6d4")
        board.set_laser_radius(16.0)

        # Calling move_laser with default args
        board.move_laser(350, 450, visible=True, hand_id="hand_2")
        item = board._laser_items["hand_2"]
        assert item.pen().color().name().lower() == "#06b6d4"
        assert item.rect().width() == 32.0


class TestSettingsDialogLaserIntegration:
    def test_settings_dialog_laser_controls(self, qapp):
        dlg = SettingsDialog(teacher_id="test_t", teacher_name="Test Teacher")

        # Initial default values
        assert dlg.laser_color == "#ef4444"
        assert dlg.laser_radius == 12.0
        assert dlg.laser_trail_enabled is True

        # Modify values
        dlg.set_laser_color("#8b5cf6")
        assert dlg.laser_color == "#8b5cf6"

        dlg.set_laser_radius(18.0)
        assert dlg.laser_radius == 18.0

        dlg.set_laser_trail_enabled(False)
        assert dlg.laser_trail_enabled is False


class TestMainWindowLaserIntegration:
    def test_main_window_laser_cycling_and_shortcut(self, qapp):
        from PySide6.QtGui import QAction
        from src.ui.main_window import MainWindow

        win = MainWindow()
        try:
            initial_color = win._scene.laser_color()

            # Trigger cycling slot
            win._on_cycle_laser_color()
            assert win._scene.laser_color() != initial_color

            # Verify action exists with shortcut Ctrl+L
            laser_acts = [a for a in win.findChildren(QAction) if a.shortcut() == QKeySequence("Ctrl+L")]
            assert len(laser_acts) >= 1
        finally:
            win.close()
