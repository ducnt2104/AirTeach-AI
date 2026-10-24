"""
AirTeach AI - Transparent Desktop Annotation & Screen Overlay Tests
===================================================================
Verifies DesktopOverlayBar acrylic widget, LessonBoard transparent mode,
MainWindow full-screen toggle lifecycle, and seamless in-air tool switching.
"""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QApplication

from src.ui.desktop_overlay_bar import DesktopOverlayBar
from src.ui.lesson_board import LessonBoard, BOARD_WIDTH, BOARD_HEIGHT, THEME_CONFIGS
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_desktop_overlay_bar_properties_and_signals(qapp):
    """Verify DesktopOverlayBar UI elements, signal emission, and active tool states."""
    bar = DesktopOverlayBar()
    assert bar.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert bar.windowFlags() & Qt.WindowType.WindowStaysOnTopHint

    # Signal emission tracking
    received = []
    bar.pen_requested.connect(lambda: received.append("pen"))
    bar.eraser_requested.connect(lambda: received.append("eraser"))
    bar.laser_requested.connect(lambda: received.append("laser"))
    bar.spotlight_requested.connect(lambda: received.append("spotlight"))
    bar.loupe_requested.connect(lambda: received.append("loupe"))
    bar.timer_requested.connect(lambda: received.append("timer"))
    bar.undo_requested.connect(lambda: received.append("undo"))
    bar.clear_requested.connect(lambda: received.append("clear"))
    bar.exit_requested.connect(lambda: received.append("exit"))

    # Click buttons
    bar.btn_pen.click()
    assert "pen" in received
    assert bar._active_tool == "pen"

    bar.btn_eraser.click()
    assert "eraser" in received
    assert bar._active_tool == "eraser"

    bar.btn_laser.click()
    assert "laser" in received
    assert bar._active_tool == "laser"

    bar.btn_spotlight.click()
    assert "spotlight" in received
    assert bar._active_tool == "spotlight"

    bar.btn_loupe.click()
    assert "loupe" in received
    assert bar._active_tool == "loupe"

    bar.btn_timer.click()
    assert "timer" in received

    # Test set_active_tool
    bar.set_active_tool("pen")
    assert bar._active_tool == "pen"


def test_desktop_overlay_bar_dragging_and_paint(qapp):
    """Verify mouse press, move, release, and acrylic paintEvent execution."""
    bar = DesktopOverlayBar()
    bar.resize(600, 44)

    # Trigger paintEvent
    bar.repaint()

    # Simulate drag
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPoint(10, 10),
        QPoint(100, 100),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    bar.mousePressEvent(press_event)
    assert bar._drag_start is not None

    move_event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPoint(30, 30),
        QPoint(120, 120),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    bar.mouseMoveEvent(move_event)

    release_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPoint(30, 30),
        QPoint(120, 120),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    bar.mouseReleaseEvent(release_event)
    assert bar._drag_start is None


def test_lesson_board_desktop_overlay_mode(qapp):
    """Verify LessonBoard transparent canvas mode toggles and background brush."""
    board = LessonBoard()
    assert board.is_desktop_overlay_mode() is False

    # Enable desktop overlay
    board.set_desktop_overlay_mode(True)
    assert board.is_desktop_overlay_mode() is True
    assert board.backgroundBrush().color() == QColor(Qt.GlobalColor.transparent)

    # Disable desktop overlay
    board.set_desktop_overlay_mode(False)
    assert board.is_desktop_overlay_mode() is False
    assert board.backgroundBrush().color() == THEME_CONFIGS[board.board_theme()]["bg_color"]


def test_main_window_desktop_overlay_integration(qapp):
    """Verify MainWindow full desktop transparent annotation lifecycle."""
    win = MainWindow(camera_index=-1)
    win.show()

    assert win.is_desktop_overlay_active() is False

    # Toggle to Desktop Mode
    active = win._on_toggle_desktop_overlay()
    assert active is True
    assert win.is_desktop_overlay_active() is True
    assert win._scene.is_desktop_overlay_mode() is True
    assert win._sidebar.isVisible() is False
    assert win._chat_panel.isVisible() is False
    assert win._desktop_overlay_bar is not None
    assert win._desktop_overlay_bar.isVisible() is True

    # Test desktop bar actions
    win._on_desktop_bar_pen()
    assert win._teaching_mode == "PEN"

    win._on_desktop_bar_eraser()
    assert win._teaching_mode == "ERASER"

    win._on_desktop_bar_laser()
    assert win._teaching_mode == "LASER"

    win._on_desktop_bar_clear()

    # Toggle back to Whiteboard
    active_again = win._on_toggle_desktop_overlay()
    assert active_again is False
    assert win.is_desktop_overlay_active() is False
    assert win._scene.is_desktop_overlay_mode() is False
    assert win._sidebar.isVisible() is True
    assert win._chat_panel.isVisible() is True
    assert win._desktop_overlay_bar.isVisible() is False

    # Test in-air menu action trigger for desktop
    win._on_menu_action_triggered("action", "toggle_desktop")
    assert win.is_desktop_overlay_active() is True

    win._on_menu_action_triggered("action", "toggle_desktop")
    assert win.is_desktop_overlay_active() is False

    win.close()
