"""
AirTeach AI - Interactive Whiteboard Measurement Tools Tests
============================================================
Validates:
  1. VirtualRulerItem: Millimeter/centimeter calibrations, geometry, rotation, and rendering.
  2. VirtualProtractorItem: 180° semicircular disk, degree graduations, needle drag & clamping.
  3. LessonBoard: Dynamic toggling, visibility queries, and background/selection isolation.
  4. MainWindow: Action shortcuts, status bar feedback, and floating menu action routing.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow
from src.ui.measurement_tools import VirtualProtractorItem, VirtualRulerItem


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_virtual_ruler_properties_and_rendering(qapp):
    """Verify VirtualRulerItem dimensions, graduations, angle setting, and painter rendering."""
    ruler = VirtualRulerItem(length_cm=12.0, pixels_per_cm=50.0, height_px=80.0)
    assert ruler.length_cm == 12.0
    assert ruler.pixels_per_cm == 50.0
    assert ruler.length_px == 600.0
    assert ruler.height_px == 80.0

    # Bounding rect
    br = ruler.boundingRect()
    assert br.width() >= 600.0
    assert br.height() >= 80.0

    # Set angle
    ruler.set_angle(45.0)
    assert ruler.rotation() == 45.0

    ruler.set_angle(375.0)
    assert ruler.rotation() == 15.0

    # Paint into an offscreen image to verify zero paint crashes
    img = QImage(700, 200, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    try:
        ruler.paint(painter, None, None)
    finally:
        painter.end()


def test_virtual_protractor_properties_and_rendering(qapp):
    """Verify VirtualProtractorItem 180° disk, needle clamping, and painter rendering."""
    prot = VirtualProtractorItem(radius=180.0)
    assert prot.radius == 180.0
    assert 0.0 <= prot.needle_angle <= 180.0

    # Needle angle clamping
    prot.set_angle(90.0)
    assert prot.needle_angle == 90.0

    prot.set_angle(210.0)  # Clamps to 180°
    assert prot.needle_angle == 180.0

    prot.set_angle(-15.0)  # Clamps to 0°
    assert prot.needle_angle == 0.0

    # Bounding rect
    br = prot.boundingRect()
    assert br.width() >= 360.0
    assert br.height() >= 180.0

    # Paint into an offscreen image to verify zero paint crashes
    img = QImage(500, 300, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    try:
        prot.paint(painter, None, None)
    finally:
        painter.end()


def test_lesson_board_ruler_and_protractor_lifecycle(qapp):
    """Verify LessonBoard show/toggle/isolation of ruler and protractor instruments."""
    board = LessonBoard()

    assert not board.is_ruler_visible()
    assert not board.is_protractor_visible()

    # 1. Toggle Ruler
    res = board.toggle_ruler()
    assert res is True
    assert board.is_ruler_visible()

    res2 = board.toggle_ruler()
    assert res2 is False
    assert not board.is_ruler_visible()

    board.show_ruler(True, x=200.0, y=150.0)
    assert board.is_ruler_visible()
    assert board._ruler_item.pos() == QPointF(200.0, 150.0)

    # 2. Toggle Protractor
    p_res = board.toggle_protractor()
    assert p_res is True
    assert board.is_protractor_visible()

    p_res2 = board.toggle_protractor()
    assert p_res2 is False
    assert not board.is_protractor_visible()

    board.show_protractor(True, x=450.0, y=300.0)
    assert board.is_protractor_visible()
    assert board._protractor_item.pos() == QPointF(450.0, 300.0)

    # 3. Isolation test: instruments must NOT be returned in get_items_in_rect
    rect = QRectF(0.0, 0.0, 1920.0, 1080.0)
    items = board.get_items_in_rect(rect)
    assert board._ruler_item not in items
    assert board._protractor_item not in items

    # 4. Clear page annotations must preserve instruments
    board.clear_page_annotations()
    assert board.is_ruler_visible()
    assert board.is_protractor_visible()


def test_main_window_measurement_tools_integration(qapp):
    """Verify MainWindow slots and floating menu routing for ruler and protractor."""
    win = MainWindow(camera_index=-1)
    win.show()

    assert not win._scene.is_ruler_visible()
    assert not win._scene.is_protractor_visible()

    # Toggle ruler via slot
    win._on_toggle_ruler()
    assert win._scene.is_ruler_visible()

    win._on_toggle_ruler()
    assert not win._scene.is_ruler_visible()

    # Toggle protractor via slot
    win._on_toggle_protractor()
    assert win._scene.is_protractor_visible()

    win._on_toggle_protractor()
    assert not win._scene.is_protractor_visible()

    # Trigger via in-air floating menu action
    win._on_menu_action_triggered("instrument", "ruler")
    assert win._scene.is_ruler_visible()

    win._on_menu_action_triggered("instrument", "protractor")
    assert win._scene.is_protractor_visible()

    win.close()


def test_spotlight_overlay_properties_and_rendering(qapp):
    """Verify SpotlightOverlayItem properties, bounding rect, position update, and rendering."""
    from src.ui.spotlight_overlay import SpotlightOverlayItem

    spot = SpotlightOverlayItem(board_width=1920, board_height=1080, initial_radius=140.0)
    assert spot.board_width == 1920
    assert spot.board_height == 1080
    assert spot.radius() == 140.0
    assert spot.spotlight_pos() == (960.0, 540.0)

    # Bounding rect
    br = spot.boundingRect()
    assert br.width() == 1920.0
    assert br.height() == 1080.0

    # Position & radius updates
    spot.set_spotlight_pos(400.0, 300.0)
    assert spot.spotlight_pos() == (400.0, 300.0)

    spot.set_radius(200.0)
    assert spot.radius() == 200.0

    # Clamping radius
    spot.set_radius(10.0)
    assert spot.radius() == 40.0

    spot.set_radius(900.0)
    assert spot.radius() == 450.0

    # Rendering without crashes
    spot.setVisible(True)
    img = QImage(1920, 1080, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    try:
        spot.paint(painter, None, None)
    finally:
        painter.end()


def test_lesson_board_spotlight_lifecycle(qapp):
    """Verify LessonBoard spotlight enabling, moving, toggling, and isolation."""
    board = LessonBoard()

    assert not board.is_spotlight_enabled()

    # Toggle spotlight
    st = board.toggle_spotlight()
    assert st is True
    assert board.is_spotlight_enabled()

    st2 = board.toggle_spotlight()
    assert st2 is False
    assert not board.is_spotlight_enabled()

    # Enable and move spotlight
    board.enable_spotlight(True, radius=150.0)
    assert board.is_spotlight_enabled()
    board.move_spotlight(500.0, 350.0)
    assert board._spotlight_item.spotlight_pos() == (500.0, 350.0)

    # Isolation check: spotlight must NOT be returned in get_items_in_rect
    rect = QRectF(0.0, 0.0, 1920.0, 1080.0)
    items = board.get_items_in_rect(rect)
    assert board._spotlight_item not in items

    # Clear page annotations must preserve spotlight
    board.clear_page_annotations()
    assert board.is_spotlight_enabled()


def test_main_window_spotlight_integration(qapp):
    """Verify MainWindow spotlight toggle slot, menu action, and gesture routing."""
    win = MainWindow(camera_index=-1)
    win.show()

    assert win._teaching_mode != "SPOTLIGHT"
    assert not win._scene.is_spotlight_enabled()

    # Toggle spotlight via slot
    win._on_toggle_spotlight()
    assert win._teaching_mode == "SPOTLIGHT"
    assert win._scene.is_spotlight_enabled()

    win._on_toggle_spotlight()
    assert win._teaching_mode == "PEN"
    assert not win._scene.is_spotlight_enabled()

    # Trigger via in-air floating menu action
    win._on_menu_action_triggered("tool", "spotlight")
    assert win._teaching_mode == "SPOTLIGHT"
    assert win._scene.is_spotlight_enabled()

    win.close()


def test_virtual_loupe_properties_and_rendering(qapp):
    """Verify VirtualLoupeItem dimensions, zoom factor, bounding rect, position update, and rendering."""
    from src.ui.virtual_loupe import VirtualLoupeItem

    loupe = VirtualLoupeItem(board_width=1920, board_height=1080, radius=120.0, zoom_factor=2.5)
    assert loupe.board_width == 1920
    assert loupe.board_height == 1080
    assert loupe.radius() == 120.0
    assert loupe.zoom_factor() == 2.5
    assert loupe.loupe_pos() == (960.0, 540.0)

    # Bounding rect
    br = loupe.boundingRect()
    assert br.width() >= 240.0
    assert br.height() >= 240.0

    # Position & zoom factor updates
    loupe.set_loupe_pos(450.0, 320.0)
    assert loupe.loupe_pos() == (450.0, 320.0)

    loupe.set_zoom_factor(3.0)
    assert loupe.zoom_factor() == 3.0

    # Clamping zoom & radius
    loupe.set_zoom_factor(0.5)
    assert loupe.zoom_factor() == 1.2

    loupe.set_zoom_factor(10.0)
    assert loupe.zoom_factor() == 5.0

    loupe.set_radius(20.0)
    assert loupe.radius() == 50.0

    loupe.set_radius(800.0)
    assert loupe.radius() == 300.0

    # Rendering without crashes
    loupe.setVisible(True)
    img = QImage(600, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    try:
        loupe.paint(painter, None, None)
    finally:
        painter.end()


def test_lesson_board_loupe_lifecycle(qapp):
    """Verify LessonBoard loupe enabling, moving, toggling, and isolation."""
    board = LessonBoard()

    assert not board.is_loupe_enabled()

    # Toggle loupe
    lt = board.toggle_loupe()
    assert lt is True
    assert board.is_loupe_enabled()

    lt2 = board.toggle_loupe()
    assert lt2 is False
    assert not board.is_loupe_enabled()

    # Enable and move loupe
    board.enable_loupe(True, zoom=2.5, radius=130.0)
    assert board.is_loupe_enabled()
    board.move_loupe(600.0, 400.0)
    assert board._loupe_item.loupe_pos() == (600.0, 400.0)

    # Change zoom
    board.set_loupe_zoom(3.5)
    assert board._loupe_item.zoom_factor() == 3.5

    # Isolation check: loupe must NOT be returned in get_items_in_rect
    rect = QRectF(0.0, 0.0, 1920.0, 1080.0)
    items = board.get_items_in_rect(rect)
    assert board._loupe_item not in items

    # Clear page annotations must preserve loupe
    board.clear_page_annotations()
    assert board.is_loupe_enabled()


def test_main_window_loupe_integration(qapp):
    """Verify MainWindow loupe toggle slot, menu action, and mode switching."""
    win = MainWindow(camera_index=-1)
    win.show()

    assert win._teaching_mode != "LOUPE"
    assert not win._scene.is_loupe_enabled()

    # Toggle loupe via slot
    win._on_toggle_loupe()
    assert win._teaching_mode == "LOUPE"
    assert win._scene.is_loupe_enabled()

    win._on_toggle_loupe()
    assert win._teaching_mode == "PEN"
    assert not win._scene.is_loupe_enabled()

    # Trigger via in-air floating menu action
    win._on_menu_action_triggered("tool", "loupe")
    assert win._teaching_mode == "LOUPE"
    assert win._scene.is_loupe_enabled()

    win.close()


def test_presentation_timer_properties_and_rendering(qapp):
    """Verify PresentationTimerItem countdown, stopwatch, presets, start/pause, and rendering."""
    from src.ui.presentation_timer import PresentationTimerItem

    timer = PresentationTimerItem(duration_sec=180)
    assert timer.width == 230.0
    assert timer.height == 120.0
    assert timer.total_seconds() == 180.0
    assert timer.remaining_seconds() == 180.0
    assert not timer.is_running()
    assert not timer.is_stopwatch()

    # Bounding rect
    br = timer.boundingRect()
    assert br.width() >= 230.0
    assert br.height() >= 120.0

    # Start and pause
    timer.start()
    assert timer.is_running()

    # Simulate tick
    timer._on_tick()
    assert timer.remaining_seconds() < 180.0

    timer.pause()
    assert not timer.is_running()

    # Reset
    timer.reset()
    assert timer.remaining_seconds() == 180.0

    # Switch to stopwatch mode (duration = 0)
    timer.set_duration(0)
    assert timer.is_stopwatch()
    assert timer.remaining_seconds() == 0.0

    timer.start()
    timer._on_tick()
    assert timer.remaining_seconds() > 0.0
    timer.pause()

    # Rendering without crashes
    img = QImage(300, 200, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    try:
        timer.paint(painter, None, None)
    finally:
        painter.end()


def test_lesson_board_timer_lifecycle(qapp):
    """Verify LessonBoard presentation timer show, toggle, isolation, and preservation."""
    board = LessonBoard()

    assert not board.is_timer_visible()

    # Toggle timer
    tt = board.toggle_timer()
    assert tt is True
    assert board.is_timer_visible()

    tt2 = board.toggle_timer()
    assert tt2 is False
    assert not board.is_timer_visible()

    # Show timer with specific coordinates
    board.show_timer(True, x=1500.0, y=100.0, duration_sec=300)
    assert board.is_timer_visible()
    assert board._timer_item.pos() == QPointF(1500.0, 100.0)

    # Start, pause, reset via board methods
    board.start_timer()
    assert board._timer_item.is_running()

    board.pause_timer()
    assert not board._timer_item.is_running()

    board.reset_timer()
    assert board._timer_item.remaining_seconds() == 300.0

    # Isolation check: timer must NOT be returned in get_items_in_rect
    rect = QRectF(0.0, 0.0, 1920.0, 1080.0)
    items = board.get_items_in_rect(rect)
    assert board._timer_item not in items

    # Clear page annotations must preserve timer
    board.clear_page_annotations()
    assert board.is_timer_visible()


def test_main_window_timer_integration(qapp):
    """Verify MainWindow timer toggle slot and in-air menu routing."""
    win = MainWindow(camera_index=-1)
    win.show()

    assert not win._scene.is_timer_visible()

    # Toggle timer via slot
    win._on_toggle_timer()
    assert win._scene.is_timer_visible()

    win._on_toggle_timer()
    assert not win._scene.is_timer_visible()

    # Trigger via in-air floating menu action
    win._on_menu_action_triggered("instrument", "timer")
    assert win._scene.is_timer_visible()

    win.close()
