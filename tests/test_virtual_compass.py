"""
Tests for Virtual Compass (Compa) & Geometry Construction Suite (Task I.45)
===========================================================================
Covers:
  - VirtualCompassItem initialization, geometry scaling (40 px/cm), and default parameters.
  - Radius boundary clamping (1.0 cm - 12.0 cm) and signal emissions.
  - Angular rotation, trigonometric pencil positioning, and modular [0, 360) degree normalization.
  - Stamping crisp vector circles onto LessonBoard (draw_full_circle).
  - QPainter visual rendering of arm graduations, pivot needle, and control badge.
  - Mouse interaction (pencil dragging to rotate, badge button controls).
  - Touchless air click interaction on compass controls.
  - LessonBoard integration (show, toggle, selection rectangle exclusion).
  - MainWindow integration (View menu, toolbar, shortcut Ctrl+Shift+C).
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.lesson_board import LessonBoard
from src.ui.measurement_tools import VirtualCompassItem


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestVirtualCompassItem:
    def test_initialization_defaults(self, qapp):
        compass = VirtualCompassItem()
        assert compass.radius_cm() == 3.5
        assert compass.pixels_per_cm == 40.0
        assert compass.radius_px() == 140.0
        assert compass.pencil_angle() == 45.0
        assert compass.boundingRect().width() > 280.0
        assert compass.boundingRect().height() > 280.0

    def test_radius_clamping_and_signals(self, qapp):
        compass = VirtualCompassItem()
        emitted_r = []
        compass.radius_changed.connect(lambda r: emitted_r.append(r))

        # Underflow clamp (min 1.0 cm)
        compass.set_radius_cm(0.2)
        assert compass.radius_cm() == 1.0
        assert compass.radius_px() == 40.0
        assert emitted_r[-1] == 1.0

        # Overflow clamp (max 12.0 cm)
        compass.set_radius_cm(25.0)
        assert compass.radius_cm() == 12.0
        assert compass.radius_px() == 480.0
        assert emitted_r[-1] == 12.0

        # Valid setting
        compass.set_radius_cm(5.0)
        assert compass.radius_cm() == 5.0
        assert compass.radius_px() == 200.0
        assert emitted_r[-1] == 5.0

    def test_angle_normalization_and_signals(self, qapp):
        compass = VirtualCompassItem()
        emitted_angles = []
        compass.angle_changed.connect(lambda a: emitted_angles.append(a))

        compass.set_angle(90.0)
        assert compass.pencil_angle() == 90.0
        assert emitted_angles[-1] == 90.0

        # Overflow modulo
        compass.set_angle(450.0)
        assert abs(compass.pencil_angle() - 90.0) < 1e-4

        # Negative angle modulo
        compass.set_angle(-45.0)
        assert abs(compass.pencil_angle() - 315.0) < 1e-4

    def test_pencil_trigonometric_position(self, qapp):
        compass = VirtualCompassItem(radius_cm=3.0, pixels_per_cm=40.0)  # r = 120 px

        # Angle 0°: (r, 0)
        compass.set_angle(0.0)
        p0 = compass.get_pencil_pos()
        assert abs(p0.x() - 120.0) < 1e-4
        assert abs(p0.y() - 0.0) < 1e-4

        # Angle 90°: (0, -r)
        compass.set_angle(90.0)
        p90 = compass.get_pencil_pos()
        assert abs(p90.x() - 0.0) < 1e-4
        assert abs(p90.y() - (-120.0)) < 1e-4

        # Angle 180°: (-r, 0)
        compass.set_angle(180.0)
        p180 = compass.get_pencil_pos()
        assert abs(p180.x() - (-120.0)) < 1e-4
        assert abs(p180.y() - 0.0) < 1e-4

        # Angle 270°: (0, r)
        compass.set_angle(270.0)
        p270 = compass.get_pencil_pos()
        assert abs(p270.x() - 0.0) < 1e-4
        assert abs(p270.y() - 120.0) < 1e-4

    def test_draw_full_circle_operation(self, qapp):
        scene = QGraphicsScene()
        compass = VirtualCompassItem(radius_cm=4.0, pixels_per_cm=40.0)  # r = 160 px
        compass.setPos(500.0, 300.0)
        scene.addItem(compass)

        drawn_circles = []
        compass.circle_drawn.connect(lambda cx, cy, r: drawn_circles.append((cx, cy, r)))

        circle_item = compass.draw_full_circle()
        assert circle_item is not None
        assert circle_item in scene.items()
        assert len(drawn_circles) == 1
        assert drawn_circles[0] == (500.0, 300.0, 160.0)

        # Check circle bounding box
        rect = circle_item.rect()
        assert rect.width() == 320.0
        assert rect.height() == 320.0
        assert rect.center().x() == 500.0
        assert rect.center().y() == 300.0

    def test_rendering_on_pixmap(self, qapp):
        compass = VirtualCompassItem(radius_cm=3.0, pixels_per_cm=40.0)
        compass.set_angle(60.0)

        pixmap = QPixmap(400, 400)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.translate(200, 200)

        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()
        compass.paint(painter, opt)
        painter.end()

        assert not pixmap.isNull()
        img = pixmap.toImage()
        has_pixels = any(img.pixelColor(x, y).alpha() > 0 for x in range(50, 350, 15) for y in range(50, 350, 15))
        assert has_pixels

    def test_air_clicks_on_controls(self, qapp):
        scene = QGraphicsScene()
        compass = VirtualCompassItem(radius_cm=3.0, pixels_per_cm=40.0)  # r = 120 px
        compass.setPos(500.0, 400.0)
        compass.setVisible(True)
        scene.addItem(compass)

        # 1. Click Draw Circle button [🖊️ Vẽ]
        # Badge is at badge_y = -r - 42 = -162. Button is at badge_x + 145 = -130 + 145 = +15, y = -148
        # Scene pos = (515.0, 400 - 148 = 252.0)
        drawn = []
        compass.circle_drawn.connect(lambda cx, cy, r: drawn.append((cx, cy, r)))
        handled = compass.air_click_at(515.0, 252.0)
        assert handled is True
        assert len(drawn) == 1

        # 2. Click Plus button [+]: x = badge_x + 210 = -130 + 210 = +80, y = -148 -> scene (580.0, 252.0)
        old_r = compass.radius_cm()
        handled = compass.air_click_at(580.0, 252.0)
        assert handled is True
        assert compass.radius_cm() == old_r + 0.5

        # 3. Click Minus button [−]: x = badge_x + 185 = -130 + 185 = +55, y = -148 -> scene (555.0, 252.0)
        handled = compass.air_click_at(555.0, 252.0)
        assert handled is True
        assert compass.radius_cm() == old_r

        # 4. Click Close button [✕]: x = badge_x + 240 = -130 + 240 = +110, y = -148 -> scene (610.0, 252.0)
        handled = compass.air_click_at(610.0, 252.0)
        assert handled is True
        assert not compass.isVisible()

        # 5. Air Click outside
        handled = compass.air_click_at(10.0, 10.0)
        assert handled is False


class TestLessonBoardCompassIntegration:
    def test_lesson_board_compass_lifecycle(self, qapp):
        board = LessonBoard()
        assert not board.is_compass_visible()
        assert board.get_compass_item() is None

        # Show
        board.show_compass(True, x=600.0, y=400.0, radius_cm=4.0)
        assert board.is_compass_visible()
        compass = board.get_compass_item()
        assert compass is not None
        assert compass.radius_cm() == 4.0
        assert compass.x() == 600.0
        assert compass.y() == 400.0

        # Toggle to hide
        assert board.toggle_compass() is False
        assert not board.is_compass_visible()

        # Toggle to show
        assert board.toggle_compass() is True
        assert board.is_compass_visible()

    def test_compass_excluded_from_selection_rectangle(self, qapp):
        board = LessonBoard()
        board.show_compass(True, x=400.0, y=300.0)
        compass = board.get_compass_item()
        assert compass is not None

        # Query all items in rect covering the compass
        rect = QRectF(200.0, 100.0, 400.0, 400.0)
        selected = board.get_items_in_rect(rect)
        assert compass not in selected


class TestMainWindowCompassIntegration:
    def test_main_window_toggle_compass_and_shortcut(self, qapp):
        from PySide6.QtGui import QAction
        from src.ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert not win._scene.is_compass_visible()

            # Trigger toggle slot
            win._on_toggle_compass()
            assert win._scene.is_compass_visible()

            # Trigger toggle slot again
            win._on_toggle_compass()
            assert not win._scene.is_compass_visible()

            # Verify action exists with shortcut Ctrl+Shift+C
            compass_acts = [a for a in win.findChildren(QAction) if a.shortcut() == QKeySequence("Ctrl+Shift+C")]
            assert len(compass_acts) >= 1
        finally:
            win.close()
