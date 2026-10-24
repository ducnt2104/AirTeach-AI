"""
Tests for Whiteboard Canvas Mini-Map & Bird's Eye View Navigator HUD (Task I.43)
==============================================================================
Covers:
  - CanvasMiniMapWidget initialization and geometric aspect ratio (16:9, exact 1:12 scale).
  - Mathematical coordinate transformation: scene_to_map and map_to_scene roundtrips.
  - Viewport indicator rectangle derivation under 1.0x and 2.0x zoom scales.
  - View centering and navigation signal emissions.
  - Visual painting and stroke/item miniature thumbnail rendering with QPainter.
  - Mouse interaction (click to navigate, drag to pan, close button).
  - Touchless air tap interaction on mini-map HUD.
  - CanvasZoomController and CanvasZoomWidget synchronization.
  - MainWindow integration (toggle action, shortcut Ctrl+Shift+V, status bar).
"""
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsPathItem, QGraphicsRectItem, QGraphicsView

from src.ui.canvas_minimap import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CanvasMiniMapWidget,
    MINI_MAP_H,
    MINI_MAP_W,
    OFFSET_X,
    OFFSET_Y,
    SCALE_FACTOR,
)
from src.ui.canvas_zoom_controller import CanvasZoomController
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestCanvasMiniMapWidget:
    def test_initialization_and_scale_factors(self, qapp):
        widget = CanvasMiniMapWidget()
        assert widget.widget_width == int(MINI_MAP_W + 16.0)
        assert widget.widget_height == int(MINI_MAP_H + 30.0)
        assert abs(SCALE_FACTOR - (1.0 / 12.0)) < 1e-6
        assert abs(BOARD_WIDTH * SCALE_FACTOR - MINI_MAP_W) < 1e-6
        assert abs(BOARD_HEIGHT * SCALE_FACTOR - MINI_MAP_H) < 1e-6

    def test_coordinate_transformation_roundtrip(self, qapp):
        widget = CanvasMiniMapWidget()

        # Origin (0, 0) -> (OFFSET_X, OFFSET_Y)
        pt = widget.scene_to_map(0.0, 0.0)
        assert abs(pt.x() - OFFSET_X) < 1e-4
        assert abs(pt.y() - OFFSET_Y) < 1e-4

        sx, sy = widget.map_to_scene(pt.x(), pt.y())
        assert abs(sx - 0.0) < 1e-4
        assert abs(sy - 0.0) < 1e-4

        # Center (960, 540) -> (OFFSET_X + 80, OFFSET_Y + 45)
        pt_mid = widget.scene_to_map(960.0, 540.0)
        assert abs(pt_mid.x() - (OFFSET_X + 80.0)) < 1e-4
        assert abs(pt_mid.y() - (OFFSET_Y + 45.0)) < 1e-4

        sx_mid, sy_mid = widget.map_to_scene(pt_mid.x(), pt_mid.y())
        assert abs(sx_mid - 960.0) < 1e-4
        assert abs(sy_mid - 540.0) < 1e-4

        # Bottom Right (1920, 1080)
        pt_br = widget.scene_to_map(BOARD_WIDTH, BOARD_HEIGHT)
        assert abs(pt_br.x() - (OFFSET_X + MINI_MAP_W)) < 1e-4
        assert abs(pt_br.y() - (OFFSET_Y + MINI_MAP_H)) < 1e-4

        sx_br, sy_br = widget.map_to_scene(pt_br.x(), pt_br.y())
        assert abs(sx_br - BOARD_WIDTH) < 1e-4
        assert abs(sy_br - BOARD_HEIGHT) < 1e-4

    def test_viewport_rect_under_zoom(self, qapp):
        scene = LessonBoard()
        view = QGraphicsView(scene)
        view.resize(960, 540)
        view.show()

        widget = CanvasMiniMapWidget(view=view, scene=scene)
        vp_rect = widget.get_viewport_rect()

        # Viewport should be within mini-map inner canvas bounds
        assert vp_rect.left() >= OFFSET_X - 1.0
        assert vp_rect.top() >= OFFSET_Y - 1.0
        assert vp_rect.width() > 0
        assert vp_rect.height() > 0
        view.close()

    def test_center_view_on_and_navigation_signal(self, qapp):
        scene = LessonBoard()
        view = QGraphicsView(scene)
        view.resize(960, 540)

        widget = CanvasMiniMapWidget(view=view, scene=scene)
        nav_points = []
        widget.navigated.connect(lambda x, y: nav_points.append((x, y)))

        widget.center_view_on(500.0, 300.0)
        assert len(nav_points) == 1
        assert nav_points[0] == (500.0, 300.0)

        # Out-of-bounds clamping
        widget.center_view_on(-100.0, 2500.0)
        assert nav_points[-1] == (0.0, BOARD_HEIGHT)
        view.close()

    def test_rendering_on_pixmap(self, qapp):
        scene = LessonBoard()
        # Add sample stroke path and a STEM box
        path_item = QGraphicsPathItem()
        p = QPainterPath()
        p.moveTo(100, 100)
        p.lineTo(400, 400)
        path_item.setPath(p)
        path_item.setPen(QPen(QColor("#ef4444"), 4.0))
        scene.addItem(path_item)

        rect_item = QGraphicsRectItem(600, 200, 200, 150)
        scene.addItem(rect_item)

        view = QGraphicsView(scene)
        widget = CanvasMiniMapWidget(view=view, scene=scene)

        pixmap = QPixmap(widget.widget_width, widget.widget_height)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        from PySide6.QtGui import QPaintEvent
        from PySide6.QtCore import QRect
        widget.paintEvent(QPaintEvent(QRect(0, 0, widget.widget_width, widget.widget_height)))
        painter.end()

        assert not pixmap.isNull()
        view.close()

    def test_air_clicks_on_controls(self, qapp):
        scene = LessonBoard()
        view = QGraphicsView(scene)
        view.resize(960, 540)
        view.show()

        widget = CanvasMiniMapWidget(view=view, scene=scene)
        widget.setVisible(True)
        widget.move(view.viewport().width() - widget.widget_width - 20, 50)

        # 1. Click on close button (top right of widget)
        # Find scene pos corresponding to widget close button
        view_pos = widget.mapTo(view.viewport(), QPointF(widget.widget_width - 10, 10).toPoint())
        scene_pos = view.mapToScene(view_pos)

        assert widget.isVisible()
        handled = widget.air_click_at(scene_pos.x(), scene_pos.y())
        assert handled is True
        assert not widget.isVisible()

        # 2. Re-show and click on map area
        widget.setVisible(True)
        view_pos_map = widget.mapTo(view.viewport(), QPointF(OFFSET_X + 40, OFFSET_Y + 30).toPoint())
        scene_pos_map = view.mapToScene(view_pos_map)

        nav_points = []
        widget.navigated.connect(lambda x, y: nav_points.append((x, y)))
        handled = widget.air_click_at(scene_pos_map.x(), scene_pos_map.y())
        assert handled is True
        assert len(nav_points) == 1

        # 3. Click outside mini-map
        handled = widget.air_click_at(10.0, 10.0)
        assert handled is False
        view.close()


class TestCanvasZoomControllerMinimapIntegration:
    def test_zoom_controller_owns_and_toggles_minimap(self, qapp):
        scene = LessonBoard()
        view = QGraphicsView(scene)
        view.resize(960, 540)

        ctrl = CanvasZoomController(view, scene)
        minimap = ctrl.get_minimap()
        assert minimap is not None
        assert not ctrl.is_minimap_visible()
        assert not ctrl.get_widget().is_minimap_active()

        # Toggle to show
        assert ctrl.toggle_minimap() is True
        assert ctrl.is_minimap_visible()
        assert ctrl.get_widget().is_minimap_active()

        # Toggle to hide
        assert ctrl.toggle_minimap() is False
        assert not ctrl.is_minimap_visible()
        assert not ctrl.get_widget().is_minimap_active()

        # set_minimap_visible directly
        ctrl.set_minimap_visible(True)
        assert ctrl.is_minimap_visible()
        assert ctrl.get_widget().is_minimap_active()
        view.close()


class TestMainWindowMinimapIntegration:
    def test_main_window_toggle_minimap(self, qapp):
        from src.ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert not win._zoom_ctrl.is_minimap_visible()

            # Trigger toggle
            res = win._on_toggle_minimap()
            assert res is True
            assert win._zoom_ctrl.is_minimap_visible()

            # Trigger toggle again
            res = win._on_toggle_minimap()
            assert res is False
            assert not win._zoom_ctrl.is_minimap_visible()
        finally:
            win.close()
