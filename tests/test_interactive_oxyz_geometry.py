"""
AirTeach AI — Tests for Interactive 3D Analytic Geometry (Oxyz) Simulator (Milestone 111)
Verifies 3D Cartesian coordinates, vector cross and dot products, plane equations,
Euclidean distance metrics, touchless air gestures, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_oxyz_geometry import (
    InteractiveOxyzGeometryItem,
    PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import MathTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_oxyz_initialization(qapp):
    item = InteractiveOxyzGeometryItem(pos=(130.0, 50.0))
    try:
        assert item.pos().x() == 130.0
        assert item.pos().y() == 50.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 680.0, 410.0)
        assert item.active_preset.code == "PLANE_THROUGH_THREE_POINTS"
        assert item.azimuth_deg == 35.0
        assert item.elevation_deg == 25.0
        assert item.is_rotating is False
        assert len(item._a) == 3
        assert len(item._b) == 3
        assert len(item._c) == 3
        assert len(item._m) == 3
    finally:
        item.cleanup()


def test_vector_cross_and_dot_products(qapp):
    # Right-hand rule basis vectors: i x j = k
    i = (1.0, 0.0, 0.0)
    j = (0.0, 1.0, 0.0)
    k = (0.0, 0.0, 1.0)
    cross_ij = InteractiveOxyzGeometryItem.vector_cross(i, j)
    assert cross_ij == (0.0, 0.0, 1.0)

    cross_ji = InteractiveOxyzGeometryItem.vector_cross(j, i)
    assert cross_ji == (0.0, 0.0, -1.0)

    # Parallel vectors cross product is 0
    u = (2.0, 3.0, 4.0)
    cross_uu = InteractiveOxyzGeometryItem.vector_cross(u, u)
    assert cross_uu == (0.0, 0.0, 0.0)

    # Dot product
    dot_val = InteractiveOxyzGeometryItem.vector_dot((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert dot_val == 32.0

    # Vector norm
    norm_val = InteractiveOxyzGeometryItem.vector_norm((3.0, 4.0, 0.0))
    assert norm_val == 5.0


def test_plane_equation_and_distance_formulas(qapp):
    item = InteractiveOxyzGeometryItem()
    try:
        # A(2,0,0), B(0,3,0), C(0,0,4)
        item.load_preset("PLANE_THROUGH_THREE_POINTS")
        n = item.plane_normal()
        assert n == (12.0, 8.0, 6.0)

        A, B, C, D = item.plane_equation()
        assert (A, B, C) == (12.0, 8.0, 6.0)
        # Point A must satisfy equation: 12(2) + 8(0) + 6(0) + D = 0 => D = -24
        assert D == -24.0

        # Point M(1, 1, 2): 12(1) + 8(1) + 6(2) - 24 = 12 + 8 + 12 - 24 = 8
        # Distance = |8| / sqrt(12^2 + 8^2 + 6^2) = 8 / sqrt(244) = 8 / 15.6205 ~ 0.512
        dist = item.distance_m_to_plane()
        expected = 8.0 / math.sqrt(12.0 ** 2 + 8.0 ** 2 + 6.0 ** 2)
        assert abs(dist - expected) < 1e-4
    finally:
        item.cleanup()


def test_all_five_presets(qapp):
    item = InteractiveOxyzGeometryItem()
    try:
        for p_code in PRESETS:
            item.load_preset(p_code)
            assert item.active_preset.code == p_code
            n = item.plane_normal()
            assert len(n) == 3
            assert item.distance_m_to_plane() >= 0.0
    finally:
        item.cleanup()


def test_camera_angles_and_rotation_timer(qapp):
    item = InteractiveOxyzGeometryItem()
    try:
        item.set_camera_angles(60.0, 40.0)
        assert item.azimuth_deg == 60.0
        assert item.elevation_deg == 40.0

        # Elevation clamping [-85, 85]
        item.set_camera_angles(60.0, 100.0)
        assert item.elevation_deg == 85.0
        item.set_camera_angles(60.0, -100.0)
        assert item.elevation_deg == -85.0

        # Toggle rotation
        assert item.is_rotating is False
        assert item.toggle_rotation() is True
        assert item.is_rotating is True

        # Tick
        az_before = item.azimuth_deg
        item._on_rotation_tick()
        assert item.azimuth_deg != az_before

        assert item.toggle_rotation() is False
        assert item.is_rotating is False
    finally:
        item.cleanup()


def test_air_pinch_interactions(qapp):
    item = InteractiveOxyzGeometryItem(pos=(100.0, 50.0))
    try:
        # Outside
        assert item.handle_air_pinch(QPointF(10.0, 10.0)) is False

        # Preset button click
        btn_dist = item._btn_rects["preset_DISTANCE_POINT_TO_PLANE"]
        scene_pt_dist = item.mapToScene(btn_dist.center())
        assert item.handle_air_pinch(scene_pt_dist) is True
        assert item.active_preset.code == "DISTANCE_POINT_TO_PLANE"

        # Azimuth left / right
        az_before = item.azimuth_deg
        btn_ar = item._btn_rects["az_right"]
        scene_pt_ar = item.mapToScene(btn_ar.center())
        assert item.handle_air_pinch(scene_pt_ar) is True
        assert item.azimuth_deg == az_before + 15.0

        btn_al = item._btn_rects["az_left"]
        scene_pt_al = item.mapToScene(btn_al.center())
        assert item.handle_air_pinch(scene_pt_al) is True
        assert item.azimuth_deg == az_before

        # Reset view
        btn_rst = item._btn_rects["reset_view"]
        scene_pt_rst = item.mapToScene(btn_rst.center())
        assert item.handle_air_pinch(scene_pt_rst) is True
        assert item.azimuth_deg == 35.0
        assert item.elevation_deg == 25.0

        # Toggle rotation
        btn_rot = item._btn_rects["toggle_rot"]
        scene_pt_rot = item.mapToScene(btn_rot.center())
        assert item.handle_air_pinch(scene_pt_rot) is True
        assert item.is_rotating is True
        assert item.handle_air_pinch(scene_pt_rot) is True
        assert item.is_rotating is False

        # Viewport direct click
        view_r = item._btn_rects["view3d_canvas"]
        scene_pt_view = item.mapToScene(view_r.center())
        assert item.handle_air_pinch(scene_pt_view) is True
    finally:
        item.cleanup()


def test_stem_tools_diagram_generation_and_stamp(qapp):
    items = MathTools.create_oxyz_geometry_diagram(
        center=(400.0, 300.0),
        preset_code="PLANE_THROUGH_THREE_POINTS",
        point_a=(2.0, 0.0, 0.0),
        point_b=(0.0, 3.0, 0.0),
        point_c=(0.0, 0.0, 4.0),
        point_m=(1.0, 1.0, 2.0),
        distance_val=1.25,
    )
    assert isinstance(items, list)
    assert len(items) >= 15

    all_texts = []
    for itm in items:
        if hasattr(itm, "toPlainText"):
            all_texts.append(itm.toPlainText())
    combined = " ".join(all_texts)
    assert "HÌNH HỌC KHÔNG GIAN" in combined or "Oxyz" in combined
    assert "Vector" in combined or "Khoảng cách" in combined or "Phương trình" in combined


def test_stamping_to_scene(qapp):
    board = LessonBoard()
    item = InteractiveOxyzGeometryItem(pos=(100.0, 50.0))
    board.addItem(item)
    try:
        stamp_called = []
        item.stamp_requested.connect(lambda: stamp_called.append(True))
        item.stamp_to_board()
        assert len(stamp_called) == 1
        assert len(board.items()) > 1
    finally:
        item.cleanup()
        board.clear()


def test_lesson_board_lifecycle(qapp):
    board = LessonBoard()
    try:
        assert board.is_oxyz_geometry_visible() is False
        assert board.get_oxyz_geometry() is None

        # Show
        shown_item = board.show_oxyz_geometry(True)
        assert shown_item is not None
        assert board.is_oxyz_geometry_visible() is True
        assert board.get_oxyz_geometry() is shown_item
        assert board.get_oxyz_geometry_item() is shown_item

        # Toggle off
        vis = board.toggle_oxyz_geometry()
        assert vis is False
        assert board.is_oxyz_geometry_visible() is False
        assert board.get_oxyz_geometry() is None

        # Toggle on
        vis = board.toggle_oxyz_geometry()
        assert vis is True
        assert board.is_oxyz_geometry_visible() is True

        # Hide
        board.show_oxyz_geometry(False)
        assert board.is_oxyz_geometry_visible() is False
    finally:
        if board.get_oxyz_geometry_item():
            board.get_oxyz_geometry_item().cleanup()
        board.clear()


def test_paint_headless(qapp):
    item = InteractiveOxyzGeometryItem()
    try:
        img = QImage(680, 410, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()
        assert not img.isNull()
    finally:
        item.cleanup()
