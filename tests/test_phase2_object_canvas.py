#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Verification Suite: Object-Based Canvas Scene Graph & Infinite Pan/Zoom.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

import main as app_main
from cv_engine.canvas_scene import (
    CanvasObject,
    SlideSceneGraph,
    compute_resized_bbox,
    make_shape_object,
    make_sticky_object,
    make_text_object,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_scene_graph_crud_zorder_marquee_and_undo_redo():
    sc = SlideSceneGraph(layer=0)
    s1 = make_shape_object("rect", 100.0, 100.0, 140.0, 90.0)
    t1 = make_text_object("Phương trình bậc 2", 320.0, 140.0)
    k1 = make_sticky_object("Lưu ý công thức Delta", 150.0, 120.0)
    sc.add_object(s1, select=False)
    sc.add_object(t1, select=False)
    sc.add_object(k1, select=False)

    # Topmost object hit-test via Rust engine (k1 overlaps s1 at (180, 150))
    hit_top = sc.hit_test(180.0, 150.0)
    assert hit_top is not None and hit_top.id == k1.id

    # Send k1 to back -> s1 becomes topmost at (180, 150)
    sc.select_only(k1.id)
    assert sc.send_selected_to_back()
    hit_after_back = sc.hit_test(180.0, 150.0)
    assert hit_after_back is not None and hit_after_back.id == s1.id

    # Bring k1 back to front
    sc.select_only(k1.id)
    assert sc.bring_selected_to_front()
    assert sc.hit_test(180.0, 150.0).id == k1.id

    # Marquee multi-select catches s1 & k1 (not t1 at x=320..500)
    sel = sc.marquee_select(90.0, 90.0, 200.0, 200.0)
    sel_ids = {o.id for o in sel}
    assert s1.id in sel_ids and k1.id in sel_ids and t1.id not in sel_ids

    # Duplicate selected (Ctrl+D)
    clones = sc.duplicate_selected(offset_x=30.0, offset_y=20.0)
    assert len(clones) == 2
    assert len(sc.objects) == 5

    # Undo duplicate -> back to 3 objects; Redo -> 5 objects
    assert sc.undo()
    assert len(sc.objects) == 3
    assert sc.redo()
    assert len(sc.objects) == 5

    # Delete selected -> back to 3; Undo delete -> 5
    deleted = sc.delete_selected()
    assert deleted == 2
    assert len(sc.objects) == 3
    assert sc.undo()
    assert len(sc.objects) == 5


def test_smartboard_canvas_object_move_resize_rotate_and_pan_zoom(qapp, tmp_path: Path):
    canvas = app_main.SmartboardCanvas()
    canvas.resize(1280, 720)
    canvas.active_tool = "SELECT"

    # 1. Create all 5 object types: Stroke, Shape, Text, Sticky, Image
    shape = canvas.add_shape_object("rect", 120.0, 140.0, 160.0, 100.0, select=False)
    txt = canvas.add_text_object("Xin chào AirTeach", 420.0, 140.0, select=False)
    stk = canvas.add_sticky_object("Ghi chú quan trọng", 720.0, 140.0, select=False)

    img_path = tmp_path / "sample_diagram.png"
    qimg = QImage(160, 100, QImage.Format_ARGB32)
    qimg.fill(Qt.cyan)
    assert qimg.save(str(img_path), "PNG")
    assert canvas.insert_image_on_board(img_path)

    # Verify independent objects & legacy sync
    assert len(canvas.scene.objects) == 4
    assert len(canvas.strokes) == 1
    assert len(canvas.annotations) == 2
    assert len(canvas.board_images) == 1

    # 2. Mouse select + drag move on `shape`
    p_press = canvas._board_to_screen(QPointF(160.0, 190.0))
    p_move = canvas._board_to_screen(QPointF(240.0, 250.0))
    canvas.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, p_press, p_press, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    canvas.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, p_move, p_move, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, p_move, p_move, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))

    moved_shape = canvas.scene.get_by_id(shape.id)
    assert moved_shape is not None
    assert abs(moved_shape.x - 200.0) < 2.0
    assert abs(moved_shape.y - 200.0) < 2.0

    # 3. Resize via bottom-right handle ("se")
    handles = moved_shape.handle_positions_world()
    se_scr = canvas._board_to_screen(QPointF(*handles["se"]))
    se_drag = canvas._board_to_screen(QPointF(handles["se"][0] + 60.0, handles["se"][1] + 40.0))
    canvas.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, se_scr, se_scr, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    canvas.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, se_drag, se_drag, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, se_drag, se_drag, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    resized = canvas.scene.get_by_id(shape.id)
    assert resized.w > 210.0 and resized.h > 130.0

    # 4. Rotate via top rotate handle
    rot_handles = resized.handle_positions_world()
    r_scr = canvas._board_to_screen(QPointF(*rot_handles["rotate"]))
    cx, cy = resized.center
    # Drag to the right of center (~90 deg rotation)
    r_drag = canvas._board_to_screen(QPointF(cx + 120.0, cy))
    canvas.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, r_scr, r_scr, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    canvas.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, r_drag, r_drag, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, r_drag, r_drag, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    rotated = canvas.scene.get_by_id(shape.id)
    assert abs(rotated.rotation - 90.0) < 8.0

    # Undo rotation -> returns to 0 deg; Redo -> returns to ~90 deg
    canvas.undo()
    assert abs(canvas.scene.get_by_id(shape.id).rotation) < 1.0
    canvas.redo()
    assert abs(canvas.scene.get_by_id(shape.id).rotation - 90.0) < 8.0

    # 5. Text edit + Duplicate + Delete
    assert canvas.edit_object_text(txt.id, "Nội dung đã cập nhật")
    assert canvas.scene.get_by_id(txt.id).text == "Nội dung đã cập nhật"
    canvas.scene.select_only(stk.id)
    clones = canvas.duplicate_selected_objects()
    assert len(clones) == 1
    assert canvas.delete_selected_objects() == 1

    # 6. Infinite Pan & Zoom + Pan Lock + Fit to Content + Reset View
    # Middle mouse pan works when unlocked
    canvas.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(300, 300), QPointF(300, 300), Qt.MiddleButton, Qt.MiddleButton, Qt.NoModifier))
    canvas.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(420, 360), QPointF(420, 360), Qt.NoButton, Qt.MiddleButton, Qt.NoModifier))
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(420, 360), QPointF(420, 360), Qt.MiddleButton, Qt.NoButton, Qt.NoModifier))
    assert abs(canvas.pan_x - 120.0) < 1.0 and abs(canvas.pan_y - 60.0) < 1.0

    # Pan Lock blocks further panning
    canvas.set_pan_locked(True)
    canvas.apply_pan_delta_norm(0.2, 0.2)
    assert abs(canvas.pan_x - 120.0) < 1.0
    canvas.set_pan_locked(False)

    # Zoom-to-cursor keeps world point under cursor invariant
    cur = QPointF(500.0, 320.0)
    before_b = canvas._screen_to_board(cur)
    canvas.apply_zoom_factor(1.5, (cur.x() / canvas.width(), cur.y() / canvas.height()))
    after_b = canvas._screen_to_board(cur)
    assert math.hypot(before_b.x() - after_b.x(), before_b.y() - after_b.y()) < 1.5

    # Fit to Content & Reset View (100%)
    canvas.fit_to_content()
    assert 0.25 <= canvas.zoom_scale <= 4.0
    canvas.reset_viewport()
    assert abs(canvas.zoom_scale - 1.0) < 1e-4 and abs(canvas.pan_x) < 1e-4 and abs(canvas.pan_y) < 1e-4
    # Mini HUD shows zoom %, coords, pan lock button
    assert "100%" in canvas.nav_hud.zoom_lbl.text()
    assert "X:" in canvas.nav_hud.coord_lbl.text()
