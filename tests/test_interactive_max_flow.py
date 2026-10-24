"""
AirTeach AI — Tests for Interactive Maximum Network Flow & Minimum Cut Simulator (Milestone 110)
Verifies Edmonds-Karp BFS shortest augmenting paths, residual network capacity tracking,
Max-Flow Min-Cut Theorem equality, touchless air gestures, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_max_flow import (
    InteractiveMaxFlowNetworkItem,
    PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import InformaticsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_max_flow_initialization(qapp):
    item = InteractiveMaxFlowNetworkItem(pos=(110.0, 45.0))
    try:
        assert item.pos().x() == 110.0
        assert item.pos().y() == 45.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 680.0, 410.0)
        assert item.active_preset.code == "CLASSIC_EDMONDS_KARP"
        assert item.total_flow == 0
        assert item.step_count == 0
        assert item.is_optimal is False
        assert item.is_running is False
        assert len(item._flows) > 0
    finally:
        item.cleanup()


def test_max_flow_all_presets(qapp):
    item = InteractiveMaxFlowNetworkItem()
    try:
        for p_code in PRESETS:
            item.load_preset(p_code)
            assert item.active_preset.code == p_code
            assert len(item.active_preset.nodes) >= 5
            assert len(item.active_preset.edges) >= 5
            assert item.active_preset.source in item.active_preset.nodes
            assert item.active_preset.sink in item.active_preset.nodes
            assert item.total_flow == 0
            assert item.is_optimal is False
    finally:
        item.cleanup()


def test_edmonds_karp_bfs_algorithm_and_max_flow_min_cut(qapp):
    item = InteractiveMaxFlowNetworkItem()
    try:
        # 1. Classic 6-node network (Expected Max Flow = 23)
        item.load_preset("CLASSIC_EDMONDS_KARP")
        steps = 0
        while item.step_edmonds_karp():
            steps += 1
            assert steps <= 10  # Must terminate quickly

        assert item.is_optimal is True
        assert item.total_flow == 23
        assert item.min_cut_capacity == 23
        assert item.total_flow == item.min_cut_capacity  # Max-Flow == Min-Cut!
        assert "S" in item._s_set
        assert "T" in item._t_set
        assert item._s_set.isdisjoint(item._t_set)
        assert len(item._cut_edges) > 0

        # 2. Bottleneck Bridge Network (Expected Max Flow = 15)
        item.load_preset("BOTTLENECK_BRIDGE")
        steps = 0
        while item.step_edmonds_karp():
            steps += 1
            assert steps <= 10

        assert item.is_optimal is True
        assert item.total_flow == 15
        assert item.min_cut_capacity == 15
        assert item.total_flow == item.min_cut_capacity

        # 3. Maximum Bipartite Matching (Expected Max Flow = 4)
        item.load_preset("BIPARTITE_MATCHING")
        steps = 0
        while item.step_edmonds_karp():
            steps += 1
            assert steps <= 10

        assert item.is_optimal is True
        assert item.total_flow == 4
        assert item.min_cut_capacity == 4
    finally:
        item.cleanup()


def test_reset_flow(qapp):
    item = InteractiveMaxFlowNetworkItem()
    try:
        item.load_preset("CLASSIC_EDMONDS_KARP")
        assert item.step_edmonds_karp() is True
        assert item.total_flow > 0
        assert item.step_count == 1

        item.reset_flow()
        assert item.total_flow == 0
        assert item.step_count == 0
        assert item.is_optimal is False
        assert len(item._last_path) == 0
    finally:
        item.cleanup()


def test_auto_run_toggle(qapp):
    item = InteractiveMaxFlowNetworkItem()
    try:
        assert item.is_running is False
        assert item.toggle_auto_run() is True
        assert item.is_running is True

        # Tick step
        item._on_timer_step()
        assert item.total_flow > 0

        assert item.toggle_auto_run() is False
        assert item.is_running is False
    finally:
        item.cleanup()


def test_air_pinch_interactions(qapp):
    item = InteractiveMaxFlowNetworkItem(pos=(100.0, 50.0))
    try:
        # Outside
        assert item.handle_air_pinch(QPointF(10.0, 10.0)) is False

        # Preset button click
        btn_bridge = item._btn_rects["preset_BOTTLENECK_BRIDGE"]
        scene_pt_bridge = item.mapToScene(btn_bridge.center())
        assert item.handle_air_pinch(scene_pt_bridge) is True
        assert item.active_preset.code == "BOTTLENECK_BRIDGE"

        # Step button click
        flow_before = item.total_flow
        btn_step = item._btn_rects["step_bfs"]
        scene_pt_step = item.mapToScene(btn_step.center())
        assert item.handle_air_pinch(scene_pt_step) is True
        assert item.total_flow > flow_before

        # Reset button click
        btn_rst = item._btn_rects["reset_flow"]
        scene_pt_rst = item.mapToScene(btn_rst.center())
        assert item.handle_air_pinch(scene_pt_rst) is True
        assert item.total_flow == 0

        # Auto-run toggle click
        btn_auto = item._btn_rects["toggle_auto"]
        scene_pt_auto = item.mapToScene(btn_auto.center())
        assert item.handle_air_pinch(scene_pt_auto) is True
        assert item.is_running is True
        assert item.handle_air_pinch(scene_pt_auto) is True
        assert item.is_running is False
    finally:
        item.cleanup()


def test_stem_tools_diagram_generation_and_stamp(qapp):
    items = InformaticsTools.create_max_flow_diagram(
        center=(400.0, 300.0),
        preset_code="CLASSIC_EDMONDS_KARP",
        total_flow=23,
        is_optimal=True,
        min_cut_cap=23,
    )
    assert isinstance(items, list)
    assert len(items) >= 15

    all_texts = []
    for itm in items:
        if hasattr(itm, "toPlainText"):
            all_texts.append(itm.toPlainText())
    combined = " ".join(all_texts)
    assert "MAX-FLOW" in combined or "LUỒNG CỰC ĐẠI" in combined
    assert "MIN-CUT" in combined or "Edmonds-Karp" in combined or "Ford-Fulkerson" in combined


def test_stamping_to_scene(qapp):
    board = LessonBoard()
    item = InteractiveMaxFlowNetworkItem(pos=(100.0, 50.0))
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
        assert board.is_max_flow_visible() is False
        assert board.get_max_flow() is None

        # Show
        shown_item = board.show_max_flow(True)
        assert shown_item is not None
        assert board.is_max_flow_visible() is True
        assert board.get_max_flow() is shown_item
        assert board.get_max_flow_item() is shown_item

        # Toggle off
        vis = board.toggle_max_flow()
        assert vis is False
        assert board.is_max_flow_visible() is False
        assert board.get_max_flow() is None

        # Toggle on
        vis = board.toggle_max_flow()
        assert vis is True
        assert board.is_max_flow_visible() is True

        # Hide
        board.show_max_flow(False)
        assert board.is_max_flow_visible() is False
    finally:
        if board.get_max_flow_item():
            board.get_max_flow_item().cleanup()
        board.clear()


def test_paint_headless(qapp):
    item = InteractiveMaxFlowNetworkItem()
    try:
        img = QImage(680, 410, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()
        assert not img.isNull()
    finally:
        item.cleanup()
