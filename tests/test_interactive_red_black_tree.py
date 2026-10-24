"""
AirTeach AI – Tests for Interactive Red-Black Self-Balancing Tree Simulator (Milestone 115)
Verifies CLRS Chapter 13 Red-Black Tree invariants, Left/Right rotations, insertion & deletion fixups,
worst-case O(log n) balancing, Linux CFS scheduler, touchless air gestures, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_red_black_tree import (
    InteractiveRedBlackTreeItem,
    NodeColor,
    RBNode,
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


def test_red_black_initialization(qapp):
    item = InteractiveRedBlackTreeItem(pos=(140.0, 50.0))
    try:
        assert item.pos().x() == 140.0
        assert item.pos().y() == 50.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 700.0, 440.0)
        assert item.active_preset.code == "CLASSIC_CONSTRUCTION"
        assert item.node_count == len(PRESETS["CLASSIC_CONSTRUCTION"].keys)
        assert item.root.color == NodeColor.BLACK
        assert item.black_height >= 1
        assert item.tree_height <= 2 * math.ceil(math.log2(item.node_count + 1))
        is_valid, _ = item.verify_invariants()
        assert is_valid is True
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_clrs_5_invariants_verification(qapp):
    item = InteractiveRedBlackTreeItem()
    try:
        is_valid, msgs = item.verify_invariants()
        assert is_valid is True
        assert len(msgs) == 5

        # Check Property 2: Root is always BLACK
        assert item.root.color == NodeColor.BLACK

        # Artificially violate Property 2 and check verification catches it
        item.root.color = NodeColor.RED
        bad_valid, bad_msgs = item.verify_invariants()
        assert bad_valid is False
        assert any("tính chất 2" in m.lower() for m in bad_msgs)

        # Restore root color
        item.root.color = NodeColor.BLACK
        restored_valid, _ = item.verify_invariants()
        assert restored_valid is True
    finally:
        item.cleanup()


def test_worst_case_sorted_sequence_self_balancing(qapp):
    """
    Inserting strictly sorted elements [1..8] would create a degenerate linked-list
    of height 8 in an unbalanced BST. Red-Black Tree must self-balance to height <= 4.
    """
    item = InteractiveRedBlackTreeItem()
    try:
        item.load_preset("WORST_CASE_DEGENERATE_SORTED")
        assert item.node_count == 8

        # Unbalanced BST height would be 8
        # Red-Black tree height must be <= 4 (or strictly <= 2 * log2(9))
        assert item.tree_height <= 4
        assert item.black_height == 2

        is_valid, _ = item.verify_invariants()
        assert is_valid is True
    finally:
        item.cleanup()


def test_rotations_and_rebalancing_mechanics(qapp):
    item = InteractiveRedBlackTreeItem()
    try:
        item.clear()
        # Insert 3 keys that trigger rotation
        item.insert(10)
        item.insert(20)
        item.insert(30)

        # After inserting 10, 20, 30: 20 must become BLACK root, 10 and 30 RED children
        assert item.root.key == 20
        assert item.root.color == NodeColor.BLACK
        assert item.root.left.key == 10
        assert item.root.left.color == NodeColor.RED
        assert item.root.right.key == 30
        assert item.root.right.color == NodeColor.RED

        is_valid, _ = item.verify_invariants()
        assert is_valid is True
    finally:
        item.cleanup()


def test_insertion_and_deletion_operations(qapp):
    item = InteractiveRedBlackTreeItem()
    try:
        item.clear()
        initial_keys = [50, 25, 75, 12, 37, 62, 87]
        for k in initial_keys:
            item.insert(k)
        assert item.node_count == 7
        is_valid_ins, _ = item.verify_invariants()
        assert is_valid_ins is True

        # Search existing and non-existing keys
        assert item.search(37) != item.NIL
        assert item.search(37).key == 37
        assert item.search(999) == item.NIL

        # Delete leaf node
        del_12 = item.delete(12)
        assert del_12 is True
        assert item.node_count == 6
        assert item.search(12) == item.NIL
        is_valid_del1, _ = item.verify_invariants()
        assert is_valid_del1 is True

        # Delete internal node with children
        del_50 = item.delete(50)
        assert del_50 is True
        assert item.node_count == 5
        assert item.search(50) == item.NIL
        is_valid_del2, _ = item.verify_invariants()
        assert is_valid_del2 is True

        # Delete non-existent key returns False
        assert item.delete(999) is False
    finally:
        item.cleanup()


def test_all_curriculum_presets(qapp):
    item = InteractiveRedBlackTreeItem()
    try:
        preset_keys = [
            "CLASSIC_CONSTRUCTION",
            "WORST_CASE_DEGENERATE_SORTED",
            "ROTATION_TRIANGLE_ZIGZAG",
            "BLACK_HEIGHT_VERIFICATION",
            "LINUX_KERNEL_SCHEDULER_CFS",
        ]
        for key in preset_keys:
            item.load_preset(key)
            assert item.active_preset.code == key
            assert len(item.active_preset.title_vi) > 0
            assert len(item.active_preset.description) > 0
            assert item.node_count == len(item.active_preset.keys)
            is_valid, _ = item.verify_invariants()
            assert is_valid is True
    finally:
        item.cleanup()


def test_linux_cfs_scheduler_preset(qapp):
    item = InteractiveRedBlackTreeItem()
    try:
        item.load_preset("LINUX_KERNEL_SCHEDULER_CFS")
        # In CFS, minimum vruntime process is the leftmost node in the tree
        lm = item.leftmost_node
        assert lm is not None
        assert lm.key == min(PRESETS["LINUX_KERNEL_SCHEDULER_CFS"].keys)
        assert len(lm.task_name) > 0
        assert lm.vruntime_ms > 0.0
    finally:
        item.cleanup()


def test_parameter_controls_and_steppers(qapp):
    item = InteractiveRedBlackTreeItem()
    try:
        # Key stepper and clamping
        item.set_pending_key(45)
        assert item.pending_key == 45
        item.set_pending_key(-10)
        assert item.pending_key == 1
        item.set_pending_key(5000)
        assert item.pending_key == 999

        # Insert staged key
        item.set_pending_key(99)
        c_before = item.node_count
        item.insert_pending_key()
        assert item.node_count == c_before + 1
        assert item.search(99) != item.NIL

        # Delete staged key
        item.set_pending_key(99)
        item.delete_pending_key()
        assert item.node_count == c_before
        assert item.search(99) == item.NIL

        # Clear tree
        item.clear()
        assert item.node_count == 0
        assert item.root == item.NIL
    finally:
        item.cleanup()


def test_touchless_air_gestures(qapp):
    item = InteractiveRedBlackTreeItem(pos=(0.0, 0.0))
    try:
        # Pre-render to build button rects cache
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        item.paint(p, None, None)
        p.end()

        # 1. Preset tabs
        r_cfs = item._btn_rects.get("preset_LINUX_KERNEL_SCHEDULER_CFS")
        assert r_cfs is not None
        assert item.handle_air_pinch(r_cfs.center()) is True
        assert item.active_preset.code == "LINUX_KERNEL_SCHEDULER_CFS"

        # 2. Key steppers
        r_kp1 = item._btn_rects.get("k_p1")
        assert r_kp1 is not None
        cur_k = item.pending_key
        assert item.handle_air_pinch(r_kp1.center()) is True
        assert item.pending_key == cur_k + 1

        # 3. Insert and Delete buttons
        r_ins = item._btn_rects.get("insert")
        assert r_ins is not None
        c_init = item.node_count
        assert item.handle_air_pinch(r_ins.center()) is True
        assert item.node_count == c_init + 1

        # 4. Out of bounds returns False
        assert item.handle_air_pinch(QPointF(-100.0, -100.0)) is False
    finally:
        item.cleanup()


def test_paint_rendering_all_presets(qapp):
    item = InteractiveRedBlackTreeItem()
    try:
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        for preset_key in PRESETS:
            item.load_preset(preset_key)
            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()
            assert not img.isNull()
    finally:
        item.cleanup()


def test_stem_tools_vector_diagram(qapp):
    items = InformaticsTools.create_red_black_tree_diagram(
        center=(400.0, 300.0),
        width=640.0,
        height=340.0,
        preset_code="CLASSIC_CONSTRUCTION",
        node_count=7,
        black_height=2,
        tree_height=4,
    )
    assert len(items) >= 10
    assert items[0].rect().width() == 640.0
    assert items[0].rect().height() == 340.0


def test_lesson_board_integration(qapp):
    board = LessonBoard()
    try:
        assert board.is_red_black_tree_visible() is False
        assert board.get_red_black_tree() is None

        # Toggle ON
        vis1 = board.toggle_red_black_tree()
        assert vis1 is True
        assert board.is_red_black_tree_visible() is True
        item = board.get_red_black_tree()
        assert item is not None
        assert isinstance(item, InteractiveRedBlackTreeItem)

        # Excluded from stroke selection
        excluded = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert item not in excluded

        # Toggle OFF
        vis2 = board.toggle_red_black_tree()
        assert vis2 is False
        assert board.is_red_black_tree_visible() is False
        assert board.get_red_black_tree() is None
        assert board.get_red_black_tree_item() is not None
    finally:
        item = board.get_red_black_tree_item()
        if item is not None and hasattr(item, "cleanup"):
            item.cleanup()
