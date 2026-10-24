"""
AirTeach AI - Test Suite for Interactive Binary Tree & Huffman Simulator
=======================================================================
Verifies authentic computer science data structures (BST, Huffman tree, Expression tree),
traversal orders (In-order, Pre-order, Post-order, Level-order), Shannon entropy,
touchless air gestures, vector diagram stamping, LessonBoard integration, and MainWindow actions.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
)

from src.ui.interactive_binary_tree import (
    PRESETS,
    InteractiveBinaryTreeItem,
    TreeNode,
)
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow
from src.ui.stem_tools import InformaticsTools


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def board(qapp):
    return LessonBoard()


@pytest.fixture
def tree_item(qapp):
    return InteractiveBinaryTreeItem()


class TestBinarySearchTreeCore:
    """Verifies BST mathematical properties and algorithmic correctness (Zero Mock)."""

    def test_bst_balanced_inorder_is_sorted(self, tree_item):
        """In-order traversal (LNR) of BST must always produce strictly ascending keys."""
        tree_item.load_preset("BST_BALANCED_SEARCH")
        res: list[TreeNode] = []
        tree_item._collect_inorder(tree_item._root, res)
        keys = [n.key for n in res]
        assert keys == [20, 30, 40, 50, 60, 70, 80]
        assert keys == sorted(keys)

    def test_bst_balanced_traversals(self, tree_item):
        """Verify Pre-order, Post-order, and Level-order traversal sequences."""
        tree_item.load_preset("BST_BALANCED_SEARCH")

        # Pre-order (NLR)
        pre_res: list[TreeNode] = []
        tree_item._collect_preorder(tree_item._root, pre_res)
        assert [n.key for n in pre_res] == [50, 30, 20, 40, 70, 60, 80]

        # Post-order (LRN)
        post_res: list[TreeNode] = []
        tree_item._collect_postorder(tree_item._root, post_res)
        assert [n.key for n in post_res] == [20, 40, 30, 60, 80, 70, 50]

        # Level-order (BFS)
        lvl_res: list[TreeNode] = []
        tree_item._collect_levelorder(tree_item._root, lvl_res)
        assert [n.key for n in lvl_res] == [50, 30, 70, 20, 40, 60, 80]

    def test_bst_metrics_and_avl_balance(self, tree_item):
        """Verify node count, tree height, and AVL balance factor."""
        tree_item.load_preset("BST_BALANCED_SEARCH")
        assert tree_item.node_count == 7
        assert tree_item.tree_height == 3
        assert tree_item.is_avl_balanced is True

    def test_bst_degenerate_unbalanced(self, tree_item):
        """Degenerate BST must have height equal to node count and fail AVL balance."""
        tree_item.load_preset("BST_DEGENERATE_UNBALANCED")
        assert tree_item.node_count == 6
        assert tree_item.tree_height == 6
        assert tree_item.is_avl_balanced is False

    def test_bst_search_success_and_failure(self, tree_item):
        """Verify BST search path tracking and comparison count."""
        tree_item.load_preset("BST_BALANCED_SEARCH")

        # 1. Successful search for 40 (Path: 50 -> 30 -> 40)
        found, comps, path = tree_item.execute_bst_search(40)
        assert found is True
        assert comps == 3
        assert [n.key for n in path] == [50, 30, 40]

        # 2. Failed search for 99 (Path: 50 -> 70 -> 80)
        found2, comps2, path2 = tree_item.execute_bst_search(99)
        assert found2 is False
        assert comps2 == 3
        assert [n.key for n in path2] == [50, 70, 80]


class TestHuffmanCodingAndExpressionTree:
    """Verifies Huffman information theory, entropy, and arithmetic expression tree."""

    def test_huffman_prefix_free_property(self, tree_item):
        """Huffman codes must satisfy the prefix-free property: no code is prefix of another."""
        tree_item.load_preset("HUFFMAN_TEXT_COMPRESSION")
        leaves = [n for n in tree_item._all_nodes if n.is_leaf and n.code]
        assert len(leaves) >= 5

        codes = [n.code for n in leaves]
        for i, c1 in enumerate(codes):
            for j, c2 in enumerate(codes):
                if i != j:
                    assert not c1.startswith(c2), f"Code {c1} has prefix {c2}!"

    def test_huffman_shannon_entropy_and_efficiency(self, tree_item):
        """Shannon source coding theorem: L_avg >= H(X), Efficiency <= 100%."""
        tree_item.load_preset("HUFFMAN_DNA_GENOME")
        hm = tree_item.calculate_huffman_metrics()

        assert hm["entropy"] > 0.0
        assert hm["avg_len"] >= hm["entropy"] - 1e-4  # L_avg >= H(X)
        assert 0.0 < hm["efficiency"] <= 100.0
        assert hm["compression_ratio"] > 0.0  # Better than 8-bit ASCII

    def test_expression_tree_traversals(self, tree_item):
        """Expression tree in-order produces infix, post-order produces RPN/postfix."""
        tree_item.load_preset("ARITHMETIC_EXPRESSION")

        # In-order (Infix notation)
        in_res: list[TreeNode] = []
        tree_item._collect_inorder(tree_item._root, in_res)
        in_tokens = [n.val for n in in_res]
        assert in_tokens == ["A", "+", "B", "×", "C", "−", "D", "÷", "E"]

        # Post-order (Reverse Polish Notation / Postfix)
        post_res: list[TreeNode] = []
        tree_item._collect_postorder(tree_item._root, post_res)
        post_tokens = [n.val for n in post_res]
        assert post_tokens == ["A", "B", "+", "C", "D", "−", "×", "E", "÷"]


class TestInteractionAndAnimation:
    """Verifies animation step, touchless air pinch, and UI responsiveness."""

    def test_traversal_animation_lifecycle(self, tree_item):
        """Test starting and stepping through traversal animation."""
        tree_item.load_preset("BST_BALANCED_SEARCH")
        tree_item.start_traversal("IN_ORDER")
        assert tree_item._is_animating is True
        assert tree_item._traversal_index == 0

        # Step
        tree_item._on_traversal_step()
        assert tree_item._traversal_index == 1

        tree_item.cleanup()
        assert tree_item._is_animating is False

    def test_air_pinch_presets(self, tree_item):
        """Touchless air pinch on preset button loads preset."""
        preset_r = tree_item._btn_rects["preset_HUFFMAN_TEXT_COMPRESSION"]
        pinch_pt = tree_item.mapToScene(preset_r.center())

        handled = tree_item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert tree_item._active_preset_code == "HUFFMAN_TEXT_COMPRESSION"

    def test_air_pinch_traversal_buttons(self, tree_item):
        """Touchless air pinch on traversal triggers animation."""
        btn_r = tree_item._btn_rects["btn_preorder"]
        pinch_pt = tree_item.mapToScene(btn_r.center())

        handled = tree_item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert tree_item._active_traversal_mode == "PRE_ORDER"
        tree_item.cleanup()

    def test_air_pinch_search_and_close(self, tree_item):
        """Air pinch triggers search and close."""
        btn_search = tree_item._btn_rects["btn_search"]
        handled_s = tree_item.handle_air_pinch(tree_item.mapToScene(btn_search.center()))
        assert handled_s is True
        assert tree_item._search_comparisons > 0

        close_r = tree_item._btn_rects["close"]
        tree_item.setVisible(True)
        handled_c = tree_item.handle_air_pinch(tree_item.mapToScene(close_r.center()))
        assert handled_c is True
        assert tree_item.isVisible() is False


class TestDiagramAndBoardIntegration:
    """Verifies vector diagram stamping, LessonBoard lifecycle, and MainWindow slots."""

    def test_informatics_tools_create_binary_tree_diagram(self):
        """Verify InformaticsTools.create_binary_tree_diagram generates styled vector items."""
        items = InformaticsTools.create_binary_tree_diagram(
            center=(400.0, 300.0),
            preset_code="BST_BALANCED_SEARCH",
            traversal_mode="IN_ORDER",
        )
        assert len(items) >= 12, "Diagram must generate background, nodes, edges, and analytics"

        # Check background rects
        bg_items = [it for it in items if isinstance(it, QGraphicsRectItem)]
        assert len(bg_items) >= 3

        # Check text items
        text_items = [it for it in items if isinstance(it, QGraphicsTextItem)]
        all_text = " ".join(t.toPlainText() for t in text_items)
        assert "CÂY NHỊ PHÂN" in all_text
        assert "Số nút:" in all_text
        assert "Chiều cao" in all_text

    def test_stamp_vector_diagram_to_board(self, board, tree_item):
        """Verify stamping vector diagram to LessonBoard."""
        board.addItem(tree_item)
        init_items = len(board.items())

        tree_item.stamp_to_board()
        new_items = len(board.items())
        assert new_items > init_items + 10

    def test_lesson_board_lifecycle(self, board):
        """Verify LessonBoard show/toggle/get methods for binary tree."""
        assert board.is_binary_tree_visible() is False
        assert board.get_binary_tree() is None

        # Show
        item = board.show_binary_tree(True)
        assert item is not None
        assert board.is_binary_tree_visible() is True
        assert board.get_binary_tree() == item
        assert board.get_binary_tree_item() == item

        # Toggle to hide
        assert board.toggle_binary_tree() is False
        assert board.is_binary_tree_visible() is False

        # Toggle to show
        assert board.toggle_binary_tree() is True
        assert board.is_binary_tree_visible() is True

    def test_main_window_toggle_and_action(self, qapp):
        """Verify MainWindow slot and UI trigger for binary tree."""
        win = MainWindow(camera_index=-1)
        win.show()

        try:
            assert win._on_toggle_binary_tree() is True
            assert win._scene.is_binary_tree_visible() is True

            assert win._on_toggle_binary_tree() is False
            assert win._scene.is_binary_tree_visible() is False
        finally:
            win.close()
