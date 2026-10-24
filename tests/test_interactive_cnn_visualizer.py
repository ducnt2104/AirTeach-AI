"""
Unit test suite for InteractiveCNNVisualizerItem (Milestone 104).
Covers 2D discrete convolution mathematics, kernel presets, activation functions,
max/average pooling, sliding receptive field, touchless air gestures, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_cnn_visualizer import (
    InteractiveCNNVisualizerItem,
    PRESETS,
    CNNPreset,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import InformaticsTools


class TestInteractiveCNNVisualizer:

    def test_initialization(self, qapp):
        """Verifies default properties, dimensions, and flags of InteractiveCNNVisualizerItem."""
        item = InteractiveCNNVisualizerItem(pos=(200.0, 120.0))
        assert item.pos() == QPointF(200.0, 120.0)
        assert item.WIDTH == 640.0
        assert item.HEIGHT == 380.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 640.0, 380.0)
        assert item.active_preset.name == "SOBEL_VERTICAL_EDGES"
        assert item.activation == "RELU"
        assert item.pooling == "MAX_POOLING"
        assert item.window_position == (1, 1)
        assert item.is_animating is False
        item.cleanup()

    def test_all_five_presets(self, qapp):
        """Verifies that all 5 CNN presets load cleanly and recompute feature maps."""
        item = InteractiveCNNVisualizerItem()
        preset_keys = [
            "SOBEL_VERTICAL_EDGES",
            "SOBEL_HORIZONTAL_EDGES",
            "RIDGE_LAPLACIAN_CONTOUR",
            "GAUSSIAN_BLUR_SMOOTHING",
            "HIGH_PASS_SHARPENING",
        ]
        for key in preset_keys:
            item.load_preset(key)
            assert item.active_preset.name == key
            assert len(item._kernel) == 3
            assert len(item._kernel[0]) == 3
            assert len(item._feature_map) == 8
            assert len(item._feature_map[0]) == 8
            assert len(item._pooled_map) == 4
            assert len(item._pooled_map[0]) == 4
        item.cleanup()

    def test_2d_discrete_convolution_math(self, qapp):
        """Verifies mathematical correctness of 2D discrete convolution sum."""
        item = InteractiveCNNVisualizerItem()
        item.load_preset("SOBEL_VERTICAL_EDGES")
        # Custom simple 8x8 grid: all 100
        item._input_grid = [[100.0 for _ in range(8)] for _ in range(8)]
        # For Sobel-X on uniform image, derivative is zero!
        raw_val = item.compute_single_cell_convolution(3, 3)
        assert abs(raw_val) < 1e-4

        # Custom step edge: left half 0, right half 200
        # Columns 0..3 = 0, Columns 4..7 = 200
        for r in range(8):
            for c in range(8):
                item._input_grid[r][c] = 200.0 if c >= 4 else 0.0

        # At (3, 3), neighborhood columns are c=2 (0), c=3 (0), c=4 (200)
        # Kernel Sobel-X: col 0 has weight -1,-2,-1; col 1 has 0; col 2 has +1,+2,+1 (sum = +4)
        # S(3, 3) = 0 * (-4) + 0 * 0 + 200 * (+4) = +800
        conv_val = item.compute_single_cell_convolution(3, 3)
        assert abs(conv_val - 800.0) < 1e-4
        item.cleanup()

    def test_activation_functions(self, qapp):
        """Verifies ReLU, Leaky ReLU, Linear, and Sigmoid activations."""
        item = InteractiveCNNVisualizerItem()

        # ReLU
        item.set_activation("RELU")
        assert item.apply_activation(50.0) == 50.0
        assert item.apply_activation(-50.0) == 0.0

        # Leaky ReLU
        item.set_activation("LEAKY_RELU")
        assert item.apply_activation(50.0) == 50.0
        assert abs(item.apply_activation(-50.0) - (-5.0)) < 1e-4

        # Linear
        item.set_activation("LINEAR")
        assert item.apply_activation(120.0) == 120.0
        assert item.apply_activation(-10.0) == 0.0
        assert item.apply_activation(300.0) == 255.0

        # Sigmoid (scaled to [0, 255])
        item.set_activation("SIGMOID")
        sig_0 = item.apply_activation(0.0)
        assert abs(sig_0 - 127.5) < 1.0
        assert 0.0 <= item.apply_activation(-100.0) < 127.5
        assert 127.5 < item.apply_activation(100.0) <= 255.0
        item.cleanup()

    def test_pooling_operations(self, qapp):
        """Verifies Max Pooling and Average Pooling downsampling."""
        item = InteractiveCNNVisualizerItem()
        # Mock feature map top-left 2x2 block
        item._feature_map[0][0] = 10.0
        item._feature_map[0][1] = 20.0
        item._feature_map[1][0] = 30.0
        item._feature_map[1][1] = 40.0

        # Max Pooling
        item.set_pooling("MAX_POOLING")
        assert item._pooled_map[0][0] == 40.0

        # Average Pooling
        item.set_pooling("AVG_POOLING")
        assert item._pooled_map[0][0] == (10.0 + 20.0 + 30.0 + 40.0) / 4.0
        item.cleanup()

    def test_sliding_window_stepping_and_reset(self, qapp):
        """Verifies raster scan stepping and resetting of receptive field."""
        item = InteractiveCNNVisualizerItem()
        item.set_window_position(0, 0)
        assert item.window_position == (0, 0)

        item.step_forward()
        assert item.window_position == (0, 1)

        # Step to end of row
        for _ in range(7):
            item.step_forward()
        # Should wrap to (1, 0)
        assert item.window_position == (1, 0)

        item.reset_window()
        assert item.window_position == (0, 0)
        item.cleanup()

    def test_animation_toggle(self, qapp):
        """Verifies automatic sliding window animation toggle."""
        item = InteractiveCNNVisualizerItem()
        assert item.is_animating is False

        is_now = item.toggle_animation()
        assert is_now is True
        assert item.is_animating is True

        # Simulate animation step
        r0, c0 = item.window_position
        item._on_animation_step()
        assert item.window_position != (r0, c0)

        is_now_off = item.toggle_animation()
        assert is_now_off is False
        assert item.is_animating is False
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        """Verifies touchless air pinch interaction on buttons and grids."""
        item = InteractiveCNNVisualizerItem(pos=(100.0, 100.0))

        # Air pinch outside
        assert item.handle_air_pinch(QPointF(50.0, 50.0)) is False

        # Air pinch on Sobel-Y preset button
        r_sobel_y = item._btn_rects.get("preset_SOBEL_HORIZONTAL_EDGES")
        assert r_sobel_y is not None
        item.handle_air_pinch(item.mapToScene(r_sobel_y.center()))
        assert item.active_preset.name == "SOBEL_HORIZONTAL_EDGES"

        # Air pinch on LeakyReLU activation button
        r_leaky = item._btn_rects.get("act_LEAKY_RELU")
        assert r_leaky is not None
        item.handle_air_pinch(item.mapToScene(r_leaky.center()))
        assert item.activation == "LEAKY_RELU"

        # Air pinch on Avg Pooling button
        r_avg = item._btn_rects.get("pool_AVG_POOLING")
        assert r_avg is not None
        item.handle_air_pinch(item.mapToScene(r_avg.center()))
        assert item.pooling == "AVG_POOLING"

        # Air pinch on Input Grid cell
        r_grid = item._btn_rects.get("input_grid_bounds")
        assert r_grid is not None
        target_pt = QPointF(r_grid.x() + 16.0 * 4 + 8.0, r_grid.y() + 16.0 * 5 + 8.0)
        item.handle_air_pinch(item.mapToScene(target_pt))
        assert item.window_position == (5, 4)

        # Air pinch on Step and Reset
        r_step = item._btn_rects.get("step_forward")
        item.handle_air_pinch(item.mapToScene(r_step.center()))
        assert item.window_position == (5, 5)

        r_reset = item._btn_rects.get("reset_window")
        item.handle_air_pinch(item.mapToScene(r_reset.center()))
        assert item.window_position == (0, 0)
        item.cleanup()

    def test_stem_tools_diagram_generation_and_stamp(self, qapp):
        """Verifies InformaticsTools.create_cnn_diagram and stamping to LessonBoard."""
        board = LessonBoard()
        item = InteractiveCNNVisualizerItem(pos=QPointF(100.0, 100.0))
        board.addItem(item)
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count

        # Test direct diagram creation via InformaticsTools
        items = InformaticsTools.create_cnn_diagram(
            center=(400.0, 300.0),
            kernel_name="Sobel-X",
            activation="RELU",
            pooling="MAX_POOLING",
            window_r=2,
            window_c=2,
        )
        assert len(items) >= 12
        text_contents = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("MẠNG NƠ-RON CNN" in t for t in text_contents)
        assert any("TÍCH CHẬP 2D" in t for t in text_contents)
        item.cleanup()

    def test_lesson_board_lifecycle_and_lasso_exclusion(self, qapp):
        """Verifies LessonBoard show/toggle/getter lifecycle and lasso exclusion."""
        board = LessonBoard()

        # Initially not visible
        assert board.is_cnn_visualizer_visible() is False
        assert board.get_cnn_visualizer() is None

        # Show
        item = board.show_cnn_visualizer(True, pos=(200.0, 150.0))
        assert item is not None
        assert board.is_cnn_visualizer_visible() is True
        assert board.get_cnn_visualizer() is item
        assert board.get_cnn_visualizer_item() is item
        assert item.pos() == QPointF(200.0, 150.0)

        # Toggle hide
        new_state = board.toggle_cnn_visualizer()
        assert new_state is False
        assert board.is_cnn_visualizer_visible() is False
        assert board.get_cnn_visualizer() is None

        # Toggle show
        new_state_2 = board.toggle_cnn_visualizer()
        assert new_state_2 is True
        assert board.is_cnn_visualizer_visible() is True

        # Verify exclusion in lasso selection
        rect = QRectF(100.0, 100.0, 500.0, 500.0)
        items_in_rect = board.get_items_in_rect(rect)
        assert item not in items_in_rect
        item.cleanup()

    def test_main_window_toggle_slot(self, qapp):
        """Verifies MainWindow slot _on_toggle_cnn_visualizer properly delegates to LessonBoard."""
        from src.ui.main_window import MainWindow
        win = MainWindow()
        assert win._scene.is_cnn_visualizer_visible() is False

        # Toggle via slot
        res = win._on_toggle_cnn_visualizer()
        assert res is True
        assert win._scene.is_cnn_visualizer_visible() is True

        # Toggle off
        res_off = win._on_toggle_cnn_visualizer()
        assert res_off is False
        assert win._scene.is_cnn_visualizer_visible() is False
        win.close()
