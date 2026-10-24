"""
AirTeach AI — Tests for Interactive Multi-Layer Perceptron (MLP) & Backpropagation Simulator (Milestone 109)
Verifies forward propagation, backpropagation gradient descent, non-linear activations,
2D decision boundary probability contours, touchless air gestures, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import pytest
import numpy as np
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_mlp_backprop import (
    InteractiveMLPBackpropItem,
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


def test_mlp_initialization(qapp):
    item = InteractiveMLPBackpropItem(pos=(120.0, 50.0))
    try:
        assert item.pos().x() == 120.0
        assert item.pos().y() == 50.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 680.0, 410.0)
        assert item.active_preset.code == "XOR_PROBLEM"
        assert item.epoch == 0
        assert item.loss > 0.0
        assert 0.0 <= item.accuracy <= 1.0
        assert item.learning_rate == 0.25
        assert item.activation_name == "Tanh"
        assert item.is_training is False
        assert item._w1.shape == (2, 6)
        assert item._w2.shape == (6, 1)
        assert item._grid_z.shape == (40, 40)
    finally:
        item.cleanup()


def test_mlp_all_presets(qapp):
    item = InteractiveMLPBackpropItem()
    try:
        for p_code in PRESETS:
            item.load_preset(p_code)
            assert item.active_preset.code == p_code
            assert item._X.shape[1] == 2
            assert item._Y.shape[1] == 1
            assert item.epoch == 0
            assert item.loss > 0.0
            assert len(item._loss_history) == 1
            assert item._grid_z.shape == (40, 40)
    finally:
        item.cleanup()


def test_activation_functions_and_derivatives(qapp):
    item = InteractiveMLPBackpropItem()
    try:
        # Tanh
        item._activation_name = "Tanh"
        z = np.array([0.0], dtype=np.float32)
        assert abs(float(item._activate(z)[0]) - 0.0) < 1e-5
        assert abs(float(item._activate_derivative(z)[0]) - 1.0) < 1e-5

        # ReLU
        item._activation_name = "ReLU"
        z_neg = np.array([-3.0], dtype=np.float32)
        z_pos = np.array([3.0], dtype=np.float32)
        assert float(item._activate(z_neg)[0]) == 0.0
        assert float(item._activate(z_pos)[0]) == 3.0
        assert float(item._activate_derivative(z_neg)[0]) == 0.0
        assert float(item._activate_derivative(z_pos)[0]) == 1.0

        # Sigmoid
        item._activation_name = "Sigmoid"
        assert abs(float(item._activate(z)[0]) - 0.5) < 1e-5
        a_half = np.array([0.5], dtype=np.float32)
        assert abs(float(item._activate_derivative(a_half)[0]) - 0.25) < 1e-5

        # Cycling
        assert item.toggle_activation() == "Tanh"
        assert item.toggle_activation() == "ReLU"
        assert item.toggle_activation() == "Sigmoid"
    finally:
        item.cleanup()


def test_forward_and_backpropagation_training(qapp):
    item = InteractiveMLPBackpropItem()
    try:
        item.load_preset("LINEAR_SEPARABLE")
        w1_init = item._w1.copy()
        w2_init = item._w2.copy()

        # Step 20 epochs
        stepped_signals = []
        item.training_stepped.connect(lambda ep, ls, acc: stepped_signals.append((ep, ls, acc)))
        item.train_epochs(20)

        assert item.epoch == 20
        assert len(stepped_signals) == 1
        assert stepped_signals[0][0] == 20
        # Weights must be updated by gradient descent
        assert not np.allclose(item._w1, w1_init)
        assert not np.allclose(item._w2, w2_init)
    finally:
        item.cleanup()


def test_learning_rate_and_clamping(qapp):
    item = InteractiveMLPBackpropItem()
    try:
        item.set_learning_rate(0.6)
        assert item.learning_rate == 0.6

        # Upper clamp [0.01, 1.5]
        item.set_learning_rate(3.0)
        assert item.learning_rate == 1.5

        # Lower clamp
        item.set_learning_rate(-0.2)
        assert item.learning_rate == 0.01
    finally:
        item.cleanup()


def test_training_timer_toggle(qapp):
    item = InteractiveMLPBackpropItem()
    try:
        assert item.is_training is False
        assert item.toggle_training() is True
        assert item.is_training is True

        # Callback tick
        ep_before = item.epoch
        item._on_training_tick()
        assert item.epoch > ep_before

        assert item.toggle_training() is False
        assert item.is_training is False
    finally:
        item.cleanup()


def test_air_pinch_interactions(qapp):
    item = InteractiveMLPBackpropItem(pos=(100.0, 50.0))
    try:
        # Outside
        assert item.handle_air_pinch(QPointF(10.0, 10.0)) is False

        # Preset button click
        btn_moon = item._btn_rects["preset_TWO_MOONS"]
        scene_pt_moon = item.mapToScene(btn_moon.center())
        assert item.handle_air_pinch(scene_pt_moon) is True
        assert item.active_preset.code == "TWO_MOONS"

        # Step button click
        ep_before = item.epoch
        btn_step = item._btn_rects["step_train"]
        scene_pt_step = item.mapToScene(btn_step.center())
        assert item.handle_air_pinch(scene_pt_step) is True
        assert item.epoch == ep_before + 10

        # LR Up / Down
        lr_before = item.learning_rate
        btn_lru = item._btn_rects["lr_up"]
        scene_pt_lru = item.mapToScene(btn_lru.center())
        assert item.handle_air_pinch(scene_pt_lru) is True
        assert item.learning_rate > lr_before

        btn_lrd = item._btn_rects["lr_down"]
        scene_pt_lrd = item.mapToScene(btn_lrd.center())
        assert item.handle_air_pinch(scene_pt_lrd) is True
        assert abs(item.learning_rate - lr_before) < 1e-4

        # Toggle activation
        act_before = item.activation_name
        btn_act = item._btn_rects["toggle_activation"]
        scene_pt_act = item.mapToScene(btn_act.center())
        assert item.handle_air_pinch(scene_pt_act) is True
        assert item.activation_name != act_before

        # Reset button
        item.train_epochs(15)
        assert item.epoch > 0
        btn_rst = item._btn_rects["reset_weights"]
        scene_pt_rst = item.mapToScene(btn_rst.center())
        assert item.handle_air_pinch(scene_pt_rst) is True
        assert item.epoch == 0

        # Train toggle
        btn_trn = item._btn_rects["toggle_train"]
        scene_pt_trn = item.mapToScene(btn_trn.center())
        assert item.handle_air_pinch(scene_pt_trn) is True
        assert item.is_training is True
        assert item.handle_air_pinch(scene_pt_trn) is True
        assert item.is_training is False
    finally:
        item.cleanup()


def test_stem_tools_diagram_generation_and_stamp(qapp):
    items = InformaticsTools.create_mlp_backprop_diagram(
        center=(400.0, 300.0),
        preset_code="XOR_PROBLEM",
        epoch=50,
        loss=0.082,
        accuracy=1.0,
        activation="Tanh",
        lr=0.25,
    )
    assert isinstance(items, list)
    assert len(items) >= 15

    all_texts = []
    for itm in items:
        if hasattr(itm, "toPlainText"):
            all_texts.append(itm.toPlainText())
    combined = " ".join(all_texts)
    assert "MẠNG NƠ-RON MLP" in combined or "BACKPROPAGATION" in combined
    assert "Epoch" in combined or "Loss" in combined or "Gradient" in combined


def test_stamping_to_scene(qapp):
    board = LessonBoard()
    item = InteractiveMLPBackpropItem(pos=(100.0, 50.0))
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
        assert board.is_mlp_backprop_visible() is False
        assert board.get_mlp_backprop() is None

        # Show
        shown_item = board.show_mlp_backprop(True)
        assert shown_item is not None
        assert board.is_mlp_backprop_visible() is True
        assert board.get_mlp_backprop() is shown_item
        assert board.get_mlp_backprop_item() is shown_item

        # Toggle off
        vis = board.toggle_mlp_backprop()
        assert vis is False
        assert board.is_mlp_backprop_visible() is False
        assert board.get_mlp_backprop() is None

        # Toggle on
        vis = board.toggle_mlp_backprop()
        assert vis is True
        assert board.is_mlp_backprop_visible() is True

        # Hide
        board.show_mlp_backprop(False)
        assert board.is_mlp_backprop_visible() is False
    finally:
        if board.get_mlp_backprop_item():
            board.get_mlp_backprop_item().cleanup()
        board.clear()


def test_paint_headless(qapp):
    item = InteractiveMLPBackpropItem()
    try:
        img = QImage(680, 410, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()
        assert not img.isNull()
    finally:
        item.cleanup()
