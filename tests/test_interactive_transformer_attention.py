"""
Unit Tests for Interactive Transformer Multi-Head Self-Attention Visualizer Suite
================================================================================
Covers:
1. Scaled dot-product attention mathematics: Attention(Q, K, V) = Softmax(Q K^T / (τ * sqrt(d_k)) + M) V.
2. Softmax normalization (all row vectors sum to 1.0).
3. Autoregressive causal masking (GPT mode: upper-triangular strictly zero).
4. Multi-head attention projections (4 heads + multi-head average).
5. 5 Curriculum Presets (Pronoun Coreference, Vietnamese Syntax, Parallel Clause, Math Logic, Code AST).
6. Softmax temperature scaling (τ in [0.2, 5.0]) and Shannon attention entropy metrics.
7. Token selection and signals (preset_changed, head_changed, token_selected).
8. Dual view modes: Heatmap matrix view vs Bipartite token ribbon view.
9. QPainter high-DPI rendering without exceptions.
10. Canvas stamping and JSON serialization roundtrip (.airteach packaging).
"""
import math
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent, QStyleOptionGraphicsItem

from src.ui.interactive_transformer_attention import (
    HEAD_COLORS,
    PRESETS,
    InteractiveTransformerAttentionItem,
    TransformerPreset,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_initialization(qapp):
    """Test default parameters, preset, dimensions, and flags."""
    item = InteractiveTransformerAttentionItem(pos=(150.0, 80.0))
    assert item.x() == 150.0
    assert item.y() == 80.0
    assert item._active_preset_code == "PRONOUN_COREFERENCE"
    assert item._num_heads == 4
    assert item._active_head == 0
    assert item._is_causal is False
    assert item._temperature == 1.0
    assert item._view_mode == "heatmap"
    assert len(item._attn_matrices) == 4

    tokens = item._preset.tokens
    T = len(tokens)
    assert T == 11
    assert item.boundingRect().width() == InteractiveTransformerAttentionItem.WIDTH
    assert item.boundingRect().height() == InteractiveTransformerAttentionItem.HEIGHT


def test_scaled_dot_product_math_and_softmax_distribution(qapp):
    """Verify authentic scaled dot-product attention mathematics and row-wise softmax normalization."""
    item = InteractiveTransformerAttentionItem()
    tokens = item._preset.tokens
    T = len(tokens)

    # All 4 heads must have shape (T, T) and valid probabilities
    for h in range(4):
        mat = item._attn_matrices[h]
        assert mat.shape == (T, T)
        # Probabilities bounded in [0, 1]
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0)
        # Each row must sum to 1.0
        row_sums = np.sum(mat, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    # Active attention matrix for head 0
    mat0 = item.active_attention_matrix()
    assert np.array_equal(mat0, item._attn_matrices[0])

    # Multi-head average
    item.set_head(-1)
    avg_mat = item.active_attention_matrix()
    expected_avg = np.mean(item._attn_matrices, axis=0)
    assert np.allclose(avg_mat, expected_avg, atol=1e-6)
    assert np.allclose(np.sum(avg_mat, axis=-1), 1.0, atol=1e-5)


def test_causal_masking_autoregressive_gpt(qapp):
    """Verify causal decoder masking sets upper-triangular attention strictly to 0."""
    item = InteractiveTransformerAttentionItem()
    assert item._is_causal is False

    is_causal = item.toggle_causal_mask()
    assert is_causal is True
    assert item._is_causal is True

    tokens = item._preset.tokens
    T = len(tokens)

    for h in range(4):
        mat = item._attn_matrices[h]
        # In causal mode, token i cannot attend to any token j > i
        for i in range(T):
            for j in range(i + 1, T):
                assert mat[i, j] < 1e-6, f"Head {h}: Position ({i}, {j}) should be masked to 0, got {mat[i, j]}"
        # Valid probability rows
        row_sums = np.sum(mat, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    # The very first token can only attend to itself
    for h in range(4):
        mat = item._attn_matrices[h]
        assert abs(mat[0, 0] - 1.0) < 1e-5


def test_curriculum_presets(qapp):
    """Test loading all 5 curriculum presets and signal emission."""
    item = InteractiveTransformerAttentionItem()
    emitted_presets = []
    item.preset_changed.connect(emitted_presets.append)

    for code in ("VIETNAMESE_SYNTAX", "PARALLEL_CLAUSE", "ARITHMETIC_EXPR", "CODE_FUNCTION", "PRONOUN_COREFERENCE"):
        item.load_preset(code)
        assert item._active_preset_code == code
        preset = PRESETS[code]
        assert item._preset == preset
        T = len(preset.tokens)
        assert len(item._attn_matrices) == 4
        assert item._attn_matrices[0].shape == (T, T)
        assert np.allclose(np.sum(item._attn_matrices[0], axis=-1), 1.0, atol=1e-5)

    assert emitted_presets == [
        "VIETNAMESE_SYNTAX",
        "PARALLEL_CLAUSE",
        "ARITHMETIC_EXPR",
        "CODE_FUNCTION",
        "PRONOUN_COREFERENCE",
    ]


def test_temperature_scaling_and_entropy(qapp):
    """Verify softmax temperature scaling (low temp = focused/lower entropy, high temp = diffuse/higher entropy)."""
    item = InteractiveTransformerAttentionItem()
    row_idx = 7  # 'it' in pronoun coreference

    # Low temperature (sharp / focused attention)
    item.set_temperature(0.5)
    entropy_sharp = item.compute_entropy(row_idx)

    # Standard temperature
    item.set_temperature(1.0)
    entropy_norm = item.compute_entropy(row_idx)

    # High temperature (diffuse attention)
    item.set_temperature(2.0)
    entropy_diff = item.compute_entropy(row_idx)

    assert entropy_sharp < entropy_norm < entropy_diff
    assert entropy_sharp >= 0.0

    # Bounds clamping [0.2, 5.0]
    item.set_temperature(0.05)
    assert item._temperature == 0.2

    item.set_temperature(10.0)
    assert item._temperature == 5.0


def test_head_selection_and_signals(qapp):
    """Test changing active attention head and head_changed signal."""
    item = InteractiveTransformerAttentionItem()
    heads_emitted = []
    item.head_changed.connect(heads_emitted.append)

    item.set_head(1)
    assert item._active_head == 1
    item.set_head(3)
    assert item._active_head == 3
    item.set_head(-1)  # Average
    assert item._active_head == -1

    # Out of range heads should be ignored
    item.set_head(10)
    assert item._active_head == -1
    item.set_head(-5)
    assert item._active_head == -1

    assert heads_emitted == [1, 3, -1]


def test_token_selection(qapp):
    """Test token selection and token_selected signal."""
    item = InteractiveTransformerAttentionItem()
    selected_tokens = []
    item.token_selected.connect(lambda idx, tok: selected_tokens.append((idx, tok)))

    item.select_token(1)
    assert item._selected_token_idx == 1
    assert selected_tokens[-1] == (1, "animal")

    item.select_token(7)
    assert item._selected_token_idx == 7
    assert selected_tokens[-1] == (7, "it")

    # Out of bounds ignored
    item.select_token(99)
    assert item._selected_token_idx == 7


def test_view_mode_toggle(qapp):
    """Test toggling between heatmap view and ribbon arc view."""
    item = InteractiveTransformerAttentionItem()
    assert item._view_mode == "heatmap"

    mode = item.toggle_view_mode()
    assert mode == "ribbon"
    assert item._view_mode == "ribbon"

    mode = item.toggle_view_mode()
    assert mode == "heatmap"
    assert item._view_mode == "heatmap"


def test_mouse_button_clicks(qapp):
    """Test mouse clicking on control buttons."""
    item = InteractiveTransformerAttentionItem()

    # Helper to create a QGraphicsSceneMouseEvent
    scene = QGraphicsScene()
    scene.addItem(item)

    # Click on head_2 button
    h2_rect = item._btn_rects["head_2"]
    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(h2_rect.center())
    item.mousePressEvent(event)
    assert item._active_head == 2

    # Click on toggle_causal button
    causal_rect = item._btn_rects["toggle_causal"]
    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(causal_rect.center())
    item.mousePressEvent(event)
    assert item._is_causal is True

    # Click on toggle_view button
    view_rect = item._btn_rects["toggle_view"]
    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(view_rect.center())
    item.mousePressEvent(event)
    assert item._view_mode == "ribbon"

    # Click on temp_sharp button
    t_rect = item._btn_rects["temp_sharp"]
    event = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
    event.setPos(t_rect.center())
    item.mousePressEvent(event)
    assert item._temperature == 0.5


def test_qpainter_rendering(qapp):
    """Test QPainter rendering in both heatmap and ribbon modes without crashes."""
    item = InteractiveTransformerAttentionItem()

    # Heatmap mode render
    img = QImage(800, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    opt = QStyleOptionGraphicsItem()
    item.paint(painter, opt)
    painter.end()

    # Ribbon mode render
    item.toggle_view_mode()
    painter2 = QPainter(img)
    item.paint(painter2, opt)
    painter2.end()


def test_canvas_stamping(qapp):
    """Test stamping current attention item onto a canvas."""
    scene = QGraphicsScene()
    item = InteractiveTransformerAttentionItem(pos=(100.0, 100.0))
    item.load_preset("VIETNAMESE_SYNTAX")
    item.set_head(2)
    item.set_temperature(1.5)

    stamped = item.stamp_to_canvas(scene)
    assert len(stamped) == 1
    clone = stamped[0]
    assert isinstance(clone, InteractiveTransformerAttentionItem)
    assert clone._active_preset_code == "VIETNAMESE_SYNTAX"
    assert clone._active_head == 2
    assert clone._temperature == 1.5


def test_serialization_roundtrip(qapp):
    """Test to_dict and from_dict state preservation."""
    item = InteractiveTransformerAttentionItem(pos=(220.0, 180.0))
    item.load_preset("CODE_FUNCTION")
    item.set_head(1)
    item.toggle_causal_mask()
    item.toggle_view_mode()
    item.set_temperature(1.8)
    item.select_token(3)

    data = item.to_dict()
    assert data["type"] == "interactive_transformer_attention"
    assert data["pos"] == [220.0, 180.0]
    assert data["preset_code"] == "CODE_FUNCTION"
    assert data["active_head"] == 1
    assert data["is_causal"] is True
    assert data["temperature"] == 1.8
    assert data["view_mode"] == "ribbon"
    assert data["selected_token_idx"] == 3

    restored = InteractiveTransformerAttentionItem.from_dict(data)
    assert restored.x() == 220.0
    assert restored.y() == 180.0
    assert restored._active_preset_code == "CODE_FUNCTION"
    assert restored._active_head == 1
    assert restored._is_causal is True
    assert restored._temperature == 1.8
    assert restored._view_mode == "ribbon"
    assert restored._selected_token_idx == 3


def test_handle_air_tap(qapp):
    """Test touchless air gesture tapping on transformer visualizer buttons."""
    item = InteractiveTransformerAttentionItem(pos=(100.0, 100.0))

    # 1. Air tap outside card
    consumed = item.handle_air_tap(QPointF(0.0, 0.0))
    assert consumed is False

    # 2. Air tap on head_3 button
    h3_rect = item._btn_rects["head_3"]
    scene_pos = item.mapToScene(h3_rect.center())
    consumed = item.handle_air_tap(scene_pos)
    assert consumed is True
    assert item._active_head == 3

    # 3. Air tap on toggle causal button
    causal_rect = item._btn_rects["toggle_causal"]
    scene_pos = item.mapToScene(causal_rect.center())
    consumed = item.handle_air_tap(scene_pos)
    assert consumed is True
    assert item._is_causal is True

    # 4. Air tap on temp_diff button
    td_rect = item._btn_rects["temp_diff"]
    scene_pos = item.mapToScene(td_rect.center())
    consumed = item.handle_air_tap(scene_pos)
    assert consumed is True
    assert item._temperature == 2.0


def test_lesson_board_integration(qapp):
    """Test LessonBoard lifecycle methods for Transformer attention visualizer."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert board.is_transformer_attention_visible() is False
    assert board.get_transformer_attention_item() is None

    # Show item
    item = board.show_transformer_attention(True, pos=(200.0, 150.0))
    assert item is not None
    assert board.is_transformer_attention_visible() is True
    assert board.get_transformer_attention_item() is item
    assert item.pos() == QPointF(200.0, 150.0)

    # Toggle visibility
    is_vis = board.toggle_transformer_attention()
    assert is_vis is False
    assert board.is_transformer_attention_visible() is False

    is_vis = board.toggle_transformer_attention()
    assert is_vis is True
    assert board.is_transformer_attention_visible() is True


def test_main_window_transformer_toggle(qapp):
    """Test MainWindow toggle action and slot for Transformer visualizer."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_toggle_transformer_attention")
        assert win._scene.is_transformer_attention_visible() is False

        # Toggle on
        is_vis = win._on_toggle_transformer_attention()
        assert is_vis is True
        assert win._scene.is_transformer_attention_visible() is True

        # Toggle off
        is_vis = win._on_toggle_transformer_attention()
        assert is_vis is False
        assert win._scene.is_transformer_attention_visible() is False
    finally:
        win.close()


