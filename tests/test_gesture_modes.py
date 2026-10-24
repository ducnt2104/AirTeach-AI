"""
Tests for AirTeach AI Dual Gesture Modes & Tracking Safety
==========================================================
Verifies:
  - Mode A (Air Pen / INDEX_UP): drawing with extended index finger
  - Mode B (Precision Pinch): pinch-to-draw, release-to-point
  - Pinch threshold configurability
  - Tracking gap safety: stops cleanly after 2 lost frames, no bridging
"""
import pytest
from src.core.hand_gesture_engine import DrawMode, GestureState, HandGestureFSM
from tests.test_core_algorithms import _make_synthetic_landmarks


def feed_frames(fsm: HandGestureFSM, landmarks: list, n: int = 5, score: float = 0.95) -> GestureState:
    state = GestureState.NONE
    for i in range(n):
        state, _, _ = fsm.process_landmarks(landmarks, score, float(i * 16))
    return state


def test_dual_mode_air_pen_vs_pinch():
    fsm = HandGestureFSM()

    # Landmark where index is extended, thumb is tucked near index PIP (~0.05 dist)
    lm_pinched = _make_synthetic_landmarks(
        index_tip_y=0.15,
        index_pip_y=0.40,
        thumb_tip_xy=(0.51, 0.38),
        middle_tip_y=0.60,
        middle_pip_y=0.45,
        ring_tip_y=0.70,
        ring_pip_y=0.50,
        pinky_tip_y=0.72,
        pinky_pip_y=0.52,
    )

    # 1. In default Air Pen mode (INDEX_UP), tucked/relaxed thumb triggers DRAW
    fsm.set_draw_mode(DrawMode.INDEX_UP)
    assert fsm.draw_mode == DrawMode.INDEX_UP
    state = feed_frames(fsm, lm_pinched, n=5)
    assert state == GestureState.DRAW

    # 2. In Precision Pinch mode:
    # When thumb is pinched to index tip, it should be DRAW
    lm_true_pinch = _make_synthetic_landmarks(
        index_tip_y=0.15,
        index_pip_y=0.40,
        thumb_tip_xy=(0.50, 0.16),  # right at the index tip!
        middle_tip_y=0.60,
        middle_pip_y=0.45,
        ring_tip_y=0.70,
        ring_pip_y=0.50,
        pinky_tip_y=0.72,
        pinky_pip_y=0.52,
    )
    fsm.set_draw_mode(DrawMode.PRECISION_PINCH)
    assert fsm.draw_mode == DrawMode.PRECISION_PINCH
    state = feed_frames(fsm, lm_true_pinch, n=5)
    assert state == GestureState.DRAW

    # When thumb is released (unpinched), in Precision Pinch mode it switches to POINT (laser)
    lm_unpinched = _make_synthetic_landmarks(
        index_tip_y=0.15,
        index_pip_y=0.40,
        thumb_tip_xy=(0.35, 0.35),  # far from index tip
        middle_tip_y=0.60,
        middle_pip_y=0.45,
        ring_tip_y=0.70,
        ring_pip_y=0.50,
        pinky_tip_y=0.72,
        pinky_pip_y=0.52,
    )
    state = feed_frames(fsm, lm_unpinched, n=5)
    assert state == GestureState.POINT


def test_pinch_threshold_configuration():
    fsm = HandGestureFSM()
    fsm.set_draw_mode(DrawMode.PRECISION_PINCH)

    # Set custom sensitivity
    fsm.set_pinch_threshold(0.20)
    assert abs(fsm.pinch_threshold - 0.20) < 1e-4

    # Clamping bounds
    fsm.set_pinch_threshold(0.01)
    assert fsm.pinch_threshold >= 0.12
    fsm.set_pinch_threshold(0.99)
    assert fsm.pinch_threshold <= 0.55


def test_tracking_gap_safety_stops_stroke():
    """Verify that losing hand detection for 2 consecutive frames cleanly PAUSEs active stroke."""
    fsm = HandGestureFSM()
    lm_draw = _make_synthetic_landmarks(
        index_tip_y=0.15,
        index_pip_y=0.40,
        thumb_tip_xy=(0.51, 0.38),
        middle_tip_y=0.60,
        middle_pip_y=0.45,
        ring_tip_y=0.70,
        ring_pip_y=0.50,
        pinky_tip_y=0.72,
        pinky_pip_y=0.52,
    )

    # Start drawing
    state = feed_frames(fsm, lm_draw, n=5)
    assert state == GestureState.DRAW

    # Frame 1 with lost tracking (confidence = 0.0)
    state, cx, cy = fsm.process_landmarks([], 0.0, 100.0)
    # Debouncing / holding state
    assert cx == -1.0 and cy == -1.0

    # Frame 2 with lost tracking: gap threshold reached -> must trigger PAUSE
    state, cx, cy = fsm.process_landmarks([], 0.0, 116.0)
    assert state == GestureState.PAUSE
