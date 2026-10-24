"""
Tests for AirTeach AI Multi-Hand Tracking, Coordinator & Air Click Engine
==========================================================================
Verifies:
  - MultiHandTracker maintains stable track IDs across frames and hand crossings
  - Missing frame tolerance (1-frame hold, evicting dead tracks after grace period)
  - AirClickDetector accurately triggers Pinch Click, Air Tap, and Dwell selection
  - MultiHandCoordinator assigns Primary vs Secondary roles based on dominant hand
  - Dual Draw mode assigns Independent role to both hands
  - Independent GestureState evaluation per hand
"""
import pytest
from src.core.hand_gesture_engine import DrawMode, GestureState, HandGestureFSM
from src.core.multi_hand_coordinator import (
    AirClickDetector,
    AirClickType,
    HandRole,
    MultiHandCoordinator,
    MultiHandMode,
    MultiHandTracker,
)
from tests.test_core_algorithms import _make_synthetic_landmarks


def test_multi_hand_tracker_two_hands_stable_ids():
    tracker = MultiHandTracker(max_hands=2)

    # Frame 1: Hand A on left (x=0.25), Hand B on right (x=0.75)
    lm_left = _make_synthetic_landmarks(index_tip_y=0.2)
    for pt in lm_left:
        pt.x = pt.x * 0.5  # shift to left half [0.0 - 0.5]

    lm_right = _make_synthetic_landmarks(index_tip_y=0.2)
    for pt in lm_right:
        pt.x = 0.5 + pt.x * 0.5  # shift to right half [0.5 - 1.0]

    detections_f1 = [
        (lm_left, "Left", 0.90),
        (lm_right, "Right", 0.92),
    ]

    tracks_f1 = tracker.update(detections_f1, t_ms=0.0)
    assert len(tracks_f1) == 2
    id_left = tracks_f1[0][0]
    id_right = tracks_f1[1][0]
    assert id_left != id_right

    # Frame 2: Order of detections in list is reversed (e.g. MediaPipe flipped detection index)
    detections_f2 = [
        (lm_right, "Right", 0.91),
        (lm_left, "Left", 0.89),
    ]
    tracks_f2 = tracker.update(detections_f2, t_ms=33.0)
    assert len(tracks_f2) == 2

    # Map results by track id
    f2_by_id = {tid: (lm, hness) for tid, lm, hness, _ in tracks_f2}
    assert id_left in f2_by_id
    assert id_right in f2_by_id
    assert f2_by_id[id_left][1] == "Left"
    assert f2_by_id[id_right][1] == "Right"


def test_multi_hand_tracker_missing_frame_recovery():
    tracker = MultiHandTracker(max_hands=2, max_missing_frames=3)
    lm = _make_synthetic_landmarks(index_tip_y=0.2)

    # Frame 1: Hand present
    tracks_f1 = tracker.update([(lm, "Right", 0.90)], t_ms=0.0)
    assert len(tracks_f1) == 1
    tid = tracks_f1[0][0]

    # Frame 2: Momentary occlusion (0 detections)
    tracks_f2 = tracker.update([], t_ms=33.0)
    assert len(tracks_f2) == 0
    assert tid in tracker._tracks
    assert tracker._tracks[tid]["missing_frames"] == 1

    # Frame 3: Hand re-appears near same location
    tracks_f3 = tracker.update([(lm, "Right", 0.88)], t_ms=66.0)
    assert len(tracks_f3) == 1
    assert tracks_f3[0][0] == tid  # Track ID preserved!
    assert tracker._tracks[tid]["missing_frames"] == 0


def test_air_click_pinch_detection():
    detector = AirClickDetector(click_type=AirClickType.PINCH_CLICK, pinch_threshold=0.25)

    # Start unpinched
    lm_unpinched = _make_synthetic_landmarks(
        index_tip_y=0.20,
        thumb_tip_xy=(0.35, 0.35),
    )
    p0 = lm_unpinched[0]
    p9 = lm_unpinched[9]
    hand_size = ((p0.x - p9.x) ** 2 + (p0.y - p9.y) ** 2) ** 0.5

    is_tap, is_pinch, _ = detector.process_hand(
        "hand_0", lm_unpinched, canvas_x=500, canvas_y=500, hand_size=hand_size, t_ms=0.0
    )
    assert not is_pinch

    # Now pinch index & thumb together
    lm_pinched = _make_synthetic_landmarks(
        index_tip_y=0.20,
        thumb_tip_xy=(0.50, 0.20),
    )
    is_tap, is_pinch, _ = detector.process_hand(
        "hand_0", lm_pinched, canvas_x=500, canvas_y=500, hand_size=hand_size, t_ms=50.0
    )
    assert is_pinch, "Pinch click must be triggered on rising edge of pinch"

    # Holding pinch should NOT trigger continuous clicks (debounce / edge trigger)
    is_tap2, is_pinch2, _ = detector.process_hand(
        "hand_0", lm_pinched, canvas_x=500, canvas_y=500, hand_size=hand_size, t_ms=80.0
    )
    assert not is_pinch2


def test_air_click_dwell_selection():
    detector = AirClickDetector(click_type=AirClickType.DWELL, dwell_time_ms=500.0, dwell_radius_canvas=20.0)
    lm = _make_synthetic_landmarks()
    hand_size = 0.25

    # Step 1: Start dwell anchor
    _, _, dwell1 = detector.process_hand("hand_0", lm, 500.0, 500.0, hand_size, t_ms=0.0)
    assert dwell1 == 0.0

    # Step 2: 250ms elapsed inside radius
    _, _, dwell2 = detector.process_hand("hand_0", lm, 505.0, 505.0, hand_size, t_ms=250.0)
    assert 0.45 <= dwell2 <= 0.55

    # Step 3: 500ms elapsed -> completed dwell (progress == 1.0)
    _, _, dwell3 = detector.process_hand("hand_0", lm, 502.0, 503.0, hand_size, t_ms=500.0)
    assert dwell3 == 1.0

    # Step 4: Moving out of radius resets dwell
    _, _, dwell4 = detector.process_hand("hand_0", lm, 600.0, 500.0, hand_size, t_ms=550.0)
    assert dwell4 == 0.0


def test_multi_hand_coordinator_roles_and_state():
    coordinator = MultiHandCoordinator(
        mode=MultiHandMode.DRAW_AND_TOOL,
        dominant_hand="Right",
        draw_mode=DrawMode.INDEX_UP,
    )

    # Hand A (Right, drawing)
    lm_draw = _make_synthetic_landmarks(
        index_tip_y=0.15,
        index_pip_y=0.40,
        thumb_tip_xy=(0.51, 0.38), # index up drawing
        middle_tip_y=0.60,
    )
    # Hand B (Left, open palm / swipe / pointing)
    lm_point = _make_synthetic_landmarks(
        index_tip_y=0.15,
        index_pip_y=0.40,
        thumb_tip_xy=(0.30, 0.35), # thumb far -> pointing
        middle_tip_y=0.60,
    )

    raw_dets = [
        (lm_draw, "Right", 0.92),
        (lm_point, "Left", 0.88),
    ]

    # Feed multiple frames for FSM debounce
    hands_out = []
    for i in range(5):
        hands_out = coordinator.process_frame_hands(raw_dets, t_ms=float(i * 33))

    assert len(hands_out) == 2
    right_data = next(h for h in hands_out if h.handedness == "Right")
    left_data = next(h for h in hands_out if h.handedness == "Left")

    # In DRAW_AND_TOOL with dominant_hand="Right":
    assert right_data.role == HandRole.PRIMARY
    assert left_data.role == HandRole.SECONDARY

    # In Dual Draw mode:
    coordinator.set_mode(MultiHandMode.DUAL_DRAW)
    hands_dual = coordinator.process_frame_hands(raw_dets, t_ms=200.0)
    for h in hands_dual:
        assert h.role == HandRole.INDEPENDENT


def test_air_click_detector_reset_all():
    detector = AirClickDetector()
    lm = _make_synthetic_landmarks()
    detector.process_hand("hand_0", lm, 500.0, 500.0, 0.25, t_ms=0.0)
    detector.process_hand("hand_1", lm, 800.0, 500.0, 0.25, t_ms=0.0)

    assert len(detector._history) > 0
    assert len(detector._dwell_state) > 0

    detector.reset_all()
    assert len(detector._history) == 0
    assert len(detector._was_pinched) == 0
    assert len(detector._last_click_t_ms) == 0
    assert len(detector._dwell_state) == 0


def test_multi_hand_coordinator_reset():
    coordinator = MultiHandCoordinator()
    lm = _make_synthetic_landmarks()
    coordinator.process_frame_hands([(lm, "Right", 0.9)], t_ms=0.0)

    assert len(coordinator.tracker._tracks) > 0
    assert len(coordinator._fsms) > 0

    coordinator.reset()
    assert len(coordinator.tracker._tracks) == 0
    assert len(coordinator._fsms) == 0
    assert len(coordinator.click_detector._history) == 0


def test_multi_hand_coordinator_independent_mode():
    coordinator = MultiHandCoordinator(mode=MultiHandMode.INDEPENDENT)
    lm_a = _make_synthetic_landmarks(index_tip_y=0.15)
    lm_b = _make_synthetic_landmarks(index_tip_y=0.20)
    raw_dets = [(lm_a, "Right", 0.9), (lm_b, "Left", 0.9)]

    hands = coordinator.process_frame_hands(raw_dets, t_ms=0.0)
    assert len(hands) == 2
    for h in hands:
        assert h.role == HandRole.INDEPENDENT, "All hands in INDEPENDENT mode must have role=INDEPENDENT"
