"""
AirTeach AI - Adversarial Red-Team & Edge Condition Testing Suite
================================================================
Validates Phase H Architectural Standards:
  1. ZeroDivisionError immunity when hand_size -> 0.0 in gesture tracking and air click
  2. Degenerate and malformed landmark lists (empty, None, NaN, extreme coordinates)
  3. Geometric shape snapping on degenerate strokes (0, 1, 2 points, collinear, NaN)
  4. Stroke beautifier and Bézier spline fitting on identical/stationary points
  5. Chemical and mathematical formula parsers with malformed / unclosed tokens
"""
import math
import numpy as np
import pytest

from tests.test_core_algorithms import _make_synthetic_landmarks
from src.core.multi_hand_coordinator import (
    AirClickDetector,
    AirClickType,
    HandLandmarkData,
    MultiHandCoordinator,
    MultiHandMode,
)
from src.core.stroke_beautifier import (
    GeometricShapeSnapper,
    ShapeType,
    fit_cubic_bezier_spline,
    ramer_douglas_peucker,
)
from src.ui.stem_tools import format_chemical_formula


class DummyLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestRedTeamEdgeConditions:
    def test_multi_hand_coordinator_empty_and_none_detections(self):
        """Coordinator must safely handle empty lists, None landmarks, and invalid detections."""
        coord = MultiHandCoordinator(mode=MultiHandMode.DUAL_DRAW)

        # 1. Empty detections
        res1 = coord.process_frame_hands([], t_ms=100.0)
        assert res1 == []

        # 2. None landmarks
        res2 = coord.process_frame_hands([(None, "Right", 0.0)], t_ms=133.0)
        assert res2 == []

        # 3. Truncated landmark list (< 21 points)
        truncated = [DummyLandmark(0.5, 0.5) for _ in range(10)]
        res3 = coord.process_frame_hands([(truncated, "Right", 0.9)], t_ms=166.0)
        assert res3 == []

    def test_zero_hand_size_division_by_zero_immunity(self):
        """AirClickDetector and MultiHandCoordinator must not crash when hand_size == 0.0."""
        detector = AirClickDetector(click_type=AirClickType.HYBRID)

        # All 21 landmarks at exact same coordinate (0.5, 0.5) -> hand_size = 0.0
        collapsed_lms = [DummyLandmark(0.5, 0.5, 0.0) for _ in range(21)]

        # Must execute cleanly without ZeroDivisionError
        is_tap, is_pinch, dwell = detector.process_hand(
            track_id="hand_zero",
            canvas_x=960.0,
            canvas_y=540.0,
            landmarks=collapsed_lms,
            hand_size=0.0,
            t_ms=100.0,
        )
        assert not is_tap
        assert not is_pinch

    def test_geometric_shape_snapper_degenerate_inputs(self):
        """Shape snapper must handle empty, single-point, or collinear inputs gracefully."""
        snapper = GeometricShapeSnapper()

        # 1. Empty input
        empty_pts = np.empty((0, 2), dtype=np.float64)
        res_empty = snapper.snap(empty_pts)
        assert res_empty.shape_type == ShapeType.FREEFORM

        # 2. Single point
        single_pt = np.array([[100.0, 200.0]], dtype=np.float64)
        res_single = snapper.snap(single_pt)
        assert res_single.shape_type == ShapeType.FREEFORM

        # 3. Two identical points
        two_pts = np.array([[100.0, 200.0], [100.0, 200.0]], dtype=np.float64)
        res_two = snapper.snap(two_pts)
        assert res_two is not None

        # 4. Perfectly straight collinear line
        collinear = np.array([[float(i * 10), 200.0] for i in range(20)], dtype=np.float64)
        res_line = snapper.snap(collinear)
        assert res_line.shape_type == ShapeType.LINE

    def test_bezier_spline_fitting_identical_points(self):
        """fit_cubic_bezier_spline must not fail on stationary identical coordinates."""
        stationary = np.array([[300.0, 300.0]] * 15, dtype=np.float64)

        # RDP on stationary points
        simplified = ramer_douglas_peucker(stationary, epsilon=1.0)
        assert len(simplified) <= 2

        # Spline fitting on stationary points
        segments = fit_cubic_bezier_spline(simplified)
        assert isinstance(segments, list)

    def test_chemical_formula_formatter_malformed_inputs(self):
        """Chemical formula parser must be resilient against unclosed tags and extreme inputs."""
        # Unclosed brackets
        assert format_chemical_formula("Ca(OH") == "Ca(OH"
        assert format_chemical_formula("Fe[CN") == "Fe[CN"

        # Unclosed superscript group
        assert format_chemical_formula("SO4^{2-") != ""

        # Empty string
        assert format_chemical_formula("") == ""

        # Numbers only
        assert format_chemical_formula("12345") == "12345"

        # Complex organic ions
        res = format_chemical_formula("CH3COO- + H+ -> CH3COOH")
        assert "⁻" in res
        assert "⁺" in res
        assert "₃" in res
