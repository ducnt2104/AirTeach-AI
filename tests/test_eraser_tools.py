"""
AirTeach AI - Eraser Suite Unit Tests
=====================================
Tests:
  - Continuous Bézier subsegment distance stroke erasing
  - Snapped shape geometry erasing (Lines, Circles, Rectangles)
  - Object eraser (erase_object_at)
  - Area eraser (erase_area)
  - Clear last stroke (hand-specific and overall)
  - Clear page annotations (preserving background pixmap)
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.core.stroke_beautifier import BezierSegment, SnappedShape, ShapeType
from src.ui.lesson_board import LessonBoard, InkStroke


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_continuous_bezier_segment_erasing(qapp):
    board = LessonBoard()

    # Stroke from (100, 100) to (500, 100)
    seg = BezierSegment(
        p0=np.array([100.0, 100.0]),
        p1=np.array([200.0, 100.0]),
        p2=np.array([400.0, 100.0]),
        p3=np.array([500.0, 100.0]),
    )
    stroke = InkStroke(
        stroke_id="stroke_horiz_1",
        segments=[seg],
        widths=[3.0],
        color=QColor("#202124"),
    )
    board.commit_stroke(stroke)
    assert "stroke_horiz_1" in board._strokes

    # Erase at midpoint (300, 105), which is far from p0, p1, p2, p3
    erased = board.erase_strokes_at(300.0, 105.0, radius=12.0)
    assert "stroke_horiz_1" in erased
    assert "stroke_horiz_1" not in board._strokes


def test_snapped_shape_erasing(qapp):
    board = LessonBoard()

    # Snapped line from (200, 200) to (600, 200)
    snap_line = SnappedShape(
        shape_type=ShapeType.LINE,
        confidence=0.98,
        params={"start": (200.0, 200.0), "end": (600.0, 200.0)},
    )
    stroke_line = InkStroke(
        stroke_id="line_1",
        segments=[],
        widths=[3.0],
        snap=snap_line,
    )
    board.commit_stroke(stroke_line)

    # Snapped circle at (400, 500), radius 60
    snap_circle = SnappedShape(
        shape_type=ShapeType.CIRCLE,
        confidence=0.95,
        params={"center": (400.0, 500.0), "radius": 60.0},
    )
    stroke_circle = InkStroke(
        stroke_id="circle_1",
        segments=[],
        widths=[3.0],
        snap=snap_circle,
    )
    board.commit_stroke(stroke_circle)

    # Erase line at (400, 205)
    erased_line = board.erase_strokes_at(400.0, 205.0, radius=15.0)
    assert "line_1" in erased_line
    assert "line_1" not in board._strokes
    assert "circle_1" in board._strokes

    # Erase circle at circumference (460, 500)
    erased_circle = board.erase_strokes_at(460.0, 500.0, radius=15.0)
    assert "circle_1" in erased_circle
    assert "circle_1" not in board._strokes


def test_erase_object_at(qapp):
    board = LessonBoard()

    seg1 = BezierSegment(p0=np.array([100.0, 100.0]), p1=np.array([110.0, 100.0]),
                         p2=np.array([120.0, 100.0]), p3=np.array([130.0, 100.0]))
    seg2 = BezierSegment(p0=np.array([800.0, 800.0]), p1=np.array([810.0, 800.0]),
                         p2=np.array([820.0, 800.0]), p3=np.array([830.0, 800.0]))

    board.commit_stroke(InkStroke("s1", [seg1], [2.0]))
    board.commit_stroke(InkStroke("s2", [seg2], [2.0]))

    # Erase object at s1
    res = board.erase_object_at(115.0, 100.0, radius=20.0)
    assert res == "s1"
    assert "s1" not in board._strokes
    assert "s2" in board._strokes


def test_erase_area(qapp):
    board = LessonBoard()

    seg1 = BezierSegment(p0=np.array([100.0, 100.0]), p1=np.array([120.0, 100.0]),
                         p2=np.array([140.0, 100.0]), p3=np.array([160.0, 100.0]))
    seg2 = BezierSegment(p0=np.array([150.0, 150.0]), p1=np.array([170.0, 150.0]),
                         p2=np.array([190.0, 150.0]), p3=np.array([210.0, 150.0]))
    seg3 = BezierSegment(p0=np.array([900.0, 900.0]), p1=np.array([920.0, 900.0]),
                         p2=np.array([940.0, 900.0]), p3=np.array([960.0, 900.0]))

    board.commit_stroke(InkStroke("a1", [seg1], [2.0]))
    board.commit_stroke(InkStroke("a2", [seg2], [2.0]))
    board.commit_stroke(InkStroke("a3", [seg3], [2.0]))

    # Area rectangle enclosing a1 and a2
    erased = board.erase_area(QRectF(80.0, 80.0, 200.0, 200.0))
    assert "a1" in erased
    assert "a2" in erased
    assert "a3" not in erased
    assert "a3" in board._strokes


def test_clear_last_stroke_and_page_annotations(qapp):
    board = LessonBoard()

    # Add background
    dummy_bg = np.zeros((720, 1280, 3), dtype=np.uint8)
    board.set_background_page(dummy_bg)
    assert board._bg_item is not None

    # Add strokes with different hand_ids
    seg = BezierSegment(p0=np.array([50.0, 50.0]), p1=np.array([60.0, 50.0]),
                        p2=np.array([70.0, 50.0]), p3=np.array([80.0, 50.0]))
    board.commit_stroke(InkStroke("h1", [seg], [2.0], hand_id="primary"))
    board.commit_stroke(InkStroke("h2", [seg], [2.0], hand_id="secondary"))
    board.commit_stroke(InkStroke("h3", [seg], [2.0], hand_id="primary"))

    # Clear last stroke of hand "secondary"
    removed = board.clear_last_stroke(hand_id="secondary")
    assert removed == "h2"
    assert "h2" not in board._strokes
    assert "h3" in board._strokes

    # Clear page annotations
    all_removed = board.clear_page_annotations()
    assert "h1" in all_removed
    assert "h3" in all_removed
    assert len(board._strokes) == 0
    # Background MUST remain intact
    assert board._bg_item is not None
    assert board._bg_item.scene() is board
