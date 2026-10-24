"""
AirTeach AI - Tests for Interactive Minimum Spanning Tree (MST) Visualizer
==========================================================================
Validates greedy graph algorithms (Kruskal with DSU and Prim with priority cut),
cycle detection, edge selection invariants, and UI rendering.
"""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QGraphicsScene, QStyleOptionGraphicsItem

from src.ui.interactive_mst_visualizer import (
    GRAPH_PRESETS,
    DisjointSetUnion,
    InteractiveMstVisualizerItem,
)


@pytest.fixture
def mst_item(qapp):
    return InteractiveMstVisualizerItem(preset_key="STANDARD_NETWORK", algorithm="KRUSKAL")


def test_dsu_operations():
    """Verify Disjoint Set Union correctly manages components and detects cycles."""
    dsu = DisjointSetUnion(5)

    # Initially all in separate sets
    assert dsu.find(0) != dsu.find(1)

    # Union (0, 1) and (1, 2)
    assert dsu.union(0, 1) is True
    assert dsu.find(0) == dsu.find(1)

    assert dsu.union(1, 2) is True
    assert dsu.find(0) == dsu.find(2)

    # (0, 2) is now redundant (would create cycle)
    assert dsu.union(0, 2) is False


def test_kruskal_algorithm_completion(mst_item):
    """Running Kruskal to completion must yield |V|-1 edges and valid MST weight."""
    mst_item.apply_preset("STANDARD_NETWORK")
    n_nodes = len(mst_item._nodes)

    # Step through all edges
    for _ in range(30):
        mst_item.step_forward()

    assert mst_item._is_completed is True
    assert len(mst_item._accepted_edge_ids) == n_nodes - 1
    assert mst_item._mst_total_weight > 0
    # In standard network (7 nodes), MST has 6 edges
    assert len(mst_item._accepted_edge_ids) == 6


def test_prim_algorithm_completion(mst_item):
    """Running Prim algorithm must yield |V|-1 edges and equal total weight to Kruskal."""
    # 1. Run Kruskal to get benchmark weight
    mst_item.apply_preset("STANDARD_NETWORK")
    for _ in range(30):
        mst_item.step_forward()
    kruskal_weight = mst_item._mst_total_weight

    # 2. Run Prim
    mst_item.toggle_algorithm()
    assert mst_item._algorithm == "PRIM"

    for _ in range(30):
        mst_item.step_forward()

    assert mst_item._is_completed is True
    assert len(mst_item._accepted_edge_ids) == len(mst_item._nodes) - 1
    assert mst_item._mst_total_weight == kruskal_weight


def test_dense_graph_cycle_rejection(mst_item):
    """In complete graph K5 (10 edges, 5 nodes), exactly 4 edges accepted and 6 rejected."""
    mst_item.apply_preset("COMPLETE_K5")

    for _ in range(25):
        mst_item.step_forward()

    assert len(mst_item._accepted_edge_ids) == 4
    # Total edges in K5 is 10, so rejected is 6
    assert len(mst_item._rejected_edge_ids) == 6


def test_presets_loading(mst_item):
    """Verify loading all graph presets."""
    for key in GRAPH_PRESETS:
        mst_item.apply_preset(key)
        assert len(mst_item._nodes) > 0
        assert len(mst_item._edges) > 0
        assert mst_item._mst_total_weight == 0


def test_stamping_signal(mst_item):
    """Verify stamp request emits complete graph data."""
    stamps = []
    mst_item.stamp_requested.connect(lambda d: stamps.append(d))
    mst_item.step_forward()
    mst_item.request_stamp()

    assert len(stamps) == 1
    assert "accepted_edges" in stamps[0]
    assert "mst_weight" in stamps[0]


def test_paint_and_render_zero_crash(qapp, mst_item):
    """Verify paint rendering executes without crashing on a QImage."""
    scene = QGraphicsScene()
    scene.addItem(mst_item)

    image = QImage(740, 460, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.black)
    painter = QPainter(image)
    option = QStyleOptionGraphicsItem()
    mst_item.paint(painter, option, None)
    painter.end()

    assert not image.isNull()
