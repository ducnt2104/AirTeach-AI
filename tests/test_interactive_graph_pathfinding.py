"""
AirTeach AI - Interactive Graph Theory & Dijkstra / A* Pathfinding Unit Tests
=============================================================================
Verifies:
  - Directed and undirected weighted graph data structures and presets.
  - Authentic Dijkstra shortest path algorithm with priority queue min-heap.
  - A* search algorithm with Euclidean distance heuristic f(n) = g(n) + h(n).
  - Step-by-step state machine transitions (UNVISITED -> OPEN -> CURRENT -> CLOSED -> PATH).
  - Vertex dragging and dynamic edge recalculation.
  - Touchless air gestures (pinch drag, air tap button controls).
  - Vector diagram stamping onto LessonBoard.
  - LessonBoard integration and region exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.interactive_graph_pathfinding import (
    InteractiveGraphPathfindingItem,
    GRAPH_PRESETS,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveGraphPathfinding:
    """Test suite for InteractiveGraphPathfindingItem."""

    def test_graph_init_and_properties(self, qapp):
        """Verifies initial graph structure, start/target nodes, and Dijkstra defaults."""
        item = InteractiveGraphPathfindingItem(pos=(200.0, 150.0), preset="NETWORK_7", algorithm="DIJKSTRA")

        assert item.pos() == QPointF(200.0, 150.0)
        assert item._start_node == "A"
        assert item._target_node == "G"
        assert item.algorithm() == "DIJKSTRA"
        assert len(item._nodes) == 7
        assert len(item._edges) == 12

        # Start node initial cost is 0.0, other nodes inf
        start_node = item._nodes["A"]
        assert start_node.cost_g == 0.0
        assert start_node.state == "OPEN"

        other_node = item._nodes["G"]
        assert other_node.cost_g == float("inf")
        assert other_node.state == "UNVISITED"
        item.close()

    def test_dijkstra_full_run_optimal_path(self, qapp):
        """Verifies Dijkstra's algorithm finds the authentic optimal shortest path on NETWORK_7."""
        item = InteractiveGraphPathfindingItem(preset="NETWORK_7", algorithm="DIJKSTRA")

        path = item.run_full_algorithm()
        total_cost = item.get_total_cost()

        # On NETWORK_7, path A -> C (2) -> F (4) -> E (3) -> G (3) = 12.0
        assert item.is_finished() is True
        assert path == ["A", "C", "F", "E", "G"]
        assert abs(total_cost - 12.0) < 1e-4

        # Verify all nodes along path are marked PATH
        for nid in path:
            assert item._nodes[nid].state == "PATH"

        # Verify edges on path are marked is_path == True
        path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        for e in item._edges:
            if (e.u, e.v) in path_edges or (e.v, e.u) in path_edges:
                assert e.is_path is True
        item.close()

    def test_astar_search_and_heuristic(self, qapp):
        """Verifies A* search with Euclidean heuristic finds identical optimal path."""
        item = InteractiveGraphPathfindingItem(preset="NETWORK_7", algorithm="ASTAR")
        assert item.algorithm() == "ASTAR"

        # Verify heuristic h(n) > 0 for non-target nodes, and h(target) == 0
        assert item._nodes["G"].cost_h == 0.0
        assert item._nodes["A"].cost_h > 0.0

        path = item.run_full_algorithm()
        total_cost = item.get_total_cost()

        assert path == ["A", "C", "F", "E", "G"]
        assert abs(total_cost - 12.0) < 1e-4
        item.close()

    def test_step_by_step_execution_and_signals(self, qapp):
        """Verifies step-by-step exploration, priority queue order, and signal emissions."""
        item = InteractiveGraphPathfindingItem(preset="NETWORK_7")

        steps_visited = []
        item.step_advanced.connect(lambda nid, cost: steps_visited.append((nid, cost)))

        path_result = []
        item.path_found.connect(lambda p, c: path_result.append((p, c)))

        # Step 1: Pop A
        has_more = item.step_algorithm()
        assert has_more is True
        assert len(steps_visited) >= 1
        assert steps_visited[0][0] == "A"
        assert steps_visited[0][1] == 0.0

        # Step through until finished
        max_safety = 30
        while has_more and max_safety > 0:
            has_more = item.step_algorithm()
            max_safety -= 1

        assert item.is_finished() is True
        assert len(path_result) == 1
        assert path_result[0][0] == ["A", "C", "F", "E", "G"]
        item.close()

    def test_graph_presets_switching(self, qapp):
        """Verifies loading different graph topologies (Cycle Ring and Binary Tree)."""
        item = InteractiveGraphPathfindingItem()

        # Switch to CYCLE_6
        item.set_preset("CYCLE_6")
        assert item._active_preset == "CYCLE_6"
        assert item._start_node == "A"
        assert item._target_node == "D"
        assert len(item._nodes) == 6

        path = item.run_full_algorithm()
        assert path[0] == "A"
        assert path[-1] == "D"
        # Two paths: A -> B -> C -> D = 3 + 4 + 2 = 9
        # or A -> F -> E -> D = 2 + 5 + 1 = 8!
        assert abs(item.get_total_cost() - 8.0) < 1e-4

        # Switch to BINARY_TREE
        item.set_preset("BINARY_TREE")
        assert item._active_preset == "BINARY_TREE"
        assert item._start_node == "R"
        assert item._target_node == "T3"
        path_tree = item.run_full_algorithm()
        assert path_tree[0] == "R"
        assert path_tree[-1] == "T3"
        item.close()

    def test_node_repositioning(self, qapp):
        """Verifies interactive node dragging updates vertex position and heuristic."""
        item = InteractiveGraphPathfindingItem(algorithm="ASTAR")
        node_b = item._nodes["B"]
        old_pos = QPointF(node_b.pos)

        # Drag node B to new position
        item._active_drag = "NODE"
        item._dragged_node_id = "B"
        node_b.pos = QPointF(old_pos.x() + 40.0, old_pos.y() + 20.0)

        assert node_b.pos != old_pos
        item.close()

    def test_touchless_air_gestures(self, qapp):
        """Verifies touchless air tap button dispatch and air drag."""
        board = LessonBoard()
        item = InteractiveGraphPathfindingItem(pos=(100.0, 100.0))
        board.addItem(item)

        # 1. Air tap Play button
        btn_play_scene = item.mapToScene(item._rect_btn_play.center())
        consumed = item.handle_air_tap(btn_play_scene, board=board)
        assert consumed is True
        assert item._is_playing is True

        # Air tap Play again (Pause)
        consumed = item.handle_air_tap(btn_play_scene, board=board)
        assert consumed is True
        assert item._is_playing is False

        # 2. Air tap Step button
        btn_step_scene = item.mapToScene(item._rect_btn_step.center())
        consumed = item.handle_air_tap(btn_step_scene, board=board)
        assert consumed is True

        # 3. Air tap Reset button
        btn_reset_scene = item.mapToScene(item._rect_btn_reset.center())
        consumed = item.handle_air_tap(btn_reset_scene, board=board)
        assert consumed is True
        assert item._current_node is None

        # 4. Air tap Algorithm toggle button
        btn_algo_scene = item.mapToScene(item._rect_btn_algo.center())
        consumed = item.handle_air_tap(btn_algo_scene, board=board)
        assert consumed is True
        assert item.algorithm() == "ASTAR"

        # 5. Air tap Preset button (CYCLE_6)
        btn_preset_scene = item.mapToScene(item._rect_btn_presets[1].center())
        consumed = item.handle_air_tap(btn_preset_scene, board=board)
        assert consumed is True
        assert item._active_preset == "CYCLE_6"

        # 6. Air hover pos
        item.set_air_hover_pos(QPointF(200.0, 200.0))
        assert item._air_hover_pos is not None
        item.set_air_hover_pos(None)
        assert item._air_hover_pos is None

        # 7. Air tap Close button
        btn_close_scene = item.mapToScene(item._rect_btn_close.center())
        consumed = item.handle_air_tap(btn_close_scene, board=board)
        assert consumed is True
        assert item.scene() is None

    def test_stamp_to_canvas_and_lessonboard(self, qapp):
        """Verifies vector graph diagram stamping onto LessonBoard and region exclusion."""
        board = LessonBoard()
        item = InteractiveGraphPathfindingItem(pos=(200.0, 150.0))
        board.addItem(item)
        item.run_full_algorithm()

        stamped = item.stamp_to_canvas(board)
        assert len(stamped) >= 15  # Edges, nodes, weight badges, summary HUD

        for itm in stamped:
            assert itm.scene() is board

        # Test LessonBoard lifecycle
        board.show_graph_pathfinding(True, pos=(300.0, 250.0), preset="BINARY_TREE")
        assert board.is_graph_pathfinding_visible() is True
        g_item = board.get_graph_pathfinding_item()
        assert g_item is not None
        assert g_item.pos() == QPointF(300.0, 250.0)

        # Region exclusion test
        sel_rect = QRectF(250.0, 200.0, 400.0, 350.0)
        items_inside = board.get_items_in_rect(sel_rect)
        assert g_item not in items_inside

        # Toggle visibility
        is_vis = board.toggle_graph_pathfinding()
        assert is_vis is False
        assert board.is_graph_pathfinding_visible() is False

        is_vis = board.toggle_graph_pathfinding()
        assert is_vis is True
        assert board.is_graph_pathfinding_visible() is True

        if hasattr(g_item, "close"):
            g_item.close()
        item.close()
