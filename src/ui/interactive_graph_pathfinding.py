"""
AirTeach AI - Interactive Computer Science Graph Theory & Pathfinding Simulator
==============================================================================
Provides an interactive computer science simulation of Graph Theory and Shortest
Path algorithms (Dijkstra & A* Search):
  - Directed and undirected weighted graphs G = (V, E).
  - Graph vertices V with draggable spatial positions and dynamic edge routing.
  - Live Priority Queue / Open Set min-heap frontier exploration.
  - Dynamic heuristic evaluation function f(n) = g(n) + h(n) in A* mode.
  - Step-by-step visual animation with color-coded states:
      * Unvisited (Gray)
      * Open Set / Frontier (Cyan)
      * Current Evaluating (Amber)
      * Closed Set / Visited (Emerald)
      * Optimal Shortest Path (Golden Glowing Neon)
  - Interactive controls: Auto-play, Step, Reset, Mode toggle, Presets.
  - Touchless air gestures (pinch drag nodes, tap buttons).
  - Vector diagram stamping onto LessonBoard via InformaticsTools.create_graph_diagram.
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass
class PathNode:
    """Vertex in the interactive pathfinding graph."""
    node_id: str
    pos: QPointF  # Local item coordinates
    cost_g: float = float("inf")  # Distance from start
    cost_h: float = 0.0          # Heuristic estimate to target
    parent_id: Optional[str] = None
    state: str = "UNVISITED"      # UNVISITED, OPEN, CURRENT, CLOSED, PATH

    @property
    def cost_f(self) -> float:
        return self.cost_g + self.cost_h


@dataclass
class PathEdge:
    """Weighted edge between two vertices."""
    u: str
    v: str
    weight: float
    is_directed: bool = False
    is_path: bool = False


# Standard Graph Presets
GRAPH_PRESETS: dict[str, dict] = {
    "NETWORK_7": {
        "name": "Mạng 7 Đỉnh (DAG)",
        "start": "A",
        "target": "G",
        "nodes": {
            "A": (75.0, 205.0),
            "B": (185.0, 115.0),
            "C": (185.0, 295.0),
            "D": (325.0, 95.0),
            "E": (325.0, 205.0),
            "F": (325.0, 315.0),
            "G": (475.0, 205.0),
        },
        "edges": [
            ("A", "B", 4.0),
            ("A", "C", 2.0),
            ("B", "D", 5.0),
            ("B", "E", 10.0),
            ("C", "B", 1.0),
            ("C", "E", 8.0),
            ("C", "F", 4.0),
            ("D", "G", 6.0),
            ("E", "D", 2.0),
            ("E", "G", 3.0),
            ("F", "E", 3.0),
            ("F", "G", 10.0),
        ],
    },
    "CYCLE_6": {
        "name": "Vòng Lục Giác (Ring)",
        "start": "A",
        "target": "D",
        "nodes": {
            "A": (120.0, 205.0),
            "B": (205.0, 105.0),
            "C": (355.0, 105.0),
            "D": (440.0, 205.0),
            "E": (355.0, 305.0),
            "F": (205.0, 305.0),
        },
        "edges": [
            ("A", "B", 3.0),
            ("B", "C", 4.0),
            ("C", "D", 2.0),
            ("A", "F", 2.0),
            ("F", "E", 5.0),
            ("E", "D", 1.0),
            ("B", "E", 6.0),
            ("C", "F", 7.0),
        ],
    },
    "BINARY_TREE": {
        "name": "Cây Nhị Phân (Tree)",
        "start": "R",
        "target": "T3",
        "nodes": {
            "R": (275.0, 95.0),
            "L1": (160.0, 185.0),
            "R1": (390.0, 185.0),
            "T1": (95.0, 295.0),
            "T2": (220.0, 295.0),
            "T3": (330.0, 295.0),
            "T4": (455.0, 295.0),
        },
        "edges": [
            ("R", "L1", 5.0),
            ("R", "R1", 7.0),
            ("L1", "T1", 3.0),
            ("L1", "T2", 4.0),
            ("R1", "T3", 2.0),
            ("R1", "T4", 6.0),
            ("T2", "T3", 3.0),
        ],
    },
}


class InteractiveGraphPathfindingItem(QGraphicsObject):
    """
    Interactive Graph Theory & Dijkstra / A* Pathfinding visualizer,
    featuring live draggable vertices, step-by-step search, Priority Queue HUD,
    and vector stamping onto LessonBoard.
    """

    step_advanced = Signal(str, float)   # (current_node_id, current_cost)
    path_found = Signal(list, float)     # (shortest_path_nodes, total_cost)
    graph_reset = Signal()

    def __init__(
        self,
        pos: tuple[float, float] = (500.0, 320.0),
        width: float = 680.0,
        height: float = 440.0,
        preset: str = "NETWORK_7",
        algorithm: str = "DIJKSTRA",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        self._active_preset = preset if preset in GRAPH_PRESETS else "NETWORK_7"
        self._algorithm = "ASTAR" if algorithm.upper() == "ASTAR" else "DIJKSTRA"

        # Graph Data Structures
        self._nodes: dict[str, PathNode] = {}
        self._edges: list[PathEdge] = []
        self._adj: dict[str, list[tuple[str, float]]] = {}
        self._start_node = "A"
        self._target_node = "G"

        # Search Algorithm State
        self._open_pq: list[tuple[float, int, str]] = []  # min-heap (priority, tie-breaker, node_id)
        self._pq_counter = 0
        self._closed_set: set[str] = set()
        self._current_node: Optional[str] = None
        self._shortest_path: list[str] = []
        self._total_cost = 0.0
        self._is_finished = False
        self._log_message = "Sẵn sàng tìm đường đi ngắn nhất."

        # Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(450)  # ~2.2 steps/sec
        self._timer.timeout.connect(self._on_timer_step)
        self._is_playing = False

        # Dragging state
        self._dragged_node_id: Optional[str] = None
        self._active_drag: Optional[str] = None  # "NODE", "BODY"
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Control button bounding rects (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_play = QRectF(self._width - 245.0, 10.0, 90.0, 26.0)
        self._rect_btn_step = QRectF(self._width - 345.0, 10.0, 95.0, 26.0)
        self._rect_btn_reset = QRectF(self._width - 435.0, 10.0, 85.0, 26.0)
        self._rect_btn_algo = QRectF(self._width - 555.0, 10.0, 115.0, 26.0)

        # Preset selector buttons across bottom left
        self._preset_keys = ["NETWORK_7", "CYCLE_6", "BINARY_TREE"]
        self._rect_btn_presets: list[QRectF] = []
        btn_w = 95.0
        start_x = 18.0
        for i in range(len(self._preset_keys)):
            self._rect_btn_presets.append(QRectF(start_x + i * (btn_w + 6.0), self._height - 36.0, btn_w, 24.0))

        # Load initial preset
        self._load_preset(self._active_preset)

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── Graph Model & Presets ────────────────────────────────────────────────

    def _load_preset(self, preset_key: str) -> None:
        """Loads nodes, edges, start, and target from a graph preset."""
        data = GRAPH_PRESETS.get(preset_key, GRAPH_PRESETS["NETWORK_7"])
        self._active_preset = preset_key
        self._start_node = data["start"]
        self._target_node = data["target"]

        self._nodes.clear()
        for nid, (nx, ny) in data["nodes"].items():
            self._nodes[nid] = PathNode(node_id=nid, pos=QPointF(nx, ny))

        self._edges.clear()
        self._adj.clear()
        for nid in self._nodes:
            self._adj[nid] = []

        for u, v, w in data["edges"]:
            edge = PathEdge(u=u, v=v, weight=w)
            self._edges.append(edge)
            self._adj[u].append((v, w))
            # Treat as undirected for symmetric graph navigation
            self._adj[v].append((u, w))

        self.reset_search()

    def set_preset(self, preset_key: str) -> None:
        """Switches to another graph preset."""
        if preset_key in GRAPH_PRESETS:
            self._load_preset(preset_key)
            self.update()

    def set_algorithm(self, algo: str) -> None:
        """Sets active algorithm: 'DIJKSTRA' or 'ASTAR'."""
        self._algorithm = "ASTAR" if algo.upper() == "ASTAR" else "DIJKSTRA"
        self.reset_search()
        self.update()

    def algorithm(self) -> str:
        """Returns current algorithm name ('DIJKSTRA' or 'ASTAR')."""
        return self._algorithm

    # ─── Search Algorithm Engine ─────────────────────────────────────────────

    def reset_search(self) -> None:
        """Resets all node exploration states and priority queue."""
        self._is_playing = False
        self._timer.stop()
        self._open_pq.clear()
        self._pq_counter = 0
        self._closed_set.clear()
        self._current_node = None
        self._shortest_path.clear()
        self._total_cost = 0.0
        self._is_finished = False

        # Reset edges
        for e in self._edges:
            e.is_path = False

        # Reset nodes
        target_pt = self._nodes[self._target_node].pos if self._target_node in self._nodes else QPointF()
        for nid, node in self._nodes.items():
            node.cost_g = float("inf")
            # Admissible Euclidean heuristic h(n) = distance(n, target) / 25.0
            if self._algorithm == "ASTAR":
                dist = math.hypot(node.pos.x() - target_pt.x(), node.pos.y() - target_pt.y())
                node.cost_h = round(dist / 25.0, 1)
            else:
                node.cost_h = 0.0
            node.parent_id = None
            node.state = "UNVISITED"

        # Initialize Start Node
        if self._start_node in self._nodes:
            start_node = self._nodes[self._start_node]
            start_node.cost_g = 0.0
            start_node.state = "OPEN"
            priority = start_node.cost_f
            self._pq_counter += 1
            heapq.heappush(self._open_pq, (priority, self._pq_counter, self._start_node))

        algo_name = "Dijkstra" if self._algorithm == "DIJKSTRA" else "A* Search"
        self._log_message = f"Khởi tạo {algo_name}: Bắt đầu tại [{self._start_node}], Đích đến [{self._target_node}]."
        self.graph_reset.emit()
        self.update()

    def step_algorithm(self) -> bool:
        """
        Executes one discrete step of Dijkstra / A* Search.
        Returns True if more steps remain, False if finished.
        """
        if self._is_finished or not self._open_pq:
            self._is_finished = True
            self._is_playing = False
            self._timer.stop()
            return False

        # 1. Pop node with lowest f(n) (or g(n)) from Priority Queue
        priority, _, u_id = heapq.heappop(self._open_pq)
        u_node = self._nodes[u_id]

        # If already visited, continue to next
        if u_id in self._closed_set:
            return self.step_algorithm()

        # Mark current node
        if self._current_node and self._current_node in self._nodes:
            self._nodes[self._current_node].state = "CLOSED"

        self._current_node = u_id
        u_node.state = "CURRENT"
        self._closed_set.add(u_id)
        self.step_advanced.emit(u_id, u_node.cost_g)

        # 2. Check if Target reached
        if u_id == self._target_node:
            self._is_finished = True
            self._is_playing = False
            self._timer.stop()
            self._total_cost = u_node.cost_g
            self._reconstruct_shortest_path()
            self._log_message = f"🎯 Đã tìm thấy đường đi ngắn nhất: {' -> '.join(self._shortest_path)} (Chi phí = {self._total_cost:.1f})"
            self.path_found.emit(self._shortest_path, self._total_cost)
            self.update()
            return False

        # 3. Relax neighbors of u
        updated_neighbors = []
        for v_id, weight in self._adj.get(u_id, []):
            if v_id in self._closed_set:
                continue

            v_node = self._nodes[v_id]
            new_g = u_node.cost_g + weight

            if new_g < v_node.cost_g:
                v_node.cost_g = new_g
                v_node.parent_id = u_id
                v_node.state = "OPEN"
                priority = v_node.cost_f
                self._pq_counter += 1
                heapq.heappush(self._open_pq, (priority, self._pq_counter, v_id))
                updated_neighbors.append(f"{v_id}(g={new_g:.1f})")

        if updated_neighbors:
            self._log_message = f"Đang xét [{u_id}] (g={u_node.cost_g:.1f}) -> Cập nhật: {', '.join(updated_neighbors)}"
        else:
            self._log_message = f"Đang xét [{u_id}] -> Không có láng giềng mới."

        self.update()
        return True

    def _reconstruct_shortest_path(self) -> None:
        """Backtracks parent pointers from Target to Start to build shortest path."""
        self._shortest_path = []
        curr: Optional[str] = self._target_node
        while curr is not None:
            self._shortest_path.append(curr)
            node = self._nodes.get(curr)
            if node:
                node.state = "PATH"
            curr = node.parent_id if node else None

        self._shortest_path.reverse()

        # Mark path edges
        path_set = set()
        for i in range(len(self._shortest_path) - 1):
            path_set.add((self._shortest_path[i], self._shortest_path[i + 1]))
            path_set.add((self._shortest_path[i + 1], self._shortest_path[i]))

        for e in self._edges:
            if (e.u, e.v) in path_set:
                e.is_path = True

    def run_full_algorithm(self) -> list[str]:
        """Runs the entire algorithm synchronously to completion."""
        while self.step_algorithm():
            pass
        return self._shortest_path

    def toggle_auto_play(self) -> bool:
        """Toggles continuous step playback."""
        if self._is_finished:
            self.reset_search()

        self._is_playing = not self._is_playing
        if self._is_playing:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()
        return self._is_playing

    def _on_timer_step(self) -> None:
        """Timer callback advancing the pathfinding algorithm."""
        has_more = self.step_algorithm()
        if not has_more:
            self._is_playing = False
            self._timer.stop()

    def get_shortest_path(self) -> list[str]:
        """Returns the list of node IDs in the optimal shortest path."""
        return list(self._shortest_path)

    def get_total_cost(self) -> float:
        """Returns the total cost / distance of the shortest path."""
        return self._total_cost

    def is_finished(self) -> bool:
        """Returns True if the search algorithm reached the target."""
        return self._is_finished

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders the comprehensive Graph Theory & Pathfinding simulation."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._height

        # 1. Frosted Glass Container Card
        card_rect = QRectF(0.0, 0.0, w, h)
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Draw Graph Edges
        self._draw_edges(painter)

        # 3. Draw Graph Nodes
        self._draw_nodes(painter)

        # 4. Priority Queue / Open Set HUD (Top Right)
        self._draw_pq_hud(painter)

        # 5. Execution Log Badge (Bottom Center)
        self._draw_log_hud(painter)

        # 6. Control & Preset Buttons
        self._draw_buttons(painter)

        painter.restore()

    def _draw_edges(self, painter: QPainter) -> None:
        """Draws all weighted edges with weight badges and path glow."""
        r_node = 18.0
        f_weight = QFont("Segoe UI", 8)
        f_weight.setBold(True)
        painter.setFont(f_weight)

        for edge in self._edges:
            u_node = self._nodes.get(edge.u)
            v_node = self._nodes.get(edge.v)
            if not u_node or not v_node:
                continue

            p1 = u_node.pos
            p2 = v_node.pos
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                continue

            ux, uy = dx / dist, dy / dist
            sx = p1.x() + ux * r_node
            sy = p1.y() + uy * r_node
            ex = p2.x() - ux * r_node
            ey = p2.y() - uy * r_node

            if edge.is_path:
                # Golden Neon Glow
                pen_glow = QPen(QColor(251, 191, 36, 90), 6.0, Qt.PenStyle.SolidLine)
                painter.setPen(pen_glow)
                painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

                pen_core = QPen(QColor("#fbbf24"), 3.0, Qt.PenStyle.SolidLine)
                painter.setPen(pen_core)
                painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))
            else:
                pen_edge = QPen(QColor("#475569"), 1.8, Qt.PenStyle.SolidLine)
                painter.setPen(pen_edge)
                painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

            # Weight badge at midpoint
            mx = (sx + ex) / 2.0
            my = (sy + ey) / 2.0
            badge_rect = QRectF(mx - 11.0, my - 9.0, 22.0, 18.0)
            painter.setPen(QPen(QColor(71, 85, 105, 180), 1.0))
            painter.setBrush(QBrush(QColor("#0f172a")))
            painter.drawRoundedRect(badge_rect, 4.0, 4.0)

            painter.setPen(QPen(QColor("#fbbf24" if edge.is_path else "#94a3b8")))
            w_str = f"{edge.weight:.0f}" if edge.weight == int(edge.weight) else f"{edge.weight:.1f}"
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, w_str)

    def _draw_nodes(self, painter: QPainter) -> None:
        """Renders vertices with dynamic color-coding and distance values."""
        r_node = 18.0

        for nid, node in self._nodes.items():
            pos = node.pos

            # Determine colors based on state
            if node.state == "PATH":
                border_col = QColor("#fbbf24")
                bg_col = QColor(251, 191, 36, 60)
                txt_col = QColor("#ffffff")
                glow_col = QColor(251, 191, 36, 120)
            elif node.state == "CURRENT":
                border_col = QColor("#f59e0b")
                bg_col = QColor(245, 158, 11, 70)
                txt_col = QColor("#fef08a")
                glow_col = QColor(245, 158, 11, 140)
            elif node.state == "CLOSED":
                border_col = QColor("#10b981")
                bg_col = QColor(16, 185, 129, 50)
                txt_col = QColor("#6ee7b7")
                glow_col = None
            elif node.state == "OPEN":
                border_col = QColor("#06b6d4")
                bg_col = QColor(6, 182, 212, 50)
                txt_col = QColor("#67e8f9")
                glow_col = QColor(6, 182, 212, 80)
            else:  # UNVISITED
                border_col = QColor("#64748b")
                bg_col = QColor(30, 41, 59, 200)
                txt_col = QColor("#cbd5e1")
                glow_col = None

            # Glow aura if active
            if glow_col is not None:
                grad_glow = QRadialGradient(pos.x(), pos.y(), r_node * 1.8)
                grad_glow.setColorAt(0.0, glow_col)
                grad_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(grad_glow))
                painter.drawEllipse(pos, r_node * 1.8, r_node * 1.8)

            # Node Core Circle
            painter.setPen(QPen(border_col, 2.2))
            painter.setBrush(QBrush(bg_col))
            painter.drawEllipse(pos, r_node, r_node)

            # Node ID Label
            f_id = QFont("Segoe UI", 11)
            f_id.setBold(True)
            painter.setFont(f_id)
            painter.setPen(QPen(txt_col))
            painter.drawText(QRectF(pos.x() - r_node, pos.y() - r_node, r_node * 2, r_node * 2), Qt.AlignmentFlag.AlignCenter, nid)

            # Cost Badge (Above Node): d=... or f=...
            f_cost = QFont("Segoe UI", 7)
            painter.setFont(f_cost)
            painter.setPen(QPen(QColor("#94a3b8")))
            if node.cost_g == float("inf"):
                cost_text = "∞"
            elif self._algorithm == "ASTAR":
                cost_text = f"f={node.cost_f:.1f}"
            else:
                cost_text = f"d={node.cost_g:.1f}"
            painter.drawText(QRectF(pos.x() - 30.0, pos.y() - r_node - 15.0, 60.0, 14.0), Qt.AlignmentFlag.AlignCenter, cost_text)

            # Start / Target Badges (Below Node)
            if nid == self._start_node:
                painter.setPen(QPen(QColor("#38bdf8")))
                painter.drawText(QRectF(pos.x() - 30.0, pos.y() + r_node + 2.0, 60.0, 14.0), Qt.AlignmentFlag.AlignCenter, "START")
            elif nid == self._target_node:
                painter.setPen(QPen(QColor("#f43f5e")))
                painter.drawText(QRectF(pos.x() - 30.0, pos.y() + r_node + 2.0, 60.0, 14.0), Qt.AlignmentFlag.AlignCenter, "GOAL")

    def _draw_pq_hud(self, painter: QPainter) -> None:
        """Renders Open Set / Priority Queue Min-Heap card."""
        hud_w = 175.0
        hud_h = 240.0
        hud_x = self._width - hud_w - 18.0
        hud_y = 48.0
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        f_title = QFont("Segoe UI", 9)
        f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QPen(QColor("#38bdf8")))
        algo_title = "Dijkstra Min-Heap" if self._algorithm == "DIJKSTRA" else "A* Open Set [f=g+h]"
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 8.0, hud_w - 16.0, 18.0), algo_title)

        f_item = QFont("Segoe UI", 8)
        painter.setFont(f_item)

        # List elements in heap sorted by priority
        sorted_open = sorted(self._open_pq, key=lambda x: x[0])
        y_cursor = hud_y + 32.0

        if not sorted_open:
            painter.setPen(QPen(QColor("#64748b")))
            painter.drawText(QRectF(hud_x + 8.0, y_cursor, hud_w - 16.0, 16.0), "Hàng đợi rỗng (Ø)")
        else:
            for idx, (prio, _, nid) in enumerate(sorted_open[:8]):
                node = self._nodes.get(nid)
                if not node:
                    continue
                painter.setPen(QPen(QColor("#67e8f9") if idx == 0 else QColor("#cbd5e1")))
                if self._algorithm == "ASTAR":
                    txt_line = f"{idx+1}. [{nid}] f={prio:.1f} (g={node.cost_g:.1f}+h={node.cost_h:.1f})"
                else:
                    txt_line = f"{idx+1}. [{nid}] d = {prio:.1f}"
                painter.drawText(QRectF(hud_x + 8.0, y_cursor, hud_w - 16.0, 16.0), txt_line)
                y_cursor += 18.0

        # Visited count
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_x + 8.0, hud_y + hud_h - 22.0, hud_w - 16.0, 16.0),
            f"Đã duyệt: {len(self._closed_set)}/{len(self._nodes)} đỉnh",
        )

    def _draw_log_hud(self, painter: QPainter) -> None:
        """Renders live execution status bar at the bottom."""
        log_x = 18.0
        log_y = self._height - 70.0
        log_w = self._width - 36.0
        log_h = 26.0
        log_rect = QRectF(log_x, log_y, log_w, log_h)

        painter.setPen(QPen(QColor(51, 65, 85, 180), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
        painter.drawRoundedRect(log_rect, 6.0, 6.0)

        f_log = QFont("Segoe UI", 8)
        painter.setFont(f_log)
        painter.setPen(QPen(QColor("#38bdf8") if self._is_finished else QColor("#f1f5f9")))
        painter.drawText(QRectF(log_x + 10.0, log_y + 4.0, log_w - 20.0, 18.0), self._log_message)

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders header controls and preset selectors."""
        f_btn = QFont("Segoe UI", 8)
        f_btn.setBold(True)
        painter.setFont(f_btn)

        # Close button [✕]
        painter.setPen(QPen(QColor("#ef4444"), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68, 40)))
        painter.drawRoundedRect(self._rect_btn_close, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ef4444")))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # Stamp button [📌 Đóng dấu]
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor(56, 189, 248, 40)))
        painter.drawRoundedRect(self._rect_btn_stamp, 6.0, 6.0)
        painter.setPen(QPen(QColor("#e0f2fe")))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # Play / Pause button
        play_txt = "⏸ Tạm dừng" if self._is_playing else "▶ Chạy tự động"
        play_col = QColor("#f59e0b") if self._is_playing else QColor("#10b981")
        painter.setPen(QPen(play_col, 1.2))
        painter.setBrush(QBrush(QColor(play_col.red(), play_col.green(), play_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_play, 6.0, 6.0)
        painter.setPen(QPen(play_col))
        painter.drawText(self._rect_btn_play, Qt.AlignmentFlag.AlignCenter, play_txt)

        # Step button [⏭ Từng bước]
        painter.setPen(QPen(QColor("#06b6d4"), 1.2))
        painter.setBrush(QBrush(QColor(6, 182, 212, 40)))
        painter.drawRoundedRect(self._rect_btn_step, 6.0, 6.0)
        painter.setPen(QPen(QColor("#06b6d4")))
        painter.drawText(self._rect_btn_step, Qt.AlignmentFlag.AlignCenter, "⏭ Từng bước")

        # Reset button [↺ Đặt lại]
        painter.setPen(QPen(QColor("#94a3b8"), 1.2))
        painter.setBrush(QBrush(QColor(148, 163, 184, 30)))
        painter.drawRoundedRect(self._rect_btn_reset, 6.0, 6.0)
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(self._rect_btn_reset, Qt.AlignmentFlag.AlignCenter, "↺ Đặt lại")

        # Algorithm Mode button [Dijkstra ↔ A*]
        algo_txt = "Thuật toán: A*" if self._algorithm == "ASTAR" else "Thuật toán: Dijkstra"
        painter.setPen(QPen(QColor("#8b5cf6"), 1.2))
        painter.setBrush(QBrush(QColor(139, 92, 246, 40)))
        painter.drawRoundedRect(self._rect_btn_algo, 6.0, 6.0)
        painter.setPen(QPen(QColor("#c4b5fd")))
        painter.drawText(self._rect_btn_algo, Qt.AlignmentFlag.AlignCenter, algo_txt)

        # Preset selector buttons across bottom left
        for i, key in enumerate(self._preset_keys):
            rect = self._rect_btn_presets[i]
            is_active = (key == self._active_preset)
            p_data = GRAPH_PRESETS.get(key, {})
            name = p_data.get("name", key)

            if is_active:
                painter.setPen(QPen(QColor("#38bdf8"), 1.4))
                painter.setBrush(QBrush(QColor(56, 189, 248, 110)))
                painter.setPen(QPen(QColor("#ffffff")))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
                painter.setPen(QPen(QColor("#94a3b8")))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, name)

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles button clicks or node dragging."""
        pos = event.pos()

        # Check buttons
        if self._rect_btn_close.contains(pos):
            self.close()
            event.accept()
            return

        if self._rect_btn_play.contains(pos):
            self.toggle_auto_play()
            event.accept()
            return

        if self._rect_btn_step.contains(pos):
            self.step_algorithm()
            event.accept()
            return

        if self._rect_btn_reset.contains(pos):
            self.reset_search()
            event.accept()
            return

        if self._rect_btn_algo.contains(pos):
            new_algo = "ASTAR" if self._algorithm == "DIJKSTRA" else "DIJKSTRA"
            self.set_algorithm(new_algo)
            event.accept()
            return

        # Check presets
        for i, rect in enumerate(self._rect_btn_presets):
            if rect.contains(pos):
                self.set_preset(self._preset_keys[i])
                event.accept()
                return

        # Check stamp button
        if self._rect_btn_stamp.contains(pos):
            scene = self.scene()
            if scene is not None:
                board = getattr(scene, "board", None)
                if board is None and hasattr(scene, "views") and scene.views():
                    view = scene.views()[0]
                    board = view if hasattr(view, "addItem") else getattr(view, "board", None)
                if board is not None:
                    self.stamp_to_canvas(board)
            event.accept()
            return

        # Check if clicking a graph node to drag it
        for nid, node in self._nodes.items():
            if (pos - node.pos).manhattanLength() < 24.0:
                self._active_drag = "NODE"
                self._dragged_node_id = nid
                event.accept()
                return

        # Background drag
        self._active_drag = "BODY"
        self._drag_start_item_pos = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles vertex repositioning during dragging."""
        if self._active_drag == "NODE" and self._dragged_node_id:
            node = self._nodes.get(self._dragged_node_id)
            if node:
                # Clamp within item boundary
                new_x = max(35.0, min(self._width - 200.0, event.pos().x()))
                new_y = max(55.0, min(self._height - 90.0, event.pos().y()))
                node.pos = QPointF(new_x, new_y)
                # Recalculate heuristic if in A* mode
                if self._algorithm == "ASTAR" and self._target_node in self._nodes:
                    target_pt = self._nodes[self._target_node].pos
                    dist = math.hypot(node.pos.x() - target_pt.x(), node.pos.y() - target_pt.y())
                    node.cost_h = round(dist / 25.0, 1)
                self.update()
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Releases drag state."""
        self._active_drag = None
        self._dragged_node_id = None
        super().mouseReleaseEvent(event)

    # ─── Touchless Air Gesture Integration ───────────────────────────────────

    def set_air_hover_pos(self, scene_pos: Optional[QPointF]) -> None:
        """Sets touchless air pointer hover position."""
        if scene_pos is not None:
            self._air_hover_pos = self.mapFromScene(scene_pos)
        else:
            self._air_hover_pos = None
        self.update()

    def handle_air_drag(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch drag of nodes or entire card."""
        pos = self.mapFromScene(scene_pos)

        if self._active_drag is None:
            # Check if near any node
            for nid, node in self._nodes.items():
                if (pos - node.pos).manhattanLength() < 30.0:
                    self._active_drag = "NODE"
                    self._dragged_node_id = nid
                    break
            if self._active_drag is None:
                self._active_drag = "BODY"
                self._drag_start_item_pos = pos

        if self._active_drag == "NODE" and self._dragged_node_id:
            node = self._nodes.get(self._dragged_node_id)
            if node:
                new_x = max(35.0, min(self._width - 200.0, pos.x()))
                new_y = max(55.0, min(self._height - 90.0, pos.y()))
                node.pos = QPointF(new_x, new_y)
                self.update()
                return True

        if self._active_drag == "BODY":
            new_pos = scene_pos - self._drag_start_item_pos
            self.setPos(new_pos)
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Touchless air tap targeting control buttons."""
        pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(pos):
            self.close()
            return True

        if self._rect_btn_play.contains(pos):
            self.toggle_auto_play()
            return True

        if self._rect_btn_step.contains(pos):
            self.step_algorithm()
            return True

        if self._rect_btn_reset.contains(pos):
            self.reset_search()
            return True

        if self._rect_btn_algo.contains(pos):
            new_algo = "ASTAR" if self._algorithm == "DIJKSTRA" else "DIJKSTRA"
            self.set_algorithm(new_algo)
            return True

        for i, rect in enumerate(self._rect_btn_presets):
            if rect.contains(pos):
                self.set_preset(self._preset_keys[i])
                return True

        if self._rect_btn_stamp.contains(pos):
            target_board = board
            if target_board is None and self.scene() is not None:
                scene = self.scene()
                target_board = getattr(scene, "board", None)
            if target_board is not None:
                self.stamp_to_canvas(target_board)
                return True

        return False

    def close(self) -> None:
        """Stops timer and removes this interactive item from its scene."""
        self._timer.stop()
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

    # ─── Stamping onto LessonBoard ───────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """
        Permanently stamps vector graph, edge weights, shortest path,
        and algorithm summary HUD onto LessonBoard.
        """
        from src.ui.stem_tools import InformaticsTools
        from PySide6.QtWidgets import QGraphicsTextItem

        stamped: list[QGraphicsItem] = []
        origin = self.pos()

        # Build absolute scene node coordinates
        scene_nodes: dict[str, tuple[float, float]] = {}
        for nid, node in self._nodes.items():
            scene_nodes[nid] = (origin.x() + node.pos.x(), origin.y() + node.pos.y())

        # Build edge tuples: (u, v, weight)
        scene_edges: list[tuple[str, str, float]] = []
        for e in self._edges:
            scene_edges.append((e.u, e.v, e.weight))

        # Stamp vector graph diagram
        graph_items = InformaticsTools.create_graph_diagram(
            nodes=scene_nodes,
            edges=scene_edges,
            shortest_path=self._shortest_path if self._is_finished else None,
            is_directed=False,
        )
        for itm in graph_items:
            board.addItem(itm)
            stamped.append(itm)

        # Algorithm Summary Text Card
        algo_name = "Dijkstra" if self._algorithm == "DIJKSTRA" else "A* Search"
        path_str = " -> ".join(self._shortest_path) if self._shortest_path else "Chưa hoàn tất"
        lbl_text = (
            f"Lý Thuyết Đồ Thị & Đường Đi Ngắn Nhất ({algo_name}):\n"
            f"- Đỉnh xuất phát: [{self._start_node}] -> Đích đến: [{self._target_node}]\n"
            f"- Lộ trình tối ưu: {path_str}\n"
            f"- Chi phí đường đi: {self._total_cost:.1f}\n"
            f"- Nguyên lý: {'Tham lam g(n) với Hàng đợi ưu tiên Min-Heap' if self._algorithm == 'DIJKSTRA' else 'Hàm đánh giá f(n) = g(n) + h(n) với khoảng cách Euclid'}"
        )
        txt_item = QGraphicsTextItem(lbl_text)
        txt_font = QFont("Segoe UI", 10)
        txt_font.setBold(True)
        txt_item.setFont(txt_font)
        txt_item.setDefaultTextColor(QColor("#f8fafc"))
        txt_item.setPos(origin.x() + 20.0, origin.y() + self._height + 15.0)
        txt_item.setZValue(12.5)
        board.addItem(txt_item)
        stamped.append(txt_item)

        return stamped
