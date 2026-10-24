"""
AirTeach AI - Interactive Informatics: Minimum Spanning Tree (MST) Simulator
=============================================================================
Visualizes fundamental greedy graph algorithms:
  1. Kruskal's Algorithm with Disjoint Set Union (DSU / Union-Find) & Path Compression.
  2. Prim's Algorithm with Priority Cut & Growing Tree.
Features step-by-step air-gesture / click stepping, cycle rejection visualization,
connected component tracking, and real-time MST weight accumulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsSceneMouseEvent


@dataclass
class GraphNode:
    id: int
    name: str
    x: float
    y: float


@dataclass
class GraphEdge:
    u: int
    v: int
    weight: int
    id: int


@dataclass(frozen=True)
class GraphPreset:
    name: str
    description: str
    nodes: List[Tuple[int, str, float, float]]  # (id, name, x, y)
    edges: List[Tuple[int, int, int]]  # (u, v, weight)


GRAPH_PRESETS: Dict[str, GraphPreset] = {
    "STANDARD_NETWORK": GraphPreset(
        name="Mạng Chuẩn Sách Giáo Khoa (7 Đỉnh, 11 Cạnh)",
        description="Đồ thị mẫu kinh điển minh họa Kruskal (DSU) và Prim (Vết cắt nhỏ nhất) tránh chu trình.",
        nodes=[
            (0, "A", -180.0, -60.0),
            (1, "B", -80.0, -120.0),
            (2, "C", 60.0, -120.0),
            (3, "D", 160.0, -60.0),
            (4, "E", 110.0, 60.0),
            (5, "F", -30.0, 70.0),
            (6, "G", -140.0, 50.0),
        ],
        edges=[
            (0, 1, 7),
            (0, 6, 5),
            (1, 2, 8),
            (1, 6, 9),
            (1, 5, 7),
            (2, 3, 5),
            (2, 4, 5),
            (3, 4, 15),
            (4, 5, 8),
            (4, 6, 6),
            (5, 6, 9),
        ],
    ),
    "COMPLETE_K5": GraphPreset(
        name="Đồ Thị Đầy Đủ K₅ (5 Đỉnh, 10 Cạnh)",
        description="Đồ thị dày đặc mọi đỉnh nối nhau; kiểm tra hiệu quả loại bỏ cạnh tạo chu trình khép kín.",
        nodes=[
            (0, "A", 0.0, -120.0),
            (1, "B", 120.0, -30.0),
            (2, "C", 75.0, 95.0),
            (3, "D", -75.0, 95.0),
            (4, "E", -120.0, -30.0),
        ],
        edges=[
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 1),
            (3, 4, 4),
            (4, 0, 5),
            (0, 2, 6),
            (0, 3, 7),
            (1, 3, 8),
            (1, 4, 3),
            (2, 4, 2),
        ],
    ),
    "METROPOLITAN_FIBER": GraphPreset(
        name="Quy Hoạch Tuyến Cáp Quang Đô Thị (6 Điểm)",
        description="Tối ưu chi phí kết nối 6 trung tâm dữ liệu với tổng chiều dài cáp nhỏ nhất không vòng lặp.",
        nodes=[
            (0, "DC1", -160.0, -90.0),
            (1, "DC2", 0.0, -110.0),
            (2, "DC3", 160.0, -80.0),
            (3, "DC4", 130.0, 70.0),
            (4, "DC5", -10.0, 85.0),
            (5, "DC6", -150.0, 60.0),
        ],
        edges=[
            (0, 1, 4),
            (1, 2, 6),
            (2, 3, 8),
            (3, 4, 9),
            (4, 5, 10),
            (5, 0, 12),
            (0, 4, 7),
            (1, 4, 2),
            (1, 3, 3),
        ],
    ),
}


class DisjointSetUnion:
    """Disjoint Set Union (DSU) with Path Compression and Union by Rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1
            return True
        return False


class InteractiveMstVisualizerItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Minimum Spanning Tree algorithms
    (Kruskal and Prim) with step-by-step visualization and cycle detection.
    """

    stamp_requested = Signal(dict)
    step_changed = Signal(dict)

    def __init__(
        self,
        preset_key: str = "STANDARD_NETWORK",
        algorithm: str = "KRUSKAL",
        pos: Optional[Tuple[float, float]] = None,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        if pos is not None:
            self.setPos(pos[0], pos[1])

        self._active_preset_key = preset_key if preset_key in GRAPH_PRESETS else "STANDARD_NETWORK"
        self._algorithm = algorithm  # "KRUSKAL" or "PRIM"

        self._width = 720.0
        self._height = 440.0

        # Graph Elements
        self._nodes: List[GraphNode] = []
        self._edges: List[GraphEdge] = []

        # Algorithm State
        self._current_step_idx = 0
        self._accepted_edge_ids: Set[int] = set()
        self._rejected_edge_ids: Set[int] = set()
        self._current_edge_id: Optional[int] = None
        self._mst_total_weight = 0
        self._step_logs: List[str] = []
        self._is_completed = False

        # UI interaction rects
        self._next_btn_rect = QRectF(0, 0, 0, 0)
        self._reset_btn_rect = QRectF(0, 0, 0, 0)
        self._algo_btn_rect = QRectF(0, 0, 0, 0)
        self._stamp_btn_rect = QRectF(0, 0, 0, 0)
        self._preset_btn_rects: Dict[str, QRectF] = {}

        self.apply_preset(self._active_preset_key)

    def boundingRect(self) -> QRectF:
        return QRectF(-self._width / 2, -self._height / 2, self._width, self._height)

    # ── Graph and Algorithm Setup ───────────────────────────────────────────

    def apply_preset(self, preset_key: str) -> None:
        if preset_key in GRAPH_PRESETS:
            self._active_preset_key = preset_key
            preset = GRAPH_PRESETS[preset_key]

            self._nodes = [
                GraphNode(id=nid, name=nname, x=nx, y=ny)
                for nid, nname, nx, ny in preset.nodes
            ]
            self._edges = [
                GraphEdge(u=u, v=v, weight=w, id=i)
                for i, (u, v, w) in enumerate(preset.edges)
            ]

            self.reset_algorithm()

    def reset_algorithm(self) -> None:
        self._current_step_idx = 0
        self._accepted_edge_ids.clear()
        self._rejected_edge_ids.clear()
        self._current_edge_id = None
        self._mst_total_weight = 0
        self._is_completed = False
        self._step_logs.clear()

        # Sort edges for Kruskal
        if self._algorithm == "KRUSKAL":
            self._edges.sort(key=lambda e: e.weight)
            self._step_logs.append("Sắp xếp tất cả các cạnh theo thứ tự trọng số tăng dần.")

        self.update()

    def toggle_algorithm(self) -> None:
        self._algorithm = "PRIM" if self._algorithm == "KRUSKAL" else "KRUSKAL"
        self.reset_algorithm()

    def step_forward(self) -> None:
        """Executes one step of Kruskal or Prim MST algorithm."""
        if self._is_completed:
            return

        if self._algorithm == "KRUSKAL":
            self._step_kruskal()
        else:
            self._step_prim()

        self.step_changed.emit(self.get_state_summary())
        self.update()

    def _step_kruskal(self) -> None:
        n = len(self._nodes)
        dsu = DisjointSetUnion(n)

        # Re-apply currently accepted edges into DSU
        for e in self._edges:
            if e.id in self._accepted_edge_ids:
                dsu.union(e.u, e.v)

        if self._current_step_idx >= len(self._edges):
            self._is_completed = True
            self._current_edge_id = None
            self._step_logs.append(f"🎉 Hoàn thành toàn bộ đồ thị! Tổng trọng số MST = {self._mst_total_weight}")
            return

        edge = self._edges[self._current_step_idx]
        self._current_edge_id = edge.id
        u_name = self._nodes[edge.u].name
        v_name = self._nodes[edge.v].name

        root_u = dsu.find(edge.u)
        root_v = dsu.find(edge.v)

        if root_u != root_v and len(self._accepted_edge_ids) < n - 1:
            # Accept edge
            self._accepted_edge_ids.add(edge.id)
            self._mst_total_weight += edge.weight
            self._step_logs.append(
                f"Bước {self._current_step_idx + 1}: Chọn ({u_name}, {v_name}) w={edge.weight} -> Nhận vào MST (Hợp nhất tập hợp)."
            )
            if len(self._accepted_edge_ids) == n - 1:
                self._step_logs.append(f"🌟 Đã đủ {n - 1} cạnh MST! Các cạnh còn lại sẽ tạo chu trình.")
        else:
            # Reject edge (cycle)
            self._rejected_edge_ids.add(edge.id)
            self._step_logs.append(
                f"Bước {self._current_step_idx + 1}: Cạnh ({u_name}, {v_name}) w={edge.weight} -> BỎ QUA (Tạo chu trình khép kín!)."
            )

        self._current_step_idx += 1
        if self._current_step_idx >= len(self._edges):
            self._is_completed = True
            self._current_edge_id = None
            self._step_logs.append(f"🎉 Kết thúc duyệt tất cả {len(self._edges)} cạnh. Tổng trọng số MST = {self._mst_total_weight}")

    def _step_prim(self) -> None:
        n = len(self._nodes)
        if len(self._accepted_edge_ids) == 0:
            visited_nodes: Set[int] = {0}  # Start at node 0
        else:
            visited_nodes = set()
            for eid in self._accepted_edge_ids:
                e = next(x for x in self._edges if x.id == eid)
                visited_nodes.add(e.u)
                visited_nodes.add(e.v)

        if len(visited_nodes) >= n or len(self._accepted_edge_ids) >= n - 1:
            self._is_completed = True
            self._current_edge_id = None
            self._step_logs.append(f"🎉 Hoàn thành Prim MST! Tổng trọng số: {self._mst_total_weight}")
            return

        # Find minimum weight cut edge connecting visited to unvisited
        candidate_edges = [
            e
            for e in self._edges
            if (e.u in visited_nodes and e.v not in visited_nodes)
            or (e.v in visited_nodes and e.u not in visited_nodes)
        ]

        if not candidate_edges:
            self._is_completed = True
            return

        min_edge = min(candidate_edges, key=lambda e: e.weight)
        self._current_edge_id = min_edge.id
        self._accepted_edge_ids.add(min_edge.id)
        self._mst_total_weight += min_edge.weight

        u_name = self._nodes[min_edge.u].name
        v_name = self._nodes[min_edge.v].name
        self._current_step_idx += 1
        self._step_logs.append(
            f"Prim bước {self._current_step_idx}: Vết cắt nhỏ nhất chọn ({u_name}, {v_name}) w={min_edge.weight}."
        )

        if len(self._accepted_edge_ids) == n - 1:
            self._is_completed = True
            self._step_logs.append(f"🎉 Cây khung Prim hoàn thành! Tổng trọng số MST = {self._mst_total_weight}")

    def get_state_summary(self) -> Dict[str, any]:
        return {
            "algorithm": self._algorithm,
            "step": self._current_step_idx,
            "mst_weight": self._mst_total_weight,
            "accepted_count": len(self._accepted_edge_ids),
            "rejected_count": len(self._rejected_edge_ids),
            "is_completed": self._is_completed,
            "last_log": self._step_logs[-1] if self._step_logs else "",
        }

    def request_stamp(self) -> None:
        preset = GRAPH_PRESETS[self._active_preset_key]
        data = {
            "center": (self.x(), self.y()),
            "preset_key": self._active_preset_key,
            "preset_name": preset.name,
            "description": preset.description,
            "algorithm": self._algorithm,
            "mst_weight": self._mst_total_weight,
            "num_nodes": len(self._nodes),
            "num_edges": len(self._edges),
            "accepted_edges": [
                (self._nodes[e.u].name, self._nodes[e.v].name, e.weight)
                for e in self._edges
                if e.id in self._accepted_edge_ids
            ],
            "rejected_edges": [
                (self._nodes[e.u].name, self._nodes[e.v].name, e.weight)
                for e in self._edges
                if e.id in self._rejected_edge_ids
            ],
        }
        self.stamp_requested.emit(data)

    # ── Rendering ────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        preset = GRAPH_PRESETS[self._active_preset_key]
        hw = self._width / 2.0
        hh = self._height / 2.0

        # 1. Outer Container
        bg_grad = QLinearGradient(-hw, -hh, -hw, hh)
        bg_grad.setColorAt(0.0, QColor(10, 15, 29))
        bg_grad.setColorAt(1.0, QColor(6, 10, 20))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(30, 48, 75), 1.5))
        painter.drawRoundedRect(-hw, -hh, self._width, self._height, 10.0, 10.0)

        # 2. Header
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        algo_title = "THUẬT TOÁN KRUSKAL (DSU)" if self._algorithm == "KRUSKAL" else "THUẬT TOÁN PRIM (VẾT CẮT)"
        painter.drawText(
            QRectF(-hw + 20, -hh + 12, self._width - 40, 22),
            Qt.AlignLeft,
            f"🌲 CÂY KHUNG NHỎ NHẤT (MST): {algo_title} - {preset.name.upper()}",
        )

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(-hw + 20, -hh + 34, self._width - 40, 18),
            Qt.AlignLeft,
            f"Số đỉnh: |V| = {len(self._nodes)} | Số cạnh: |E| = {len(self._edges)} | Đã chọn: {len(self._accepted_edge_ids)}/{len(self._nodes)-1} cạnh | Tổng trọng số MST = {self._mst_total_weight}",
        )

        # 3. Preset Tabs
        self._draw_preset_tabs(painter, -hw + 20, -hh + 54)

        # 4. Left Graph Canvas
        graph_rect = QRectF(-hw + 20, -hh + 84, 430, 240)
        self._draw_graph_canvas(painter, graph_rect)

        # 5. Right Status & Execution Log Panel
        info_rect = QRectF(graph_rect.right() + 14, -hh + 84, self._width - graph_rect.width() - 54, 240)
        self._draw_info_panel(painter, info_rect, preset)

        # 6. Bottom Controls: Next step, Reset, Algorithm toggle, Stamp button
        controls_rect = QRectF(-hw + 20, hh - 100, self._width - 40, 88)
        self._draw_controls(painter, controls_rect)

    def _draw_preset_tabs(self, painter: QPainter, x: float, y: float) -> None:
        self._preset_btn_rects.clear()
        tab_w = 210.0
        tab_h = 22.0

        keys = list(GRAPH_PRESETS.keys())
        for i, key in enumerate(keys):
            preset = GRAPH_PRESETS[key]
            rect = QRectF(x + i * (tab_w + 5), y, tab_w, tab_h)
            self._preset_btn_rects[key] = rect

            is_active = key == self._active_preset_key
            if is_active:
                painter.setBrush(QBrush(QColor(14, 116, 144)))
                painter.setPen(QPen(QColor(56, 189, 248), 1.5))
            else:
                painter.setBrush(QBrush(QColor(18, 27, 44)))
                painter.setPen(QPen(QColor(36, 52, 78), 1.0))

            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(241, 245, 249) if is_active else QColor(148, 163, 184))
            short_name = preset.name.split(" (")[0]
            painter.drawText(rect, Qt.AlignCenter, short_name)

    def _draw_graph_canvas(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(24, 38, 60), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Center offset inside canvas
        cx = rect.center().x()
        cy = rect.center().y()

        # Draw Edges
        for e in self._edges:
            nu = self._nodes[e.u]
            nv = self._nodes[e.v]
            x1, y1 = cx + nu.x, cy + nu.y
            x2, y2 = cx + nv.x, cy + nv.y

            is_accepted = e.id in self._accepted_edge_ids
            is_rejected = e.id in self._rejected_edge_ids
            is_current = e.id == self._current_edge_id

            if is_accepted:
                edge_pen = QPen(QColor(34, 197, 94), 3.5, Qt.SolidLine, Qt.RoundCap)
            elif is_rejected:
                edge_pen = QPen(QColor(239, 68, 68, 160), 1.8, Qt.DashLine, Qt.RoundCap)
            elif is_current:
                edge_pen = QPen(QColor(250, 204, 21), 3.0, Qt.SolidLine, Qt.RoundCap)
            else:
                edge_pen = QPen(QColor(71, 85, 105, 180), 1.4, Qt.SolidLine, Qt.RoundCap)

            painter.setPen(edge_pen)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Weight Badge
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            badge_r = 10.0
            badge_rect = QRectF(mx - badge_r, my - badge_r, badge_r * 2, badge_r * 2)

            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.setPen(QPen(edge_pen.color(), 1.0))
            painter.drawEllipse(badge_rect)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(241, 245, 249))
            painter.drawText(badge_rect, Qt.AlignCenter, str(e.weight))

        # Draw Nodes
        node_radius = 16.0
        for node in self._nodes:
            nx = cx + node.x
            ny = cy + node.y
            n_rect = QRectF(nx - node_radius, ny - node_radius, node_radius * 2, node_radius * 2)

            # Node glow
            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.setPen(QPen(QColor(56, 189, 248), 2.0))
            painter.drawEllipse(n_rect)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(n_rect, Qt.AlignCenter, node.name)

    def _draw_info_panel(self, painter: QPainter, rect: QRectF, preset: GraphPreset) -> None:
        painter.setBrush(QBrush(QColor(8, 14, 26)))
        painter.setPen(QPen(QColor(24, 38, 60), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 8, rect.width() - 20, 16),
            Qt.AlignLeft,
            "📋 TIẾN TRÌNH THUẬT TOÁN",
        )

        n = len(self._nodes)
        rows = [
            ("Thuật toán đang chạy:", self._algorithm, QColor(250, 204, 21)),
            ("Số đỉnh (|V|):", str(n), QColor(203, 213, 225)),
            ("Cạnh mục tiêu (|V|-1):", str(n - 1), QColor(56, 189, 248)),
            ("Cạnh đã nhận (MST):", f"{len(self._accepted_edge_ids)} cạnh", QColor(34, 197, 94)),
            ("Cạnh bỏ qua (Chu trình):", f"{len(self._rejected_edge_ids)} cạnh", QColor(239, 68, 68)),
            ("TỔNG TRỌNG SỐ MST:", f"{self._mst_total_weight}", QColor(236, 72, 153)),
            ("Trạng thái:", "HOÀN THÀNH" if self._is_completed else f"Đang xét bước {self._current_step_idx}", QColor(34, 197, 94) if self._is_completed else QColor(250, 204, 21)),
        ]

        y = rect.top() + 26.0
        painter.setFont(QFont("Segoe UI", 7))
        for label, val, col in rows:
            painter.setPen(QColor(140, 160, 190))
            painter.drawText(QRectF(rect.left() + 8, y, 130, 13), Qt.AlignLeft, label)
            painter.setPen(col)
            painter.drawText(QRectF(rect.right() - 110, y, 102, 13), Qt.AlignRight, val)
            y += 15.0

        # Log box
        log_box = QRectF(rect.left() + 6, y + 4, rect.width() - 12, 85)
        painter.setBrush(QBrush(QColor(6, 10, 18)))
        painter.setPen(QPen(QColor(20, 32, 50), 1.0))
        painter.drawRoundedRect(log_box, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6.8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(log_box.left() + 6, log_box.top() + 4, log_box.width() - 12, 12), Qt.AlignLeft, "NHẬT KÝ THỰC THI (LOG):")

        painter.setFont(QFont("Segoe UI", 6.8))
        painter.setPen(QColor(203, 213, 225))

        recent_logs = self._step_logs[-4:] if self._step_logs else ["Sẵn sàng bắt đầu. Nhấn 'BƯỚC TIẾP' để duyệt từng cạnh."]
        ly = log_box.top() + 18.0
        for log in recent_logs:
            painter.drawText(QRectF(log_box.left() + 6, ly, log_box.width() - 12, 13), Qt.AlignLeft, f"• {log}")
            ly += 14.0

    def _draw_controls(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor(10, 16, 28)))
        painter.setPen(QPen(QColor(28, 44, 72), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # 1. Next Step Button
        btn_next = QRectF(rect.left() + 16, rect.top() + 12, 130, 28)
        self._next_btn_rect = btn_next
        if not self._is_completed:
            painter.setBrush(QBrush(QColor(16, 185, 129)))
            painter.setPen(QPen(QColor(52, 211, 153), 1.5))
        else:
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(btn_next, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(255, 255, 255) if not self._is_completed else QColor(148, 163, 184))
        painter.drawText(btn_next, Qt.AlignCenter, "▶ BƯỚC TIẾP (STEP)")

        # 2. Reset Button
        btn_reset = QRectF(btn_next.right() + 12, rect.top() + 12, 100, 28)
        self._reset_btn_rect = btn_reset
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(btn_reset, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(203, 213, 225))
        painter.drawText(btn_reset, Qt.AlignCenter, "🔄 ĐẶT LẠI")

        # 3. Toggle Algorithm Button (Kruskal <-> Prim)
        btn_algo = QRectF(btn_reset.right() + 12, rect.top() + 12, 150, 28)
        self._algo_btn_rect = btn_algo
        painter.setBrush(QBrush(QColor(14, 116, 144)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawRoundedRect(btn_algo, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(btn_algo, Qt.AlignCenter, f"🔀 ĐỔI: {self._algorithm}")

        # 4. Stamp Button
        btn_stamp = QRectF(rect.right() - 130, rect.top() + 12, 115, 28)
        self._stamp_btn_rect = btn_stamp
        painter.setBrush(QBrush(QColor(124, 58, 237)))
        painter.setPen(QPen(QColor(167, 139, 250), 1.5))
        painter.drawRoundedRect(btn_stamp, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(btn_stamp, Qt.AlignCenter, "📌 ĐÓNG DẤU MST")

        # Instruction Hint
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(100, 116, 139))
        painter.drawText(
            QRectF(rect.left() + 16, rect.top() + 50, rect.width() - 32, 20),
            Qt.AlignLeft,
            "💡 Nhấn 'BƯỚC TIẾP' để duyệt từng cạnh: Cạnh xanh = Nhận vào MST, Cạnh đỏ đứt đoạn = Bỏ qua do tạo chu trình (DSU).",
        )

    # ── Mouse Interaction ────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        for key, rect in self._preset_btn_rects.items():
            if rect.contains(pos):
                self.apply_preset(key)
                event.accept()
                return

        if self._next_btn_rect.contains(pos):
            self.step_forward()
            event.accept()
            return

        if self._reset_btn_rect.contains(pos):
            self.reset_algorithm()
            event.accept()
            return

        if self._algo_btn_rect.contains(pos):
            self.toggle_algorithm()
            event.accept()
            return

        if self._stamp_btn_rect.contains(pos):
            self.request_stamp()
            event.accept()
            return

        super().mousePressEvent(event)
