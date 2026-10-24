"""
AirTeach AI — Interactive Maximum Network Flow & Minimum Cut Simulator (Edmonds-Karp / Ford-Fulkerson)
Authentic discrete graph theory, residual capacity networks, BFS shortest augmenting paths,
bottleneck capacity detection, Max-Flow Min-Cut theorem, and touchless air gesture integration.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)


@dataclass(frozen=True)
class FlowEdge:
    u: str
    v: str
    capacity: int


@dataclass
class FlowNetworkPreset:
    code: str
    name_vi: str
    source: str
    sink: str
    nodes: dict[str, tuple[float, float]]  # Node id -> (norm_x, norm_y) in [0, 1]
    edges: list[FlowEdge]
    description: str


PRESETS: dict[str, FlowNetworkPreset] = {
    "CLASSIC_EDMONDS_KARP": FlowNetworkPreset(
        code="CLASSIC_EDMONDS_KARP",
        name_vi="Mạng kinh điển 6 đỉnh (Edmonds-Karp)",
        source="S",
        sink="T",
        nodes={
            "S": (0.08, 0.50),
            "A": (0.35, 0.20),
            "B": (0.35, 0.80),
            "C": (0.65, 0.20),
            "D": (0.65, 0.80),
            "T": (0.92, 0.50),
        },
        edges=[
            FlowEdge("S", "A", 16),
            FlowEdge("S", "B", 13),
            FlowEdge("A", "B", 10),
            FlowEdge("A", "C", 12),
            FlowEdge("B", "A", 4),
            FlowEdge("B", "D", 14),
            FlowEdge("C", "B", 9),
            FlowEdge("C", "T", 20),
            FlowEdge("D", "C", 7),
            FlowEdge("D", "T", 4),
        ],
        description="Mạng 6 đỉnh kinh điển chứng minh cạnh ngược trong đồ thị thặng dư giúp chuyển luồng và đạt luồng cực đại 23.",
    ),
    "BOTTLENECK_BRIDGE": FlowNetworkPreset(
        code="BOTTLENECK_BRIDGE",
        name_vi="Nghẽn cổ chai cầu nối (Bridge Bottleneck)",
        source="S",
        sink="T",
        nodes={
            "S": (0.10, 0.50),
            "A": (0.42, 0.22),
            "B": (0.42, 0.78),
            "C": (0.68, 0.50),
            "T": (0.90, 0.50),
        },
        edges=[
            FlowEdge("S", "A", 10),
            FlowEdge("S", "B", 10),
            FlowEdge("A", "B", 2),
            FlowEdge("A", "C", 8),
            FlowEdge("B", "C", 9),
            FlowEdge("C", "T", 15),
        ],
        description="Cổ chai C->T giới hạn toàn bộ luồng mạng; dung lượng lát cắt cực tiểu Min-Cut = 15.",
    ),
    "BIPARTITE_MATCHING": FlowNetworkPreset(
        code="BIPARTITE_MATCHING",
        name_vi="Cặp ghép cực đại đồ thị 2 phía (Bipartite Matching)",
        source="S",
        sink="T",
        nodes={
            "S": (0.08, 0.50),
            "U1": (0.35, 0.15),
            "U2": (0.35, 0.38),
            "U3": (0.35, 0.62),
            "U4": (0.35, 0.85),
            "V1": (0.65, 0.15),
            "V2": (0.65, 0.38),
            "V3": (0.65, 0.62),
            "V4": (0.65, 0.85),
            "T": (0.92, 0.50),
        },
        edges=[
            FlowEdge("S", "U1", 1),
            FlowEdge("S", "U2", 1),
            FlowEdge("S", "U3", 1),
            FlowEdge("S", "U4", 1),
            FlowEdge("U1", "V1", 1),
            FlowEdge("U1", "V2", 1),
            FlowEdge("U2", "V1", 1),
            FlowEdge("U2", "V3", 1),
            FlowEdge("U3", "V3", 1),
            FlowEdge("U3", "V4", 1),
            FlowEdge("U4", "V2", 1),
            FlowEdge("U4", "V4", 1),
            FlowEdge("V1", "T", 1),
            FlowEdge("V2", "T", 1),
            FlowEdge("V3", "T", 1),
            FlowEdge("V4", "T", 1),
        ],
        description="Quy bài toán ghép người-việc (Bipartite Matching) về luồng cực đại với dung lượng đơn vị; đạt cặp ghép hoàn hảo 4/4.",
    ),
    "MULTI_ROUTE_TELECOM": FlowNetworkPreset(
        code="MULTI_ROUTE_TELECOM",
        name_vi="Định tuyến mạng viễn thông đa kênh (Multi-Path)",
        source="S",
        sink="T",
        nodes={
            "S": (0.08, 0.50),
            "R1": (0.32, 0.18),
            "R2": (0.32, 0.50),
            "R3": (0.32, 0.82),
            "R4": (0.68, 0.25),
            "R5": (0.68, 0.75),
            "T": (0.92, 0.50),
        },
        edges=[
            FlowEdge("S", "R1", 18),
            FlowEdge("S", "R2", 12),
            FlowEdge("S", "R3", 15),
            FlowEdge("R1", "R4", 14),
            FlowEdge("R2", "R4", 8),
            FlowEdge("R2", "R5", 6),
            FlowEdge("R3", "R5", 16),
            FlowEdge("R4", "T", 20),
            FlowEdge("R5", "T", 18),
        ],
        description="Mạng cáp quang viễn thông phân tán luồng gói tin song song tối đa hóa băng thông truyền tải giữa 2 Data Center.",
    ),
    "DISJOINT_PATHS_CONNECTIVITY": FlowNetworkPreset(
        code="DISJOINT_PATHS_CONNECTIVITY",
        name_vi="Định lý Menger: Đường đi rời nhau (Disjoint Paths)",
        source="S",
        sink="T",
        nodes={
            "S": (0.10, 0.50),
            "A": (0.38, 0.22),
            "B": (0.38, 0.78),
            "C": (0.62, 0.22),
            "D": (0.62, 0.78),
            "T": (0.90, 0.50),
        },
        edges=[
            FlowEdge("S", "A", 1),
            FlowEdge("S", "B", 1),
            FlowEdge("A", "C", 1),
            FlowEdge("A", "D", 1),
            FlowEdge("B", "C", 1),
            FlowEdge("B", "D", 1),
            FlowEdge("C", "T", 1),
            FlowEdge("D", "T", 1),
        ],
        description="Định lý Menger (1927): Số đường đi rời nhau về cạnh giữa S và T bằng kích thước lát cắt nhỏ nhất (Min-Cut = 2).",
    ),
}


class InteractiveMaxFlowNetworkItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Maximum Network Flow and Minimum Cut
    using authentic Edmonds-Karp (BFS) algorithm and residual capacity networks.
    """

    WIDTH = 680.0
    HEIGHT = 410.0

    stamp_requested = Signal()
    step_completed = Signal(int, int)  # current_flow, max_flow

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (150.0, 65.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)

        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(QPointF(float(pos[0]), float(pos[1])))

        self.setZValue(25.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Simulation State
        self._active_preset_code: str = "CLASSIC_EDMONDS_KARP"
        self._preset: FlowNetworkPreset = PRESETS[self._active_preset_code]

        # Flow tracking: (u, v) -> current flow
        self._flows: dict[tuple[str, str], int] = {}
        self._total_flow: int = 0
        self._step_count: int = 0
        self._is_optimal: bool = False

        # Last Augmenting Path found by BFS: list of node ids [S, v1, v2, ..., T]
        self._last_path: list[str] = []
        self._last_bottleneck: int = 0

        # Min-Cut Sets (S_set and T_set)
        self._s_set: set[str] = set()
        self._t_set: set[str] = set()
        self._min_cut_capacity: int = 0
        self._cut_edges: list[tuple[str, str]] = []

        # Auto-run animation timer
        self._is_running: bool = False
        self._run_timer = QTimer()
        self._run_timer.setInterval(600)  # 600 ms per augmentation step
        self._run_timer.timeout.connect(self._on_timer_step)

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

        # Initialize network flow
        self.load_preset(self._active_preset_code)

    def _setup_layout(self) -> None:
        """Sets up layout rects for toolbar buttons, graph canvas, and diagnostics."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0

        # 5 Preset buttons
        preset_names = [
            ("CLASSIC_EDMONDS_KARP", "Edmonds-Karp"),
            ("BOTTLENECK_BRIDGE", "Nghẽn cầu"),
            ("BIPARTITE_MATCHING", "Ghép 2 phía"),
            ("MULTI_ROUTE_TELECOM", "Đa kênh"),
            ("DISJOINT_PATHS_CONNECTIVITY", "Định lý Menger"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 74.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 4.0

        # Controls
        r_step = QRectF(cur_x + 4.0, btn_y, 46.0, btn_h)
        self._btn_rects["step_bfs"] = r_step

        r_auto = QRectF(r_step.right() + 4.0, btn_y, 48.0, btn_h)
        self._btn_rects["toggle_auto"] = r_auto

        r_reset = QRectF(r_auto.right() + 4.0, btn_y, 36.0, btn_h)
        self._btn_rects["reset_flow"] = r_reset

        r_stamp = QRectF(r_reset.right() + 4.0, btn_y, 46.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        # Viewport areas
        self._graph_rect = QRectF(16.0, 78.0, 420.0, 205.0)
        self._diag_rect = QRectF(446.0, 78.0, 220.0, 205.0)

    def cleanup(self) -> None:
        """Stops auto-run timer to prevent leaks."""
        if self._run_timer.isActive():
            self._run_timer.stop()

    # -------------------------------------------------------------------------
    # Network Initialization & Reset
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined flow network preset and resets all flows to 0."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self.reset_flow()

    def reset_flow(self) -> None:
        """Resets all edge flows to 0 and clears augmenting path traces."""
        self._flows.clear()
        for edge in self._preset.edges:
            self._flows[(edge.u, edge.v)] = 0
            self._flows[(edge.v, edge.u)] = 0  # Skew symmetric initialization

        self._total_flow = 0
        self._step_count = 0
        self._is_optimal = False
        self._last_path.clear()
        self._last_bottleneck = 0
        self._s_set.clear()
        self._t_set.clear()
        self._cut_edges.clear()
        self._min_cut_capacity = 0
        self.update()

    # -------------------------------------------------------------------------
    # Edmonds-Karp BFS Algorithm (Authentic Discrete Mathematics)
    # -------------------------------------------------------------------------
    def _residual_capacity(self, u: str, v: str) -> int:
        """Returns residual capacity c_f(u, v) in residual network G_f."""
        # Find original capacity if (u, v) in original edges
        cap = 0
        for edge in self._preset.edges:
            if edge.u == u and edge.v == v:
                cap = edge.capacity
                break
        current_flow = self._flows.get((u, v), 0)
        return cap - current_flow

    def _find_augmenting_path_bfs(self) -> Optional[tuple[list[str], int]]:
        """
        Runs Breadth-First Search on residual graph G_f from source to sink.
        Returns (path_nodes, bottleneck_capacity) or None if no augmenting path exists.
        """
        src = self._preset.source
        snk = self._preset.sink

        # All node names in graph
        all_nodes = list(self._preset.nodes.keys())
        parent: dict[str, str] = {}
        visited: set[str] = {src}
        queue = collections.deque([src])

        while queue:
            curr = queue.popleft()
            if curr == snk:
                break

            for nxt in all_nodes:
                if nxt not in visited and self._residual_capacity(curr, nxt) > 0:
                    visited.add(nxt)
                    parent[nxt] = curr
                    queue.append(nxt)

        if snk not in parent:
            return None

        # Reconstruct path from sink to source
        path: list[str] = []
        curr = snk
        while curr != src:
            path.append(curr)
            curr = parent[curr]
        path.append(src)
        path.reverse()

        # Find bottleneck capacity along path
        bottleneck = min(self._residual_capacity(path[i], path[i + 1]) for i in range(len(path) - 1))
        return path, bottleneck

    def step_edmonds_karp(self) -> bool:
        """
        Executes one single augmenting step of the Edmonds-Karp algorithm.
        Returns True if an augmenting path was found and flow increased; False if optimal.
        """
        if self._is_optimal:
            return False

        res = self._find_augmenting_path_bfs()
        if res is not None:
            path, bottleneck = res
            self._last_path = path
            self._last_bottleneck = bottleneck

            # Augment flow along path
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                self._flows[(u, v)] = self._flows.get((u, v), 0) + bottleneck
                self._flows[(v, u)] = self._flows.get((v, u), 0) - bottleneck

            self._total_flow += bottleneck
            self._step_count += 1
            self.step_completed.emit(self._total_flow, self._total_flow)
            self.update()
            return True
        else:
            # Optimal max-flow reached! Compute S-T Min-Cut
            self._is_optimal = True
            self._last_path.clear()
            self._last_bottleneck = 0

            # Find reachable nodes from source in residual network G_f
            src = self._preset.source
            reachable: set[str] = {src}
            q = collections.deque([src])
            all_nodes = list(self._preset.nodes.keys())

            while q:
                curr = q.popleft()
                for nxt in all_nodes:
                    if nxt not in reachable and self._residual_capacity(curr, nxt) > 0:
                        reachable.add(nxt)
                        q.append(nxt)

            self._s_set = reachable
            self._t_set = set(all_nodes) - reachable

            # Identify cut edges from S to T
            self._cut_edges.clear()
            cut_cap = 0
            for edge in self._preset.edges:
                if edge.u in self._s_set and edge.v in self._t_set:
                    self._cut_edges.append((edge.u, edge.v))
                    cut_cap += edge.capacity

            self._min_cut_capacity = cut_cap

            # Stop timer if running
            if self._is_running:
                self.toggle_auto_run()

            self.step_completed.emit(self._total_flow, self._total_flow)
            self.update()
            return False

    def toggle_auto_run(self) -> bool:
        """Toggles continuous automatic step timer."""
        self._is_running = not self._is_running
        if self._is_running:
            self._run_timer.start()
        else:
            self._run_timer.stop()
        self.update()
        return self._is_running

    def _on_timer_step(self) -> None:
        """Timer callback to step forward."""
        if not self._is_running:
            return
        more = self.step_edmonds_karp()
        if not more:
            self.toggle_auto_run()

    # -------------------------------------------------------------------------
    # Properties & Diagnostics
    # -------------------------------------------------------------------------
    @property
    def total_flow(self) -> int:
        return self._total_flow

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def is_optimal(self) -> bool:
        return self._is_optimal

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def active_preset(self) -> FlowNetworkPreset:
        return self._preset

    @property
    def min_cut_capacity(self) -> int:
        return self._min_cut_capacity

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active Max Flow Network & Min-Cut diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import InformaticsTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = InformaticsTools.create_max_flow_diagram(
            center=center,
            preset_code=self._active_preset_code,
            total_flow=self._total_flow,
            is_optimal=self._is_optimal,
            min_cut_cap=self._min_cut_capacity,
        )
        for itm in items:
            scene.addItem(itm)
        self.stamp_requested.emit()

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Single step BFS
        r_step = self._btn_rects.get("step_bfs")
        if r_step and r_step.contains(local_pos):
            self.step_edmonds_karp()
            return True

        # 3. Toggle Auto-run
        r_auto = self._btn_rects.get("toggle_auto")
        if r_auto and r_auto.contains(local_pos):
            self.toggle_auto_run()
            return True

        # 4. Reset flow
        r_reset = self._btn_rects.get("reset_flow")
        if r_reset and r_reset.contains(local_pos):
            self.reset_flow()
            return True

        # 5. Stamp
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
            return True

        return True

    # -------------------------------------------------------------------------
    # Rendering & Painting
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if painter is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header Banner
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 36.0), 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 26.0, self.WIDTH, 10.0))

        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(14.0, 0.0, 480.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "🌊 LUỒNG CỰC ĐẠI TRÊN MẠNG & LÁT CẮT HẸP NHẤT (EDMONDS-KARP MAX-FLOW)",
        )

        # Preset Badge
        p_badge = QRectF(self.WIDTH - 164.0, 7.0, 150.0, 22.0)
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.setPen(QPen(QColor("#818cf8"), 1.0))
        painter.drawRoundedRect(p_badge, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#c7d2fe"))
        painter.drawText(p_badge, Qt.AlignmentFlag.AlignCenter, self._preset.name_vi[:24])

        # 3. Preset Buttons & Controls
        preset_items = [
            ("CLASSIC_EDMONDS_KARP", "Edmonds-Karp"),
            ("BOTTLENECK_BRIDGE", "Nghẽn cầu"),
            ("BIPARTITE_MATCHING", "Ghép 2 phía"),
            ("MULTI_ROUTE_TELECOM", "Đa kênh"),
            ("DISJOINT_PATHS_CONNECTIVITY", "Định lý Menger"),
        ]
        for p_code, p_lbl in preset_items:
            r = self._btn_rects[f"preset_{p_code}"]
            is_active = (p_code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#3b82f6") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#60a5fa") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, p_lbl)

        # Step BFS Button
        r_step = self._btn_rects["step_bfs"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(r_step, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_step, Qt.AlignmentFlag.AlignCenter, "➡ Bước")

        # Auto Run Button
        r_auto = self._btn_rects["toggle_auto"]
        painter.setBrush(QBrush(QColor("#065f46") if self._is_running else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#10b981") if self._is_running else QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_auto, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff") if self._is_running else QColor("#94a3b8"))
        painter.drawText(r_auto, Qt.AlignmentFlag.AlignCenter, "⏸ Dừng" if self._is_running else "▶ Tự động")

        # Reset Button
        r_rst = self._btn_rects["reset_flow"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(r_rst, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(r_rst, Qt.AlignmentFlag.AlignCenter, "Reset")

        # Stamp Button
        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#831843")))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # ---------------------------------------------------------------------
        # 4. Left Panel: Flow Network Topology & Live Capacities
        # ---------------------------------------------------------------------
        gx, gy, gw, gh = self._graph_rect.x(), self._graph_rect.y(), self._graph_rect.width(), self._graph_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._graph_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(gx + 8.0, gy + 16.0), "MẠNG LUỒNG & DUNG LƯỢNG f(u,v) / c(u,v)")

        # Compute pixel positions for all nodes
        node_pts: dict[str, QPointF] = {}
        for nid, (nx, ny) in self._preset.nodes.items():
            px = gx + 15.0 + nx * (gw - 30.0)
            py = gy + 22.0 + ny * (gh - 40.0)
            node_pts[nid] = QPointF(px, py)

        # Draw Directed Edges
        path_edges = set()
        if len(self._last_path) > 1:
            for i in range(len(self._last_path) - 1):
                path_edges.add((self._last_path[i], self._last_path[i + 1]))

        for edge in self._preset.edges:
            p1 = node_pts[edge.u]
            p2 = node_pts[edge.v]
            f_val = self._flows.get((edge.u, edge.v), 0)
            c_val = edge.capacity

            is_path = (edge.u, edge.v) in path_edges
            is_cut = (edge.u, edge.v) in self._cut_edges

            # Edge line
            if is_cut and self._is_optimal:
                e_pen = QPen(QColor("#ef4444"), 2.2, Qt.PenStyle.DashLine)
            elif is_path:
                e_pen = QPen(QColor("#fbbf24"), 2.5)  # Golden highlight
            elif f_val > 0:
                e_pen = QPen(QColor("#38bdf8"), 1.8)  # Active flow
            else:
                e_pen = QPen(QColor("#334155"), 1.0)

            painter.setPen(e_pen)
            painter.drawLine(p1, p2)

            # Arrowhead and flow text label
            mid_x = (p1.x() + p2.x()) / 2.0
            mid_y = (p1.y() + p2.y()) / 2.0
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            dist = math.hypot(dx, dy)
            if dist > 1e-4:
                # Small arrowhead
                ux = dx / dist
                uy = dy / dist
                arrow_p1 = QPointF(mid_x - ux * 5.0 - uy * 3.5, mid_y - uy * 5.0 + ux * 3.5)
                arrow_p2 = QPointF(mid_x - ux * 5.0 + uy * 3.5, mid_y - uy * 5.0 - ux * 3.5)
                painter.drawLine(QPointF(mid_x, mid_y), arrow_p1)
                painter.drawLine(QPointF(mid_x, mid_y), arrow_p2)

                # Flow text: f / c
                lbl_pos = QPointF(mid_x - uy * 9.0, mid_y + ux * 9.0)
                painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                txt_col = QColor("#fbbf24") if is_path else (QColor("#38bdf8") if f_val > 0 else QColor("#64748b"))
                painter.setPen(txt_col)
                painter.drawText(QRectF(lbl_pos.x() - 15.0, lbl_pos.y() - 6.0, 30.0, 12.0), Qt.AlignmentFlag.AlignCenter, f"{f_val}/{c_val}")

        # Draw Nodes
        for nid, pt in node_pts.items():
            # Node color
            if nid == self._preset.source:
                n_col = QColor("#10b981")  # Emerald Source
            elif nid == self._preset.sink:
                n_col = QColor("#f43f5e")  # Rose Sink
            elif self._is_optimal and nid in self._s_set:
                n_col = QColor("#059669")  # S-set partition
            elif self._is_optimal and nid in self._t_set:
                n_col = QColor("#e11d48")  # T-set partition
            else:
                n_col = QColor("#2563eb")  # Intermediate Blue

            painter.setBrush(QBrush(n_col))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.drawEllipse(pt, 10.0, 10.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(pt.x() - 10.0, pt.y() - 10.0, 20.0, 20.0), Qt.AlignmentFlag.AlignCenter, nid)

        # ---------------------------------------------------------------------
        # 5. Right Panel: Algorithm Trace & Min-Cut Diagnostics
        # ---------------------------------------------------------------------
        dx, dy, dw, dh = self._diag_rect.x(), self._diag_rect.y(), self._diag_rect.width(), self._diag_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._diag_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(dx + 8.0, dy + 16.0), "THÔNG SỐ & VẾT THUẬT TOÁN (BFS)")

        status_str = "ĐÃ ĐẠT TỐI ƯU (MIN-CUT)!" if self._is_optimal else f"Đang tăng luồng (Bước {self._step_count})"
        status_col = QColor("#10b981") if self._is_optimal else QColor("#fbbf24")

        path_str = " ➔ ".join(self._last_path) if self._last_path else "(Chưa có / Đã kết thúc)"
        if len(path_str) > 28:
            path_str = path_str[:25] + "..."

        lines = [
            (f"• Trạng thái: {status_str}", status_col, True),
            (f"• Luồng cực đại |f|: {self._total_flow}", QColor("#38bdf8"), True),
            (f"• Lát cắt hẹp nhất: {self._min_cut_capacity}" if self._is_optimal else "• Lát cắt: (Đang tính...)", QColor("#f43f5e"), True),
            (f"• Đường tăng luồng BFS:", QColor("#94a3b8"), False),
            (f"  {path_str}", QColor("#fbbf24"), True),
            (f"• Nghẽn cổ chai Δ: +{self._last_bottleneck}" if self._last_bottleneck > 0 else "• Nghẽn cổ chai Δ: 0", QColor("#34d399"), False),
            (f"• Tập S: {{{', '.join(sorted(self._s_set))}}}" if self._is_optimal else f"• Nguồn: {self._preset.source} | Đích: {self._preset.sink}", QColor("#a78bfa"), False),
            (f"• Tập T: {{{', '.join(sorted(self._t_set))}}}" if self._is_optimal else f"• Tổng số cạnh: {len(self._preset.edges)}", QColor("#f472b6"), False),
        ]

        ly = dy + 32.0
        for l_txt, l_col, is_bld in lines:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_bld else QFont.Weight.Normal))
            painter.setPen(l_col)
            painter.drawText(QPointF(dx + 8.0, ly), l_txt)
            ly += 17.0

        # Flow summary banner
        flow_banner = QRectF(dx + 8.0, dy + dh - 24.0, dw - 16.0, 16.0)
        painter.setBrush(QBrush(QColor("#065f46") if self._is_optimal else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#10b981") if self._is_optimal else QColor("#334155"), 1.0))
        painter.drawRoundedRect(flow_banner, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        banner_msg = f"MAX-FLOW = MIN-CUT = {self._total_flow}" if self._is_optimal else f"LUỒNG HIỆN TẠI: {self._total_flow}"
        painter.drawText(flow_banner, Qt.AlignmentFlag.AlignCenter, banner_msg)

        # ---------------------------------------------------------------------
        # 6. Bottom Panel: Ford-Fulkerson & Menger Theorem Principles
        # ---------------------------------------------------------------------
        bot_y = self._graph_rect.bottom() + 10.0
        bot_h = self.HEIGHT - bot_y - 10.0
        bot_rect = QRectF(14.0, bot_y, self.WIDTH - 28.0, bot_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(bot_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(22.0, bot_y + 16.0), "ĐỊNH LÝ LUỒNG CỰC ĐẠI - LÁT CẮT HẸP NHẤT (FORD-FULKERSON, 1956):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        b_desc = (
            f"• Định lý Max-Flow Min-Cut: Giá trị luồng cực đại luôn bằng dung lượng của lát cắt hẹp nhất: max |f| = min c(S, T).\n"
            f"• Thuật toán Edmonds-Karp sử dụng BFS tìm đường tăng luồng ngắn nhất trên đồ thị thặng dư G_f, độ phức tạp O(V·E²).\n"
            f"• Đặc tính mẫu: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, bot_y + 22.0, bot_rect.width() - 20.0, 58.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            b_desc,
        )
