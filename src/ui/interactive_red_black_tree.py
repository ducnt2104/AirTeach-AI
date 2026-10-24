"""
AirTeach AI – Interactive Computer Science: Red-Black Self-Balancing Binary Search Tree Simulator
================================================================================================
Provides an authentic computer science simulation of Red-Black Trees (CLRS Chapter 13):
  - 5 Fundamental Red-Black Tree Invariants:
      1. Every node is either RED or BLACK.
      2. The root node is always BLACK.
      3. Every leaf (NIL node) is BLACK.
      4. If a node is RED, both of its children must be BLACK (No consecutive RED nodes).
      5. For each node, all simple paths from the node to descendant leaves contain
         the same number of BLACK nodes (Equal Black-Height bh(x)).
  - Guaranteed Complexity:
      Height h <= 2 * log2(n + 1). Search, Insertion, and Deletion run in O(log n) worst-case!
  - Rotations & Rebalancing:
      - Left-Rotate(T, x) and Right-Rotate(T, y) maintaining in-order BST order.
      - Insertion Fixup: Case 1 (Red Uncle - Recolor), Case 2 (Black Uncle Triangle - Rotate),
        Case 3 (Black Uncle Line - Recolor & Rotate).
      - Deletion Fixup: Double-black resolution across 4 canonical cases.
  - 5 Curriculum Presets:
      1. CLASSIC_CONSTRUCTION: Sequential insertion [10, 20, 30, 15, 25, 5, 1].
      2. WORST_CASE_DEGENERATE_SORTED: Ordered sequence [1, 2, 3, 4, 5, 6, 7, 8] demonstrating
         self-balancing from O(n) line into O(log n) balanced tree.
      3. ROTATION_TRIANGLE_ZIGZAG: Left-Right and Right-Left double rotation cases.
      4. BLACK_HEIGHT_VERIFICATION: Comprehensive black-height bh(x) verification across all paths.
      5. LINUX_KERNEL_SCHEDULER_CFS: Linux Completely Fair Scheduler (CFS) vruntime task queue.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
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
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)


class NodeColor(enum.Enum):
    RED = "RED"
    BLACK = "BLACK"


class RBNode:
    """Authentic Red-Black Tree Node conforming to CLRS Chapter 13."""

    def __init__(
        self,
        key: int,
        color: NodeColor = NodeColor.RED,
        task_name: str = "",
        vruntime_ms: float = 0.0,
    ) -> None:
        self.key: int = key
        self.color: NodeColor = color
        self.left: Optional[RBNode] = None
        self.right: Optional[RBNode] = None
        self.parent: Optional[RBNode] = None

        # Metadata for Linux CFS scheduler preset
        self.task_name: str = task_name
        self.vruntime_ms: float = vruntime_ms

        # Graphical layout coordinates
        self.x: float = 0.0
        self.y: float = 0.0

    @property
    def is_nil(self) -> bool:
        return self.key == -1


@dataclass(frozen=True)
class RBPreset:
    code: str
    title_vi: str
    badge_label: str
    keys: list[int]
    description: str


PRESETS: dict[str, RBPreset] = {
    "CLASSIC_CONSTRUCTION": RBPreset(
        code="CLASSIC_CONSTRUCTION",
        title_vi="Khởi tạo chuẩn CLRS [10, 20, 30, 15, 25, 5, 1]",
        badge_label="CHUẨN CLRS",
        keys=[10, 20, 30, 15, 25, 5, 1],
        description="Chèn tuần tự minh họa 3 trường hợp: Đổi màu khi chú Đỏ, Xoay kép tam giác và Xoay đơn đường thẳng.",
    ),
    "WORST_CASE_DEGENERATE_SORTED": RBPreset(
        code="WORST_CASE_DEGENERATE_SORTED",
        title_vi="Chèn dãy tăng dần [1..8] (Tránh suy biến O(n))",
        badge_label="DÃY TĂNG",
        keys=[1, 2, 3, 4, 5, 6, 7, 8],
        description="Nếu là BST thường cây sẽ suy biến thành danh sách liên kết cao 8; Cây Đỏ-Đen tự cân bằng về chiều cao 4.",
    ),
    "ROTATION_TRIANGLE_ZIGZAG": RBPreset(
        code="ROTATION_TRIANGLE_ZIGZAG",
        title_vi="Xoay kép Zigzag (Left-Right & Right-Left Rotations)",
        badge_label="XOAY ZIGZAG",
        keys=[50, 20, 30, 70, 90, 80],
        description="Minh họa chuyển đổi Trường hợp 2 (Tam giác) thành Trường hợp 3 (Đường thẳng) thông qua phép xoay con trước.",
    ),
    "BLACK_HEIGHT_VERIFICATION": RBPreset(
        code="BLACK_HEIGHT_VERIFICATION",
        title_vi="Xác thực Chiều cao Đen (Black-Height bh Invariant)",
        badge_label="CHIỀU CAO ĐEN",
        keys=[40, 20, 60, 10, 30, 50, 70],
        description="Kiểm tra tính chất 5: Mọi đường đi từ gốc đến các lá NIL đều chứa số lượng nút ĐEN bằng nhau tuyệt đối.",
    ),
    "LINUX_KERNEL_SCHEDULER_CFS": RBPreset(
        code="LINUX_KERNEL_SCHEDULER_CFS",
        title_vi="Bộ điều phối Linux CFS (Completely Fair Scheduler)",
        badge_label="LINUX CFS",
        keys=[12, 5, 24, 2, 8, 18, 30],
        description="Kernel Linux lưu trữ tiến trình theo thời gian chạy ảo vruntime trong Cây Đỏ-Đen; nút cực trái luôn được CPU chọn trước.",
    ),
}


class InteractiveRedBlackTreeItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Red-Black Self-Balancing Binary Search Tree,
    CLRS 5 properties verification, Left/Right rotations, and Linux CFS scheduling.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    tree_updated = Signal(int)  # Emits current node count

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (140.0, 50.0),
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

        # Sentinel NIL Node (CLRS standard: black leaf)
        self.NIL = RBNode(key=-1, color=NodeColor.BLACK)
        self.NIL.left = self.NIL
        self.NIL.right = self.NIL
        self.root: RBNode = self.NIL

        # Active Preset & Simulation State
        self._active_preset_code: str = "CLASSIC_CONSTRUCTION"
        self._preset: RBPreset = PRESETS[self._active_preset_code]

        self._pending_key: int = 16              # Key staged in the stepper to insert/delete
        self._last_rotated_node: Optional[int] = None
        self._operation_log: list[str] = []

        # Animation state
        self._anim_phase: float = 0.0
        self._is_animating: bool = True

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

        # Load initial preset keys
        self.load_preset(self._active_preset_code)

        # 30 FPS Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Core Red-Black Tree Implementation (Zero Mock Data - CLRS Chapter 13)
    # -------------------------------------------------------------------------
    def clear(self) -> None:
        """Empties the tree to NIL root."""
        self.root = self.NIL
        self._operation_log.clear()

    def left_rotate(self, x: RBNode) -> None:
        """Left-rotates node x according to CLRS 13.2."""
        y = x.right
        if y is None or y == self.NIL:
            return

        x.right = y.left
        if y.left != self.NIL and y.left is not None:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y
        self._last_rotated_node = x.key
        self._operation_log.append(f"Xoay Trái tại nút {x.key}")

    def right_rotate(self, y: RBNode) -> None:
        """Right-rotates node y according to CLRS 13.2."""
        x = y.left
        if x is None or x == self.NIL:
            return

        y.left = x.right
        if x.right != self.NIL and x.right is not None:
            x.right.parent = y

        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x

        x.right = y
        y.parent = x
        self._last_rotated_node = y.key
        self._operation_log.append(f"Xoay Phải tại nút {y.key}")

    def insert(self, key: int, task_name: str = "", vruntime_ms: float = 0.0) -> None:
        """Inserts key into Red-Black Tree with CLRS rebalancing."""
        # Avoid duplicate keys
        if self.search(key) != self.NIL:
            return

        z = RBNode(key=key, color=NodeColor.RED, task_name=task_name, vruntime_ms=vruntime_ms)
        z.left = self.NIL
        z.right = self.NIL

        y: Optional[RBNode] = None
        x = self.root

        while x != self.NIL and x is not None:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right

        z.parent = y
        if y is None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

        self._operation_log.append(f"Chèn nút ĐỎ {key}")
        self._insert_fixup(z)
        self.recompute_layout()
        self.tree_updated.emit(self.node_count)
        self.update()

    def _insert_fixup(self, z: RBNode) -> None:
        """Restores Red-Black properties after insertion (CLRS 13.3)."""
        while z.parent is not None and z.parent.color == NodeColor.RED:
            if z.parent.parent is not None and z.parent == z.parent.parent.left:
                y = z.parent.parent.right  # Uncle
                if y is not None and y.color == NodeColor.RED:
                    # Case 1: Uncle is RED -> Recolor
                    z.parent.color = NodeColor.BLACK
                    y.color = NodeColor.BLACK
                    z.parent.parent.color = NodeColor.RED
                    self._operation_log.append(f"TH1: Đổi màu cha & chú của {z.key}")
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        # Case 2: Uncle is BLACK & z is right child (Triangle) -> Left rotate
                        z = z.parent
                        self._operation_log.append(f"TH2: Xoay Trái giảm tam giác tại {z.key}")
                        self.left_rotate(z)
                    # Case 3: Uncle is BLACK & z is left child (Line) -> Recolor & Right rotate
                    if z.parent is not None:
                        z.parent.color = NodeColor.BLACK
                        if z.parent.parent is not None:
                            z.parent.parent.color = NodeColor.RED
                            self._operation_log.append(f"TH3: Đổi màu & Xoay Phải ông tại {z.parent.parent.key}")
                            self.right_rotate(z.parent.parent)
            elif z.parent.parent is not None:
                # Symmetrical case: parent is right child of grandparent
                y = z.parent.parent.left  # Uncle
                if y is not None and y.color == NodeColor.RED:
                    # Case 1: Uncle is RED -> Recolor
                    z.parent.color = NodeColor.BLACK
                    y.color = NodeColor.BLACK
                    z.parent.parent.color = NodeColor.RED
                    self._operation_log.append(f"TH1: Đổi màu cha & chú của {z.key}")
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        # Case 2: Uncle is BLACK & z is left child -> Right rotate
                        z = z.parent
                        self._operation_log.append(f"TH2: Xoay Phải giảm tam giác tại {z.key}")
                        self.right_rotate(z)
                    # Case 3: Line -> Recolor & Left rotate
                    if z.parent is not None:
                        z.parent.color = NodeColor.BLACK
                        if z.parent.parent is not None:
                            z.parent.parent.color = NodeColor.RED
                            self._operation_log.append(f"TH3: Đổi màu & Xoay Trái ông tại {z.parent.parent.key}")
                            self.left_rotate(z.parent.parent)

        # Property 2: Root is always BLACK
        self.root.color = NodeColor.BLACK

    def search(self, key: int) -> RBNode:
        """Searches for key in tree, returns node or NIL."""
        cur = self.root
        while cur != self.NIL and cur is not None:
            if key == cur.key:
                return cur
            elif key < cur.key:
                cur = cur.left
            else:
                cur = cur.right
        return self.NIL

    def delete(self, key: int) -> bool:
        """Deletes key from Red-Black Tree conforming to CLRS 13.4."""
        z = self.search(key)
        if z == self.NIL or z is None:
            return False

        y = z
        y_original_color = y.color

        if z.left == self.NIL or z.left is None:
            x = z.right
            self._rb_transplant(z, z.right)
        elif z.right == self.NIL or z.right is None:
            x = z.left
            self._rb_transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                if x is not None:
                    x.parent = y
            else:
                self._rb_transplant(y, y.right)
                y.right = z.right
                if y.right is not None:
                    y.right.parent = y
            self._rb_transplant(z, y)
            y.left = z.left
            if y.left is not None:
                y.left.parent = y
            y.color = z.color

        if y_original_color == NodeColor.BLACK and x is not None and x != self.NIL:
            self._delete_fixup(x)

        self._operation_log.append(f"Xóa nút {key}")
        self.recompute_layout()
        self.tree_updated.emit(self.node_count)
        self.update()
        return True

    def _rb_transplant(self, u: RBNode, v: Optional[RBNode]) -> None:
        if u.parent is None:
            self.root = v if v is not None else self.NIL
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None and v != self.NIL:
            v.parent = u.parent

    def _minimum(self, node: RBNode) -> RBNode:
        cur = node
        while cur.left != self.NIL and cur.left is not None:
            cur = cur.left
        return cur

    def _delete_fixup(self, x: RBNode) -> None:
        while x != self.root and x.color == NodeColor.BLACK:
            if x.parent is not None and x == x.parent.left:
                w = x.parent.right
                if w is not None and w.color == NodeColor.RED:
                    w.color = NodeColor.BLACK
                    x.parent.color = NodeColor.RED
                    self.left_rotate(x.parent)
                    w = x.parent.right
                if w is not None and (w.left is None or w.left.color == NodeColor.BLACK) and (w.right is None or w.right.color == NodeColor.BLACK):
                    w.color = NodeColor.RED
                    x = x.parent
                else:
                    if w is not None and (w.right is None or w.right.color == NodeColor.BLACK):
                        if w.left is not None:
                            w.left.color = NodeColor.BLACK
                        w.color = NodeColor.RED
                        self.right_rotate(w)
                        w = x.parent.right
                    if w is not None:
                        w.color = x.parent.color
                        if w.right is not None:
                            w.right.color = NodeColor.BLACK
                    if x.parent is not None:
                        x.parent.color = NodeColor.BLACK
                        self.left_rotate(x.parent)
                    x = self.root
            elif x.parent is not None:
                w = x.parent.left
                if w is not None and w.color == NodeColor.RED:
                    w.color = NodeColor.BLACK
                    x.parent.color = NodeColor.RED
                    self.right_rotate(x.parent)
                    w = x.parent.left
                if w is not None and (w.right is None or w.right.color == NodeColor.BLACK) and (w.left is None or w.left.color == NodeColor.BLACK):
                    w.color = NodeColor.RED
                    x = x.parent
                else:
                    if w is not None and (w.left is None or w.left.color == NodeColor.BLACK):
                        if w.right is not None:
                            w.right.color = NodeColor.BLACK
                        w.color = NodeColor.RED
                        self.left_rotate(w)
                        w = x.parent.left
                    if w is not None:
                        w.color = x.parent.color
                        if w.left is not None:
                            w.left.color = NodeColor.BLACK
                    if x.parent is not None:
                        x.parent.color = NodeColor.BLACK
                        self.right_rotate(x.parent)
                    x = self.root
        x.color = NodeColor.BLACK

    # -------------------------------------------------------------------------
    # Invariant Verification & Diagnostics (Zero Mock Data)
    # -------------------------------------------------------------------------
    def verify_invariants(self) -> tuple[bool, list[str]]:
        """
        Verifies the 5 Red-Black tree invariants.
        Returns: (all_valid, list of status messages).
        """
        violations = []

        if self.root == self.NIL:
            return True, ["Cây rỗng (Hợp lệ)"]

        # Property 2: Root is BLACK
        if self.root.color != NodeColor.BLACK:
            violations.append("Vi phạm tính chất 2: Gốc không có màu ĐEN!")

        # Property 4: No two adjacent RED nodes
        def check_no_consecutive_red(node: Optional[RBNode]) -> bool:
            if node is None or node == self.NIL:
                return True
            if node.color == NodeColor.RED:
                if (node.left is not None and node.left.color == NodeColor.RED) or \
                   (node.right is not None and node.right.color == NodeColor.RED):
                    return False
            return check_no_consecutive_red(node.left) and check_no_consecutive_red(node.right)

        if not check_no_consecutive_red(self.root):
            violations.append("Vi phạm tính chất 4: Tồn tại 2 nút ĐỎ liền kề nhau!")

        # Property 5: Equal Black Height across all paths
        def check_black_height(node: Optional[RBNode]) -> int:
            if node is None or node == self.NIL:
                return 1  # NIL nodes are black
            left_bh = check_black_height(node.left)
            right_bh = check_black_height(node.right)
            if left_bh == -1 or right_bh == -1 or left_bh != right_bh:
                return -1
            return left_bh + (1 if node.color == NodeColor.BLACK else 0)

        bh = check_black_height(self.root)
        if bh == -1:
            violations.append("Vi phạm tính chất 5: Chiều cao đen giữa các nhánh không bằng nhau!")

        is_valid = (len(violations) == 0)
        messages = violations if not is_valid else [
            "Tính chất 1: Tất cả nút đều có màu ĐỎ hoặc ĐEN ✓",
            "Tính chất 2: Nút gốc có màu ĐEN ✓",
            "Tính chất 3: Mọi nút lá NIL đều là ĐEN ✓",
            "Tính chất 4: Không có 2 nút ĐỎ kề nhau ✓",
            f"Tính chất 5: Chiều cao Đen đồng nhất bh = {self.black_height} ✓",
        ]
        return is_valid, messages

    @property
    def black_height(self) -> int:
        """Returns the black height bh(root)."""
        if self.root == self.NIL or self.root is None:
            return 0
        bh = 0
        cur = self.root
        while cur != self.NIL and cur is not None:
            if cur.color == NodeColor.BLACK:
                bh += 1
            cur = cur.left
        return bh

    @property
    def tree_height(self) -> int:
        """Computes true height h of tree."""
        def _h(node: Optional[RBNode]) -> int:
            if node is None or node == self.NIL:
                return 0
            return 1 + max(_h(node.left), _h(node.right))
        return _h(self.root)

    @property
    def node_count(self) -> int:
        """Returns number of internal keys."""
        def _count(node: Optional[RBNode]) -> int:
            if node is None or node == self.NIL:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        return _count(self.root)

    @property
    def leftmost_node(self) -> Optional[RBNode]:
        """Returns leftmost node (lowest key / lowest vruntime in CFS)."""
        if self.root == self.NIL or self.root is None:
            return None
        return self._minimum(self.root)

    # -------------------------------------------------------------------------
    # Spatial Layout Engine
    # -------------------------------------------------------------------------
    def recompute_layout(self) -> None:
        """Positions tree nodes symmetrically inside the viewport."""
        if self.root == self.NIL or self.root is None:
            return

        viewport_w = 445.0
        cx = 14.0 + viewport_w / 2.0
        start_y = 75.0

        def _assign_coords(node: Optional[RBNode], x: float, y: float, offset_x: float) -> None:
            if node is None or node == self.NIL:
                return
            node.x = x
            node.y = y
            dy = 50.0
            next_offset = max(24.0, offset_x * 0.52)
            if node.left != self.NIL and node.left is not None:
                _assign_coords(node.left, x - offset_x, y + dy, next_offset)
            if node.right != self.NIL and node.right is not None:
                _assign_coords(node.right, x + offset_x, y + dy, next_offset)

        _assign_coords(self.root, cx, start_y, 95.0)

    # -------------------------------------------------------------------------
    # Preset Loader & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> RBPreset:
        return self._preset

    @property
    def pending_key(self) -> int:
        return self._pending_key

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads preset key sequence and re-constructs the Red-Black tree."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self.clear()

        # Custom task names for CFS mode
        cfs_tasks = ["audio", "renderer", "network", "desktop_ui", "db_worker", "kernel_daemon", "sensor"]

        for idx, k in enumerate(self._preset.keys):
            t_name = cfs_tasks[idx % len(cfs_tasks)] if "CFS" in preset_code else ""
            vrt = float(k) * 1.5 if "CFS" in preset_code else 0.0
            self.insert(k, task_name=t_name, vruntime_ms=vrt)

        self._pending_key = max(self._preset.keys) + 5
        self.update()

    def set_pending_key(self, key: int) -> None:
        self._pending_key = max(1, min(999, key))
        self.update()

    def insert_pending_key(self) -> None:
        self.insert(self._pending_key)
        self._pending_key += 5
        self.update()

    def delete_pending_key(self) -> None:
        self.delete(self._pending_key)
        self.update()

    def toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def stamp_to_board(self) -> None:
        self.stamp_requested.emit()

    def cleanup(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_tick(self) -> None:
        self._anim_phase += 0.08
        if self._anim_phase >= 2.0 * math.pi:
            self._anim_phase -= 2.0 * math.pi
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides
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

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Box
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#070b14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Tree Structure)
        left_w = 445.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_viewport(painter, view_rect)

        # 4. Right Control & Verification Panel
        right_rect = QRectF(view_rect.right() + 10.0, 48.0, self.WIDTH - view_rect.right() - 24.0, 310.0)
        self._draw_control_panel(painter, right_rect)

        # 5. Bottom Theory Banner
        bot_rect = QRectF(14.0, view_rect.bottom() + 10.0, self.WIDTH - 28.0, self.HEIGHT - view_rect.bottom() - 20.0)
        self._draw_theory_banner(painter, bot_rect)

        painter.restore()

    # -------------------------------------------------------------------------
    # Sub-Drawing Modules
    # -------------------------------------------------------------------------
    def _draw_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "CÂY TỰ CÂN BẰNG ĐỎ-ĐEN (RED-BLACK TREE & CLRS INVARIANTS)")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Xoay Trái/Phải • Đổi màu chú • Chiều cao đen bh(x) • Điều phối Linux CFS")

        # Preset Tabs
        px = 440.0
        py = 10.0
        pw = 42.0
        ph = 24.0

        for code, preset in PRESETS.items():
            btn_r = QRectF(px, py, pw, ph)
            self._btn_rects[f"preset_{code}"] = btn_r

            is_sel = (code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#0284c7") if is_sel else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#334155"), 1.0))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)

            lbl = preset.badge_label.split()[0]
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_sel else QColor("#94a3b8"))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, lbl)
            px += pw + 4.0

        # Stamp Button
        stamp_r = QRectF(self.WIDTH - 52.0, 10.0, 38.0, 24.0)
        self._btn_rects["stamp"] = stamp_r
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(stamp_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(stamp_r, Qt.AlignmentFlag.AlignCenter, "Ghim")

    def _draw_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        # 1. Status banner
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 20.0), f"CHẾ ĐỘ: {self._preset.badge_label} (N = {self.node_count} nút, Chiều cao h = {self.tree_height}, bh = {self.black_height})")

        # 2. Draw Tree Edges first
        self._draw_edges(painter, self.root)

        # 3. Draw Tree Nodes & NIL Leaves
        self._draw_nodes(painter, self.root)

        painter.restore()

    def _draw_edges(self, painter: QPainter, node: Optional[RBNode]) -> None:
        if node is None or node == self.NIL:
            return

        pen = QPen(QColor("#475569"), 1.6)
        painter.setPen(pen)

        if node.left != self.NIL and node.left is not None:
            painter.drawLine(QPointF(node.x, node.y), QPointF(node.left.x, node.left.y))
            self._draw_edges(painter, node.left)
        elif node.left == self.NIL:
            # Short stub to NIL leaf
            nil_x = node.x - 14.0
            nil_y = node.y + 28.0
            painter.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(node.x, node.y), QPointF(nil_x, nil_y))
            painter.setPen(pen)

        if node.right != self.NIL and node.right is not None:
            painter.drawLine(QPointF(node.x, node.y), QPointF(node.right.x, node.right.y))
            self._draw_edges(painter, node.right)
        elif node.right == self.NIL:
            # Short stub to NIL leaf
            nil_x = node.x + 14.0
            nil_y = node.y + 28.0
            painter.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(node.x, node.y), QPointF(nil_x, nil_y))
            painter.setPen(pen)

    def _draw_nodes(self, painter: QPainter, node: Optional[RBNode]) -> None:
        if node is None or node == self.NIL:
            return

        r = 15.0

        # Highlighting for CFS leftmost process
        is_leftmost = (node == self.leftmost_node and "CFS" in self._active_preset_code)

        if node.color == NodeColor.RED:
            brush = QBrush(QColor("#dc2626"))
            pen = QPen(QColor("#f87171"), 1.8)
        else:
            brush = QBrush(QColor("#0f172a"))
            pen = QPen(QColor("#64748b"), 1.8)

        if is_leftmost:
            pen = QPen(QColor("#fbbf24"), 2.5)

        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(node.x, node.y), r, r)

        # Node Key Text
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            QRectF(node.x - r, node.y - r, 2 * r, 2 * r),
            Qt.AlignmentFlag.AlignCenter,
            str(node.key),
        )

        # CFS metadata callout
        if "CFS" in self._active_preset_code and node.task_name:
            painter.setFont(QFont("Segoe UI", 5))
            painter.setPen(QColor("#fbbf24") if is_leftmost else QColor("#94a3b8"))
            painter.drawText(QPointF(node.x - 20.0, node.y + r + 10.0), f"{node.task_name}")

        # Draw NIL stubs
        if node.left == self.NIL:
            self._draw_nil_leaf(painter, QPointF(node.x - 14.0, node.y + 28.0))
        else:
            self._draw_nodes(painter, node.left)

        if node.right == self.NIL:
            self._draw_nil_leaf(painter, QPointF(node.x + 14.0, node.y + 28.0))
        else:
            self._draw_nodes(painter, node.right)

    def _draw_nil_leaf(self, painter: QPainter, pos: QPointF) -> None:
        w = 12.0
        h = 8.0
        r = QRectF(pos.x() - w / 2.0, pos.y() - h / 2.0, w, h)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRect(r)

        painter.setFont(QFont("Segoe UI", 5))
        painter.setPen(QColor("#64748b"))
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, "NIL")

    # -------------------------------------------------------------------------
    # Right Control & Verification Panel
    # -------------------------------------------------------------------------
    def _draw_control_panel(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        ox = rect.left() + 10.0
        w = rect.width() - 20.0
        y = rect.top() + 14.0

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "THAO TÁC CÂY ĐỎ-ĐEN:")

        # Staged Key Stepper Card
        y += 18.0
        key_card = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(key_card, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QRectF(ox + 8.0, y, 60.0, 32.0), Qt.AlignmentFlag.AlignVCenter, f"K={self._pending_key}")

        # Insert & Delete Buttons
        btn_ins = QRectF(ox + 75.0, y + 4.0, 56.0, 24.0)
        btn_del = QRectF(ox + 135.0, y + 4.0, 56.0, 24.0)
        self._btn_rects["insert"] = btn_ins
        self._btn_rects["delete"] = btn_del

        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(btn_ins, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(btn_ins, Qt.AlignmentFlag.AlignCenter, "+ Chèn")

        painter.setBrush(QBrush(QColor("#b91c1c")))
        painter.setPen(QPen(QColor("#ef4444"), 1.0))
        painter.drawRoundedRect(btn_del, 3.0, 3.0)
        painter.drawText(btn_del, Qt.AlignmentFlag.AlignCenter, "- Xóa")

        # Steppers for Pending Key
        y += 38.0
        k_m5 = QRectF(ox, y, 32.0, 20.0)
        k_m1 = QRectF(ox + 36.0, y, 32.0, 20.0)
        k_p1 = QRectF(ox + 100.0, y, 32.0, 20.0)
        k_p5 = QRectF(ox + 136.0, y, 32.0, 20.0)

        self._btn_rects["k_m5"] = k_m5
        self._btn_rects["k_m1"] = k_m1
        self._btn_rects["k_p1"] = k_p1
        self._btn_rects["k_p5"] = k_p5

        for b_r, b_txt in [(k_m5, "-5"), (k_m1, "-1"), (k_p1, "+1"), (k_p5, "+5")]:
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.drawRoundedRect(b_r, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, b_txt)

        # Invariant Verification Box
        y += 28.0
        is_valid, inv_msgs = self.verify_invariants()

        inv_rect = QRectF(ox, y, w, 82.0)
        painter.setBrush(QBrush(QColor("#064e3b") if is_valid else QColor("#7f1d1d")))
        painter.setPen(QPen(QColor("#10b981") if is_valid else QColor("#ef4444"), 1.0))
        painter.drawRoundedRect(inv_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QPointF(ox + 6.0, y + 12.0), "XÁC THỰC 5 TÍNH CHẤT CLRS:")

        my = y + 24.0
        painter.setFont(QFont("Segoe UI", 6))
        for msg in inv_msgs[:4]:
            painter.drawText(QPointF(ox + 6.0, my), msg)
            my += 14.0

        # Tree Metrics Readout
        y += 90.0
        n = self.node_count
        h = self.tree_height
        bh = self.black_height
        max_h = 2 * math.ceil(math.log2(n + 1)) if n > 0 else 0

        rows = [
            (f"• Số lượng nút n: {n}", QColor("#38bdf8")),
            (f"• Chiều cao cây h: {h} (≤ {max_h} tối đa)", QColor("#10b981")),
            (f"• Chiều cao đen bh: {bh}", QColor("#fbbf24")),
            (f"• Độ phức tạp: O(log n) đảm bảo", QColor("#cbd5e1")),
        ]

        if "CFS" in self._active_preset_code and self.leftmost_node:
            lm = self.leftmost_node
            rows.append((f"• CFS Tiếp theo: [{lm.task_name}] v={lm.vruntime_ms:.1f}ms", QColor("#f43f5e")))

        for txt, col in rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 15.0

        # Action Buttons
        y = rect.bottom() - 24.0
        clr_r = QRectF(ox, y, w / 2.0 - 4.0, 18.0)
        anim_r = QRectF(ox + w / 2.0 + 4.0, y, w / 2.0 - 4.0, 18.0)
        self._btn_rects["clear"] = clr_r
        self._btn_rects["toggle_anim"] = anim_r

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(clr_r, 3.0, 3.0)
        painter.drawRoundedRect(anim_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(clr_r, Qt.AlignmentFlag.AlignCenter, "Xóa cây")
        painter.drawText(anim_r, Qt.AlignmentFlag.AlignCenter, "Dừng" if self._is_animating else "Chạy")

    # -------------------------------------------------------------------------
    # Bottom Physics Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "NGUYÊN LÝ CÂY ĐỎ-ĐEN (BAYER 1972 & GUYBAS-SEDGEWICK 1978):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Cây Đỏ-Đen đảm bảo không nhánh nào dài gấp đôi nhánh khác nhờ 5 tiên đề CLRS. Chiều cao luôn thỏa h ≤ 2·log₂(n + 1).\n"
            f"• Thao tác tái cân bằng sử dụng phép Xoay Trái/Phải O(1) và Đổi màu nút. "
            f"Được ứng dụng cốt lõi trong std::map / std::set (C++), TreeMap (Java) và bộ điều phối Completely Fair Scheduler (Linux Kernel).\n"
            f"• Mô tả chế độ: {self._preset.description}"
        )
        painter.drawText(
            QRectF(rect.left() + 10.0, rect.top() + 22.0, rect.width() - 20.0, 38.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc,
        )

    # -------------------------------------------------------------------------
    # Touchless Gesture & Mouse Interaction
    # -------------------------------------------------------------------------
    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture pinch handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset tabs
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(local_pos):
                self.load_preset(code)
                return True

        # 2. Key steppers
        if self._btn_rects.get("k_m5") and self._btn_rects["k_m5"].contains(local_pos):
            self.set_pending_key(self._pending_key - 5)
            return True
        if self._btn_rects.get("k_m1") and self._btn_rects["k_m1"].contains(local_pos):
            self.set_pending_key(self._pending_key - 1)
            return True
        if self._btn_rects.get("k_p1") and self._btn_rects["k_p1"].contains(local_pos):
            self.set_pending_key(self._pending_key + 1)
            return True
        if self._btn_rects.get("k_p5") and self._btn_rects["k_p5"].contains(local_pos):
            self.set_pending_key(self._pending_key + 5)
            return True

        # 3. Insert / Delete buttons
        if self._btn_rects.get("insert") and self._btn_rects["insert"].contains(local_pos):
            self.insert_pending_key()
            return True
        if self._btn_rects.get("delete") and self._btn_rects["delete"].contains(local_pos):
            self.delete_pending_key()
            return True

        # 4. Action buttons
        if self._btn_rects.get("clear") and self._btn_rects["clear"].contains(local_pos):
            self.clear()
            self.update()
            return True
        if self._btn_rects.get("toggle_anim") and self._btn_rects["toggle_anim"].contains(local_pos):
            self.toggle_animation()
            return True
        if self._btn_rects.get("stamp") and self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()
        if self.handle_air_pinch(self.mapToScene(loc)):
            event.accept()
            return
        super().mousePressEvent(event)
