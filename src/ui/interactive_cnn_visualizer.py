"""
AirTeach AI — Interactive Convolutional Neural Network (CNN) 2D Feature Map & Kernel Convolution Inspector
Authentic 2D discrete convolution, receptive field sliding window, activation functions, pooling,
and touchless air gesture integration for teaching AI & Computer Science.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Callable

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
class CNNPreset:
    name: str
    vietnamese_title: str
    kernel_name: str
    kernel_matrix: tuple[tuple[float, ...], ...]
    activation: str  # RELU, LEAKY_RELU, LINEAR, SIGMOID
    pooling: str  # MAX_POOLING, AVG_POOLING
    input_pattern: tuple[tuple[int, ...], ...]
    description: str


# 8x8 standard pattern: Stylized digit 7
DIGIT_7 = (
    (240, 240, 240, 240, 240, 240, 240, 30),
    (30,  30,  30,  30,  50,  180, 240, 30),
    (30,  30,  30,  50,  200, 240, 100, 30),
    (30,  30,  80,  220, 240, 80,  30,  30),
    (30,  30,  180, 240, 90,  30,  30,  30),
    (30,  80,  240, 200, 30,  30,  30,  30),
    (30,  180, 240, 90,  30,  30,  30,  30),
    (30,  240, 240, 30,  30,  30,  30,  30),
)

# 8x8 standard pattern: Box contour
BOX_CONTOUR = (
    (20,  20,  20,  20,  20,  20,  20,  20),
    (20, 240, 240, 240, 240, 240, 240,  20),
    (20, 240,  30,  30,  30,  30, 240,  20),
    (20, 240,  30,  30,  30,  30, 240,  20),
    (20, 240,  30,  30,  30,  30, 240,  20),
    (20, 240,  30,  30,  30,  30, 240,  20),
    (20, 240, 240, 240, 240, 240, 240,  20),
    (20,  20,  20,  20,  20,  20,  20,  20),
)

# 8x8 standard pattern: Horizontal stripes
HORIZONTAL_STRIPES = (
    (240, 240, 240, 240, 240, 240, 240, 240),
    (240, 240, 240, 240, 240, 240, 240, 240),
    (30,  30,  30,  30,  30,  30,  30,  30),
    (30,  30,  30,  30,  30,  30,  30,  30),
    (240, 240, 240, 240, 240, 240, 240, 240),
    (240, 240, 240, 240, 240, 240, 240, 240),
    (30,  30,  30,  30,  30,  30,  30,  30),
    (30,  30,  30,  30,  30,  30,  30,  30),
)

# 8x8 standard pattern: Checkerboard
CHECKERBOARD = (
    (240, 30, 240, 30, 240, 30, 240, 30),
    (30, 240, 30, 240, 30, 240, 30, 240),
    (240, 30, 240, 30, 240, 30, 240, 30),
    (30, 240, 30, 240, 30, 240, 30, 240),
    (240, 30, 240, 30, 240, 30, 240, 30),
    (30, 240, 30, 240, 30, 240, 30, 240),
    (240, 30, 240, 30, 240, 30, 240, 30),
    (30, 240, 30, 240, 30, 240, 30, 240),
)

# 8x8 standard pattern: Diagonal gradient
DIAGONAL_GRADIENT = (
    (30,  60,  90,  120, 150, 180, 210, 240),
    (60,  90,  120, 150, 180, 210, 240, 210),
    (90,  120, 150, 180, 210, 240, 210, 180),
    (120, 150, 180, 210, 240, 210, 180, 150),
    (150, 180, 210, 240, 210, 180, 150, 120),
    (180, 210, 240, 210, 180, 150, 120, 90),
    (210, 240, 210, 180, 150, 120, 90,  60),
    (240, 210, 180, 150, 120, 90,  60,  30),
)

PRESETS: dict[str, CNNPreset] = {
    "SOBEL_VERTICAL_EDGES": CNNPreset(
        name="SOBEL_VERTICAL_EDGES",
        vietnamese_title="Phát hiện biên dọc (Sobel-X)",
        kernel_name="Sobel-X",
        kernel_matrix=(
            (-1.0, 0.0, 1.0),
            (-2.0, 0.0, 2.0),
            (-1.0, 0.0, 1.0),
        ),
        activation="RELU",
        pooling="MAX_POOLING",
        input_pattern=DIGIT_7,
        description="Lọc gradient theo phương ngang làm nổi bật các cạnh thẳng đứng.",
    ),
    "SOBEL_HORIZONTAL_EDGES": CNNPreset(
        name="SOBEL_HORIZONTAL_EDGES",
        vietnamese_title="Phát hiện biên ngang (Sobel-Y)",
        kernel_name="Sobel-Y",
        kernel_matrix=(
            (-1.0, -2.0, -1.0),
            (0.0,  0.0,  0.0),
            (1.0,  2.0,  1.0),
        ),
        activation="RELU",
        pooling="MAX_POOLING",
        input_pattern=HORIZONTAL_STRIPES,
        description="Lọc gradient theo phương dọc làm nổi bật các đường phân cách ngang.",
    ),
    "RIDGE_LAPLACIAN_CONTOUR": CNNPreset(
        name="RIDGE_LAPLACIAN_CONTOUR",
        vietnamese_title="Dò đường bao đa hướng (Laplacian)",
        kernel_name="Laplacian",
        kernel_matrix=(
            (-1.0, -1.0, -1.0),
            (-1.0,  8.0, -1.0),
            (-1.0, -1.0, -1.0),
        ),
        activation="RELU",
        pooling="MAX_POOLING",
        input_pattern=BOX_CONTOUR,
        description="Đạo hàm bậc hai bắt trọn viền và góc nhọn của vật thể trên ảnh.",
    ),
    "GAUSSIAN_BLUR_SMOOTHING": CNNPreset(
        name="GAUSSIAN_BLUR_SMOOTHING",
        vietnamese_title="Làm mịn giảm nhiễu (Gaussian Blur)",
        kernel_name="Gaussian 3x3",
        kernel_matrix=(
            (0.0625, 0.125, 0.0625),
            (0.125,  0.25,  0.125),
            (0.0625, 0.125, 0.0625),
        ),
        activation="LINEAR",
        pooling="AVG_POOLING",
        input_pattern=CHECKERBOARD,
        description="Bộ lọc thông thấp Gaussian khử nhiễu tần số cao, tạo ảnh chuyển tiếp mượt.",
    ),
    "HIGH_PASS_SHARPENING": CNNPreset(
        name="HIGH_PASS_SHARPENING",
        vietnamese_title="Làm nét ảnh (Sharpen High-Pass)",
        kernel_name="Sharpen",
        kernel_matrix=(
            (0.0, -1.0,  0.0),
            (-1.0, 5.0, -1.0),
            (0.0, -1.0,  0.0),
        ),
        activation="LINEAR",
        pooling="MAX_POOLING",
        input_pattern=DIAGONAL_GRADIENT,
        description="Khuếch đại độ tương phản cục bộ tại các điểm ảnh biên.",
    ),
}


class InteractiveCNNVisualizerItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for demonstrating 2D Convolutional Neural Network operations.
    Supports touchless air pinch, sliding receptive field, activation functions, pooling,
    and direct vector stamping to LessonBoard.
    """

    WIDTH = 640.0
    HEIGHT = 380.0

    stamp_requested = Signal()
    step_changed = Signal(int, int)  # row, col

    def __init__(self, pos: QPointF | tuple[float, float] = (180.0, 100.0), parent: Optional[QGraphicsItem] = None) -> None:
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

        # State variables
        self._active_preset_code: str = "SOBEL_VERTICAL_EDGES"
        self._preset: CNNPreset = PRESETS[self._active_preset_code]

        # 8x8 input grid
        self._input_grid: list[list[float]] = [list(row) for row in self._preset.input_pattern]

        # 3x3 kernel
        self._kernel: list[list[float]] = [list(row) for row in self._preset.kernel_matrix]
        self._kernel_name: str = self._preset.kernel_name

        # Hyperparameters
        self._activation: str = self._preset.activation  # RELU, LEAKY_RELU, LINEAR, SIGMOID
        self._pooling: str = self._preset.pooling  # MAX_POOLING, AVG_POOLING

        # Sliding window position (center cell in 8x8 grid: [1..6] for valid 3x3 window)
        self._window_r: int = 1
        self._window_c: int = 1

        # Computed feature maps
        self._feature_map: list[list[float]] = [[0.0 for _ in range(8)] for _ in range(8)]
        self._pooled_map: list[list[float]] = [[0.0 for _ in range(4)] for _ in range(4)]

        # Animation state
        self._is_animating: bool = False
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(250)  # 4 steps per second
        self._anim_timer.timeout.connect(self._on_animation_step)

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()
        self._compute_full_convolution()

    def _setup_layout(self) -> None:
        """Precomputes button rects for click and air pinch interaction."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0
        preset_names = [
            ("SOBEL_VERTICAL_EDGES", "Sobel-X"),
            ("SOBEL_HORIZONTAL_EDGES", "Sobel-Y"),
            ("RIDGE_LAPLACIAN_CONTOUR", "Laplacian"),
            ("GAUSSIAN_BLUR_SMOOTHING", "Gaussian"),
            ("HIGH_PASS_SHARPENING", "Sharpen"),
        ]
        for p_code, _ in preset_names:
            pw = 72.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 6.0

        ctrl_w = 42.0
        r_play = QRectF(cur_x + 10.0, btn_y, ctrl_w, btn_h)
        self._btn_rects["toggle_anim"] = r_play
        r_step = QRectF(r_play.right() + 6.0, btn_y, ctrl_w, btn_h)
        self._btn_rects["step_forward"] = r_step
        r_reset = QRectF(r_step.right() + 6.0, btn_y, ctrl_w, btn_h)
        self._btn_rects["reset_window"] = r_reset
        r_stamp = QRectF(r_reset.right() + 6.0, btn_y, 48.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        grid_y = 80.0
        grid_x = 18.0
        cell_size = 16.0
        self._btn_rects["input_grid_bounds"] = QRectF(grid_x, grid_y, 8 * cell_size, 8 * cell_size)

        tog_y = grid_y + 8 * cell_size + 16.0
        act_x = 75.0
        for act in ("RELU", "LEAKY_RELU", "LINEAR", "SIGMOID"):
            aw = 60.0
            self._btn_rects[f"act_{act}"] = QRectF(act_x, tog_y, aw, 20.0)
            act_x += aw + 5.0

        pool_x = act_x + 85.0
        for pool in ("MAX_POOLING", "AVG_POOLING"):
            pw = 85.0
            self._btn_rects[f"pool_{pool}"] = QRectF(pool_x, tog_y, pw, 20.0)
            pool_x += pw + 5.0

    def cleanup(self) -> None:
        """Stops any running timers to avoid memory leaks."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    # -------------------------------------------------------------------------
    # Analytical & Mathematical Engine
    # -------------------------------------------------------------------------
    def apply_activation(self, val: float) -> float:
        """Applies chosen activation function."""
        if self._activation == "RELU":
            return max(0.0, val)
        elif self._activation == "LEAKY_RELU":
            return val if val > 0.0 else 0.1 * val
        elif self._activation == "SIGMOID":
            # Normalised sigmoid scaled to [0, 255]
            z = val / 64.0
            sig = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
            return sig * 255.0
        else:  # LINEAR
            return max(0.0, min(255.0, val))

    def compute_single_cell_convolution(self, r: int, c: int) -> float:
        """
        Computes discrete 2D convolution at (r, c) over 3x3 neighborhood:
        S(r, c) = sum_{u=-1}^1 sum_{v=-1}^1 I(r+u, c+v) * K(u+1, v+1)
        """
        acc = 0.0
        for u in (-1, 0, 1):
            for v in (-1, 0, 1):
                # Clamp-to-edge boundary condition
                rr = max(0, min(7, r + u))
                cc = max(0, min(7, c + v))
                i_val = self._input_grid[rr][cc]
                k_val = self._kernel[u + 1][v + 1]
                acc += i_val * k_val
        return acc

    def _compute_pooling(self) -> None:
        """Computes 4x4 pooled map from current 8x8 feature map."""
        for pr in range(4):
            for pc in range(4):
                block = [
                    self._feature_map[2 * pr][2 * pc],
                    self._feature_map[2 * pr][2 * pc + 1],
                    self._feature_map[2 * pr + 1][2 * pc],
                    self._feature_map[2 * pr + 1][2 * pc + 1],
                ]
                if self._pooling == "MAX_POOLING":
                    self._pooled_map[pr][pc] = max(block)
                else:  # AVG_POOLING
                    self._pooled_map[pr][pc] = sum(block) / 4.0

    def _compute_full_convolution(self) -> None:
        """Computes full 8x8 feature map and 4x4 pooled map."""
        for r in range(8):
            for c in range(8):
                raw = self.compute_single_cell_convolution(r, c)
                self._feature_map[r][c] = self.apply_activation(raw)
        self._compute_pooling()

    # -------------------------------------------------------------------------
    # Properties & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> CNNPreset:
        return self._preset

    @property
    def window_position(self) -> tuple[int, int]:
        return (self._window_r, self._window_c)

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def pooling(self) -> str:
        return self._pooling

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined CNN preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._input_grid = [list(row) for row in self._preset.input_pattern]
        self._kernel = [list(row) for row in self._preset.kernel_matrix]
        self._kernel_name = self._preset.kernel_name
        self._activation = self._preset.activation
        self._pooling = self._preset.pooling
        self._window_r = 1
        self._window_c = 1
        self._compute_full_convolution()
        self.update()

    def set_window_position(self, r: int, c: int) -> None:
        """Sets sliding window center position (clamped to [0, 7])."""
        self._window_r = max(0, min(7, int(r)))
        self._window_c = max(0, min(7, int(c)))
        self.step_changed.emit(self._window_r, self._window_c)
        self.update()

    def set_activation(self, act: str) -> None:
        """Sets activation function: RELU, LEAKY_RELU, LINEAR, SIGMOID."""
        valid = ("RELU", "LEAKY_RELU", "LINEAR", "SIGMOID")
        if act in valid:
            self._activation = act
            self._compute_full_convolution()
            self.update()

    def set_pooling(self, pool: str) -> None:
        """Sets pooling mode: MAX_POOLING, AVG_POOLING."""
        if pool in ("MAX_POOLING", "AVG_POOLING"):
            self._pooling = pool
            self._compute_pooling()
            self.update()

    def toggle_animation(self) -> bool:
        """Toggles automatic sliding window scanning."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def step_forward(self) -> None:
        """Moves sliding window one step forward along raster scan."""
        next_c = self._window_c + 1
        next_r = self._window_r
        if next_c > 7:
            next_c = 0
            next_r += 1
            if next_r > 7:
                next_r = 0
        self.set_window_position(next_r, next_c)

    def reset_window(self) -> None:
        """Resets sliding window to (0, 0)."""
        self.set_window_position(0, 0)

    def _on_animation_step(self) -> None:
        """Timer callback for sliding window auto-advance."""
        self.step_forward()

    # -------------------------------------------------------------------------
    # Touchless Air Gesture Handling & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active CNN diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import InformaticsTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = InformaticsTools.create_cnn_diagram(
            center=center,
            preset_code=self._active_preset_code,
            kernel_name=self._kernel_name,
            activation=self._activation,
            pooling=self._pooling,
            window_r=self._window_r,
            window_c=self._window_c,
        )
        for itm in items:
            scene.addItem(itm)
        self.stamp_requested.emit()

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Activation buttons
        for act in ("RELU", "LEAKY_RELU", "LINEAR", "SIGMOID"):
            r = self._btn_rects.get(f"act_{act}")
            if r and r.contains(local_pos):
                self.set_activation(act)
                return True

        # 3. Pooling buttons
        for pool in ("MAX_POOLING", "AVG_POOLING"):
            r = self._btn_rects.get(f"pool_{pool}")
            if r and r.contains(local_pos):
                self.set_pooling(pool)
                return True

        # 4. Play/Pause toggle
        r_play = self._btn_rects.get("toggle_anim")
        if r_play and r_play.contains(local_pos):
            self.toggle_animation()
            return True

        # 5. Step forward
        r_step = self._btn_rects.get("step_forward")
        if r_step and r_step.contains(local_pos):
            self.step_forward()
            return True

        # 6. Reset
        r_reset = self._btn_rects.get("reset_window")
        if r_reset and r_reset.contains(local_pos):
            self.reset_window()
            return True

        # 7. Stamp to board
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
            return True

        # 8. Direct click on Input Grid to position window
        r_grid = self._btn_rects.get("input_grid_bounds")
        if r_grid and r_grid.contains(local_pos):
            cell_size = 16.0
            col = int((local_pos.x() - r_grid.x()) / cell_size)
            row = int((local_pos.y() - r_grid.y()) / cell_size)
            self.set_window_position(row, col)
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

        # 1. Background Card
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
            QRectF(14.0, 0.0, self.WIDTH - 28.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"🧠 MẠNG NƠ-RON CNN: TÍCH CHẬP 2D & BẢN ĐỒ ĐẶC TRƯNG — {self._preset.vietnamese_title}",
        )

        # 3. Preset Buttons (Row 1 under header)
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0
        preset_names = [
            ("SOBEL_VERTICAL_EDGES", "Sobel-X"),
            ("SOBEL_HORIZONTAL_EDGES", "Sobel-Y"),
            ("RIDGE_LAPLACIAN_CONTOUR", "Laplacian"),
            ("GAUSSIAN_BLUR_SMOOTHING", "Gaussian"),
            ("HIGH_PASS_SHARPENING", "Sharpen"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 72.0
            r_btn = QRectF(cur_x, btn_y, pw, btn_h)
            self._btn_rects[f"preset_{p_code}"] = r_btn
            is_active = self._active_preset_code == p_code

            painter.setBrush(QBrush(QColor("#0284c7") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r_btn, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r_btn, Qt.AlignmentFlag.AlignCenter, p_lbl)
            cur_x += pw + 6.0

        # Animation & Step Controls on the right
        ctrl_w = 42.0
        r_play = QRectF(cur_x + 10.0, btn_y, ctrl_w, btn_h)
        self._btn_rects["toggle_anim"] = r_play
        painter.setBrush(QBrush(QColor("#10b981") if self._is_animating else QColor("#334155")))
        painter.setPen(QPen(QColor("#34d399"), 1.0))
        painter.drawRoundedRect(r_play, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, "⏸ Dừng" if self._is_animating else "▶ Chạy")

        r_step = QRectF(r_play.right() + 6.0, btn_y, ctrl_w, btn_h)
        self._btn_rects["step_forward"] = r_step
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_step, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_step, Qt.AlignmentFlag.AlignCenter, "▶| Bước")

        r_reset = QRectF(r_step.right() + 6.0, btn_y, ctrl_w, btn_h)
        self._btn_rects["reset_window"] = r_reset
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_reset, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_reset, Qt.AlignmentFlag.AlignCenter, "↺ Đặt lại")

        r_stamp = QRectF(r_reset.right() + 6.0, btn_y, 48.0, btn_h)
        self._btn_rects["stamp"] = r_stamp
        painter.setBrush(QBrush(QColor("#7c3aed")))
        painter.setPen(QPen(QColor("#a78bfa"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng")

        # ---------------------------------------------------------------------
        # 4. Input Grid (8x8)
        # ---------------------------------------------------------------------
        grid_y = 80.0
        grid_x = 18.0
        cell_size = 16.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(QPointF(grid_x, grid_y - 6.0), "1. Ảnh đầu vào I (8×8):")

        grid_bounds = QRectF(grid_x, grid_y, 8 * cell_size, 8 * cell_size)
        self._btn_rects["input_grid_bounds"] = grid_bounds

        for r in range(8):
            for c in range(8):
                val = self._input_grid[r][c]
                clamped_v = max(0, min(255, int(val)))
                cr = QRectF(grid_x + c * cell_size, grid_y + r * cell_size, cell_size, cell_size)
                painter.setBrush(QBrush(QColor(clamped_v, clamped_v, clamped_v)))
                painter.setPen(QPen(QColor("#1e293b"), 0.5))
                painter.drawRect(cr)

        # Highlight Receptive Field 3x3 Window over input
        win_r0 = max(0, self._window_r - 1)
        win_c0 = max(0, self._window_c - 1)
        win_r1 = min(7, self._window_r + 1)
        win_c1 = min(7, self._window_c + 1)
        win_w = (win_c1 - win_c0 + 1) * cell_size
        win_h = (win_r1 - win_r0 + 1) * cell_size
        win_rect = QRectF(grid_x + win_c0 * cell_size, grid_y + win_r0 * cell_size, win_w, win_h)
        painter.setBrush(QBrush(QColor(239, 68, 68, 60)))
        painter.setPen(QPen(QColor("#ef4444"), 2.0))
        painter.drawRect(win_rect)

        # ---------------------------------------------------------------------
        # 5. Convolution Operator (*) & Kernel Matrix (3x3)
        # ---------------------------------------------------------------------
        k_x = grid_x + 8 * cell_size + 24.0
        k_size = 24.0
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(QPointF(k_x - 18.0, grid_y + 4 * cell_size), "∗")

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(QPointF(k_x, grid_y - 6.0), f"2. Kernel K (3×3) — {self._kernel_name}:")

        for kr in range(3):
            for kc in range(3):
                kval = self._kernel[kr][kc]
                kr_rect = QRectF(k_x + kc * k_size, grid_y + kr * k_size, k_size, k_size)
                # Color code positive, zero, negative
                if kval > 0.0:
                    bg_c = QColor("#065f46")
                elif kval < 0.0:
                    bg_c = QColor("#7f1d1d")
                else:
                    bg_c = QColor("#1e293b")

                painter.setBrush(QBrush(bg_c))
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.drawRect(kr_rect)

                painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
                painter.setPen(QColor("#ffffff"))
                txt_val = f"{kval:+.1f}" if abs(kval) < 1.0 and kval != 0.0 else f"{kval:+.0f}"
                painter.drawText(kr_rect, Qt.AlignmentFlag.AlignCenter, txt_val)

        # ---------------------------------------------------------------------
        # 6. Feature Map S (8x8)
        # ---------------------------------------------------------------------
        f_x = k_x + 3 * k_size + 28.0
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(f_x - 18.0, grid_y + 4 * cell_size), "=")

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(QPointF(f_x, grid_y - 6.0), f"3. Đặc trưng ({self._activation}):")

        for r in range(8):
            for c in range(8):
                val = self._feature_map[r][c]
                norm_v = max(0, min(255, int(val)))
                fr_rect = QRectF(f_x + c * cell_size, grid_y + r * cell_size, cell_size, cell_size)

                # Cyan heatmap coloration
                c_val = QColor(0, norm_v, int(norm_v * 0.9))
                painter.setBrush(QBrush(c_val))
                painter.setPen(QPen(QColor("#1e293b"), 0.5))
                painter.drawRect(fr_rect)

        # Active output cell highlight
        act_rect = QRectF(
            f_x + self._window_c * cell_size,
            grid_y + self._window_r * cell_size,
            cell_size,
            cell_size,
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#f59e0b"), 2.0))
        painter.drawRect(act_rect)

        # ---------------------------------------------------------------------
        # 7. Pooling Grid (4x4)
        # ---------------------------------------------------------------------
        p_x = f_x + 8 * cell_size + 24.0
        p_cell = 20.0
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(QPointF(p_x - 18.0, grid_y + 4 * cell_size), "➔")

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        pool_lbl = "Max" if self._pooling == "MAX_POOLING" else "Avg"
        painter.drawText(QPointF(p_x, grid_y - 6.0), f"4. Pooling ({pool_lbl} 2×2):")

        for pr in range(4):
            for pc in range(4):
                val = self._pooled_map[pr][pc]
                norm_v = max(0, min(255, int(val)))
                pr_rect = QRectF(p_x + pc * p_cell, grid_y + pr * p_cell, p_cell, p_cell)

                # Emerald coloration
                p_col = QColor(0, norm_v, int(norm_v * 0.5))
                painter.setBrush(QBrush(p_col))
                painter.setPen(QPen(QColor("#1e293b"), 1.0))
                painter.drawRect(pr_rect)

        # ---------------------------------------------------------------------
        # 8. Interactive Parameter Toggles (Activation & Pooling)
        # ---------------------------------------------------------------------
        tog_y = grid_y + 8 * cell_size + 16.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(18.0, tog_y + 14.0), "Kích hoạt:")

        act_x = 75.0
        for act in ("RELU", "LEAKY_RELU", "LINEAR", "SIGMOID"):
            aw = 60.0
            r_act = QRectF(act_x, tog_y, aw, 20.0)
            self._btn_rects[f"act_{act}"] = r_act
            is_active = self._activation == act

            painter.setBrush(QBrush(QColor("#0284c7") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r_act, 3.0, 3.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r_act, Qt.AlignmentFlag.AlignCenter, act)
            act_x += aw + 5.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(act_x + 10.0, tog_y + 14.0), "Gộp Pooling:")

        pool_x = act_x + 85.0
        for pool, pool_title in (("MAX_POOLING", "Max Pool 2×2"), ("AVG_POOLING", "Avg Pool 2×2")):
            pw = 85.0
            r_pool = QRectF(pool_x, tog_y, pw, 20.0)
            self._btn_rects[f"pool_{pool}"] = r_pool
            is_active = self._pooling == pool

            painter.setBrush(QBrush(QColor("#059669") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#34d399") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r_pool, 3.0, 3.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r_pool, Qt.AlignmentFlag.AlignCenter, pool_title)
            pool_x += pw + 5.0

        # ---------------------------------------------------------------------
        # 9. Theory & Live Step Inspector Card (Bottom Panel)
        # ---------------------------------------------------------------------
        card_y = tog_y + 28.0
        card_h = self.HEIGHT - card_y - 12.0
        card_rect = QRectF(14.0, card_y, self.WIDTH - 28.0, card_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(card_rect, 6.0, 6.0)

        # Inspect current cell math
        raw_val = self.compute_single_cell_convolution(self._window_r, self._window_c)
        act_val = self.apply_activation(raw_val)

        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(
            QRectF(22.0, card_y + 6.0, card_rect.width() - 16.0, 16.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"Vị trí cửa sổ (r={self._window_r}, c={self._window_c}): Tích chập S = (I ∗ K) = {raw_val:+.1f} ➔ {self._activation}(S) = {act_val:.1f}",
        )

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc_line = (
            f"• Định nghĩa toán học: S(i, j) = ∑∑ I(i+u, j+v) · K(u+1, v+1) | Kích thước xuất: O = (W - K + 2P)/S + 1\n"
            f"• Vai trò sư phạm: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, card_y + 24.0, card_rect.width() - 16.0, 36.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc_line,
        )
