"""
AirTeach AI — Interactive Multi-Layer Perceptron (MLP), Backpropagation & Decision Boundary Simulator
Authentic Rumelhart-Hinton-Williams 1986 gradient descent, 2D decision boundary probability contours,
loss function convergence, non-linear activations, and touchless air gesture integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
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
class DatasetPreset:
    code: str
    name_vi: str
    description: str


PRESETS: dict[str, DatasetPreset] = {
    "XOR_PROBLEM": DatasetPreset(
        code="XOR_PROBLEM",
        name_vi="Bài toán XOR (Phi tuyến điển hình)",
        description="Minsky & Papert 1969 chứng minh Perceptron đơn lớp thất bại; bắt buộc cần tầng ẩn để phân tách phi tuyến.",
    ),
    "CONCENTRIC_CIRCLES": DatasetPreset(
        code="CONCENTRIC_CIRCLES",
        name_vi="Vòng tròn đồng tâm (Concentric Rings)",
        description="Lớp 0 nằm trong nhân hình tròn r < 0.5; lớp 1 bao quanh tạo thành vành khuyên khép kín.",
    ),
    "TWO_MOONS": DatasetPreset(
        code="TWO_MOONS",
        name_vi="Hai vầng trăng khuyết (Two Moons)",
        description="Hai đường cong lồng nhau phi tuyến uốn lượn; kiểm tra khả năng bẻ cong không gian đặc trưng.",
    ),
    "LINEAR_SEPARABLE": DatasetPreset(
        code="LINEAR_SEPARABLE",
        name_vi="Phân tách tuyến tính (Linearly Separable)",
        description="Hai cụm dữ liệu phân tách rời rạc; mạng học siêu phẳng phân chia trong vài epoch đầu tiên.",
    ),
    "SPIRAL_TWIN": DatasetPreset(
        code="SPIRAL_TWIN",
        name_vi="Xoắn ốc kép (Twin Spirals)",
        description="Cặp xoắn ốc đối xứng phức tạp đòi hỏi nhiều nơ-ron ẩn và hàm kích hoạt uốn dẻo để phân lớp hoàn hảo.",
    ),
}


class InteractiveMLPBackpropItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Multi-Layer Perceptron (MLP) forward propagation,
    backpropagation gradient descent, 2D decision boundary contours, and real-time loss tracking.
    """

    WIDTH = 680.0
    HEIGHT = 410.0

    stamp_requested = Signal()
    training_stepped = Signal(int, float, float)  # epoch, loss, accuracy

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

        # Network Hyperparameters
        self._n_input: int = 2
        self._n_hidden: int = 6
        self._n_output: int = 1
        self._learning_rate: float = 0.25
        self._momentum: float = 0.85
        self._activation_name: str = "Tanh"  # "Tanh", "ReLU", "Sigmoid"

        # Active Preset & Dataset
        self._active_preset_code: str = "XOR_PROBLEM"
        self._preset: DatasetPreset = PRESETS[self._active_preset_code]

        self._X: np.ndarray = np.zeros((4, 2), dtype=np.float32)
        self._Y: np.ndarray = np.zeros((4, 1), dtype=np.float32)

        # Network Parameters
        self._w1: np.ndarray = np.zeros((2, 6), dtype=np.float32)
        self._b1: np.ndarray = np.zeros((1, 6), dtype=np.float32)
        self._w2: np.ndarray = np.zeros((6, 1), dtype=np.float32)
        self._b2: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        # Momentum velocity buffers
        self._vw1: np.ndarray = np.zeros_like(self._w1)
        self._vb1: np.ndarray = np.zeros_like(self._b1)
        self._vw2: np.ndarray = np.zeros_like(self._w2)
        self._vb2: np.ndarray = np.zeros_like(self._b2)

        # Training history
        self._epoch: int = 0
        self._loss: float = 0.693
        self._accuracy: float = 0.50
        self._loss_history: list[float] = []

        # Decision boundary heatmap cache
        self._grid_res: int = 40
        self._grid_z: np.ndarray = np.zeros((self._grid_res, self._grid_res), dtype=np.float32)

        # Initialize network & data
        self.load_preset(self._active_preset_code)

        # Training loop timer (runs epochs when training)
        self._is_training: bool = False
        self._train_timer = QTimer()
        self._train_timer.setInterval(40)
        self._train_timer.timeout.connect(self._on_training_tick)

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Sets up interactive buttons, canvas, and graph layout rects."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0

        # 5 Dataset buttons
        preset_names = [
            ("XOR_PROBLEM", "XOR"),
            ("CONCENTRIC_CIRCLES", "Đồng tâm"),
            ("TWO_MOONS", "2 Trăng khuyết"),
            ("LINEAR_SEPARABLE", "Tuyến tính"),
            ("SPIRAL_TWIN", "Xoắn ốc"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 68.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 4.0

        # Controls
        r_play = QRectF(cur_x + 4.0, btn_y, 44.0, btn_h)
        self._btn_rects["toggle_train"] = r_play

        r_step = QRectF(r_play.right() + 4.0, btn_y, 42.0, btn_h)
        self._btn_rects["step_train"] = r_step

        r_lr_down = QRectF(r_step.right() + 4.0, btn_y, 30.0, btn_h)
        self._btn_rects["lr_down"] = r_lr_down

        r_lr_up = QRectF(r_lr_down.right() + 3.0, btn_y, 30.0, btn_h)
        self._btn_rects["lr_up"] = r_lr_up

        r_act = QRectF(r_lr_up.right() + 4.0, btn_y, 48.0, btn_h)
        self._btn_rects["toggle_activation"] = r_act

        r_reset = QRectF(r_act.right() + 4.0, btn_y, 36.0, btn_h)
        self._btn_rects["reset_weights"] = r_reset

        r_stamp = QRectF(r_reset.right() + 4.0, btn_y, 46.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        # Canvas layout
        self._boundary_rect = QRectF(16.0, 78.0, 205.0, 205.0)
        self._btn_rects["boundary_canvas"] = self._boundary_rect

        self._topology_rect = QRectF(232.0, 78.0, 230.0, 205.0)
        self._loss_rect = QRectF(474.0, 78.0, 192.0, 205.0)

    def cleanup(self) -> None:
        """Stops training timer to avoid leaks."""
        if self._train_timer.isActive():
            self._train_timer.stop()

    # -------------------------------------------------------------------------
    # Dataset Generation & Network Initialization
    # -------------------------------------------------------------------------
    def _generate_dataset(self, preset_code: str) -> tuple[np.ndarray, np.ndarray]:
        """Generates authentic 2D classification datasets."""
        rng = np.random.default_rng(42)
        if preset_code == "XOR_PROBLEM":
            x = np.array([[-0.8, -0.8], [0.8, 0.8], [-0.8, 0.8], [0.8, -0.8]], dtype=np.float32)
            y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
            # Add small cluster variations
            jitter = rng.normal(0.0, 0.08, size=(16, 2)).astype(np.float32)
            x_rep = np.repeat(x, 4, axis=0) + jitter
            y_rep = np.repeat(y, 4, axis=0)
            return x_rep, y_rep

        elif preset_code == "CONCENTRIC_CIRCLES":
            n = 30
            # Inner circle (class 0)
            r_in = rng.uniform(0.0, 0.45, size=n)
            th_in = rng.uniform(0, 2 * math.pi, size=n)
            x0 = np.stack([r_in * np.cos(th_in), r_in * np.sin(th_in)], axis=1)
            y0 = np.zeros((n, 1), dtype=np.float32)

            # Outer ring (class 1)
            r_out = rng.uniform(0.75, 1.15, size=n)
            th_out = rng.uniform(0, 2 * math.pi, size=n)
            x1 = np.stack([r_out * np.cos(th_out), r_out * np.sin(th_out)], axis=1)
            y1 = np.ones((n, 1), dtype=np.float32)

            x = np.vstack([x0, x1]).astype(np.float32)
            y = np.vstack([y0, y1]).astype(np.float32)
            return x, y

        elif preset_code == "TWO_MOONS":
            n = 28
            th = np.linspace(0, math.pi, n)
            x0 = np.stack([np.cos(th) - 0.4, np.sin(th) - 0.2], axis=1) + rng.normal(0, 0.06, (n, 2))
            y0 = np.zeros((n, 1), dtype=np.float32)

            x1 = np.stack([1.0 - np.cos(th) - 0.4, 1.0 - np.sin(th) - 0.7], axis=1) + rng.normal(0, 0.06, (n, 2))
            y1 = np.ones((n, 1), dtype=np.float32)

            x = np.vstack([x0, x1]).astype(np.float32)
            y = np.vstack([y0, y1]).astype(np.float32)
            return x, y

        elif preset_code == "LINEAR_SEPARABLE":
            n = 25
            x0 = rng.normal([-0.6, -0.6], 0.22, (n, 2)).astype(np.float32)
            y0 = np.zeros((n, 1), dtype=np.float32)
            x1 = rng.normal([0.6, 0.6], 0.22, (n, 2)).astype(np.float32)
            y1 = np.ones((n, 1), dtype=np.float32)
            return np.vstack([x0, x1]), np.vstack([y0, y1])

        else:  # SPIRAL_TWIN
            n = 30
            th = np.linspace(0, 3 * math.pi, n)
            r = np.linspace(0.1, 1.1, n)
            x0 = np.stack([r * np.cos(th), r * np.sin(th)], axis=1) + rng.normal(0, 0.04, (n, 2))
            y0 = np.zeros((n, 1), dtype=np.float32)
            x1 = np.stack([-r * np.cos(th), -r * np.sin(th)], axis=1) + rng.normal(0, 0.04, (n, 2))
            y1 = np.ones((n, 1), dtype=np.float32)
            return np.vstack([x0, x1]).astype(np.float32), np.vstack([y0, y1]).astype(np.float32)

    def _reset_network_weights(self) -> None:
        """Initializes weights with Xavier/He normal distribution."""
        rng = np.random.default_rng()
        # He initialization for hidden layer
        self._w1 = rng.normal(0.0, np.sqrt(2.0 / self._n_input), (self._n_input, self._n_hidden)).astype(np.float32)
        self._b1 = np.zeros((1, self._n_hidden), dtype=np.float32)

        # Xavier initialization for output layer
        self._w2 = rng.normal(0.0, np.sqrt(1.0 / self._n_hidden), (self._n_hidden, self._n_output)).astype(np.float32)
        self._b2 = np.zeros((1, self._n_output), dtype=np.float32)

        self._vw1.fill(0.0)
        self._vb1.fill(0.0)
        self._vw2.fill(0.0)
        self._vb2.fill(0.0)

        self._epoch = 0
        self._loss_history.clear()
        self._compute_loss_and_accuracy()
        self._compute_decision_boundary()

    def load_preset(self, preset_code: str) -> None:
        """Loads dataset preset and reinitializes network."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._X, self._Y = self._generate_dataset(preset_code)
        self._reset_network_weights()
        self.update()

    # -------------------------------------------------------------------------
    # Forward Propagation & Activations
    # -------------------------------------------------------------------------
    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self._activation_name == "ReLU":
            return np.maximum(0.0, z)
        elif self._activation_name == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        else:  # Tanh
            return np.tanh(z)

    def _activate_derivative(self, a: np.ndarray) -> np.ndarray:
        if self._activation_name == "ReLU":
            return (a > 0.0).astype(np.float32)
        elif self._activation_name == "Sigmoid":
            return a * (1.0 - a)
        else:  # Tanh
            return 1.0 - a ** 2

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Runs forward propagation pass."""
        z1 = np.dot(x, self._w1) + self._b1
        a1 = self._activate(z1)
        z2 = np.dot(a1, self._w2) + self._b2
        # Output layer with Sigmoid
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(z2, -20.0, 20.0)))
        return z1, a1, z2, y_hat

    # -------------------------------------------------------------------------
    # Backpropagation & Training Step
    # -------------------------------------------------------------------------
    def train_epochs(self, epochs: int = 10) -> None:
        """Executes authentic backpropagation gradient descent with momentum."""
        m = self._X.shape[0]
        lr = self._learning_rate
        beta = self._momentum

        for _ in range(epochs):
            z1, a1, z2, y_hat = self._forward(self._X)

            # Output error (Binary Cross-Entropy with Sigmoid)
            dz2 = (y_hat - self._Y) / m
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            # Hidden layer error
            da1 = np.dot(dz2, self._w2.T)
            dz1 = da1 * self._activate_derivative(a1)
            dw1 = np.dot(self._X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Momentum updates
            self._vw2 = beta * self._vw2 + (1.0 - beta) * dw2
            self._vb2 = beta * self._vb2 + (1.0 - beta) * db2
            self._vw1 = beta * self._vw1 + (1.0 - beta) * dw1
            self._vb1 = beta * self._vb1 + (1.0 - beta) * db1

            self._w2 -= lr * self._vw2
            self._b2 -= lr * self._vb2
            self._w1 -= lr * self._vw1
            self._b1 -= lr * self._vb1

            self._epoch += 1

        self._compute_loss_and_accuracy()
        self._compute_decision_boundary()
        self.training_stepped.emit(self._epoch, self._loss, self._accuracy)
        self.update()

    def _compute_loss_and_accuracy(self) -> None:
        """Calculates binary cross-entropy loss and classification accuracy."""
        _, _, _, y_hat = self._forward(self._X)
        eps = 1e-7
        bce = -np.mean(self._Y * np.log(y_hat + eps) + (1.0 - self._Y) * np.log(1.0 - y_hat + eps))
        self._loss = float(bce)
        self._loss_history.append(self._loss)
        if len(self._loss_history) > 80:
            self._loss_history.pop(0)

        preds = (y_hat >= 0.5).astype(np.float32)
        acc = np.mean(preds == self._Y)
        self._accuracy = float(acc)

    def _compute_decision_boundary(self) -> None:
        """Evaluates network over a 40x40 spatial grid for decision contour heatmap."""
        coords = np.linspace(-1.5, 1.5, self._grid_res, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(coords, coords)
        grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        _, _, _, y_grid = self._forward(grid_pts)
        self._grid_z = y_grid.reshape((self._grid_res, self._grid_res))

    def _on_training_tick(self) -> None:
        """Timer callback when continuous training is active."""
        if not self._is_training:
            return
        self.train_epochs(15)

    def toggle_training(self) -> bool:
        """Toggles continuous training timer."""
        self._is_training = not self._is_training
        if self._is_training:
            self._train_timer.start()
        else:
            self._train_timer.stop()
        self.update()
        return self._is_training

    def toggle_activation(self) -> str:
        """Cycles between Tanh -> ReLU -> Sigmoid."""
        order = ["Tanh", "ReLU", "Sigmoid"]
        idx = order.index(self._activation_name)
        self._activation_name = order[(idx + 1) % len(order)]
        self._reset_network_weights()
        self.update()
        return self._activation_name

    def set_learning_rate(self, lr: float) -> None:
        """Sets learning rate clamped between 0.01 and 1.5."""
        self._learning_rate = max(0.01, min(1.5, float(lr)))
        self.update()

    # -------------------------------------------------------------------------
    # Properties & Getters
    # -------------------------------------------------------------------------
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def activation_name(self) -> str:
        return self._activation_name

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def active_preset(self) -> DatasetPreset:
        return self._preset

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active MLP network & decision boundary diagram to LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import InformaticsTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = InformaticsTools.create_mlp_backprop_diagram(
            center=center,
            preset_code=self._active_preset_code,
            epoch=self._epoch,
            loss=self._loss,
            accuracy=self._accuracy,
            activation=self._activation_name,
            lr=self._learning_rate,
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

        # 2. Play/Pause continuous training
        r_play = self._btn_rects.get("toggle_train")
        if r_play and r_play.contains(local_pos):
            self.toggle_training()
            return True

        # 3. Single step (10 epochs)
        r_step = self._btn_rects.get("step_train")
        if r_step and r_step.contains(local_pos):
            self.train_epochs(10)
            return True

        # 4. Learning rate steppers
        r_lrd = self._btn_rects.get("lr_down")
        if r_lrd and r_lrd.contains(local_pos):
            self.set_learning_rate(self._learning_rate - 0.05)
            return True

        r_lru = self._btn_rects.get("lr_up")
        if r_lru and r_lru.contains(local_pos):
            self.set_learning_rate(self._learning_rate + 0.05)
            return True

        # 5. Toggle Activation
        r_act = self._btn_rects.get("toggle_activation")
        if r_act and r_act.contains(local_pos):
            self.toggle_activation()
            return True

        # 6. Reset Weights
        r_rst = self._btn_rects.get("reset_weights")
        if r_rst and r_rst.contains(local_pos):
            self._reset_network_weights()
            self.update()
            return True

        # 7. Stamp to LessonBoard
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
            "🧠 MẠNG NƠ-RON MLP & LAN TRUYỀN NGƯỢC (BACKPROPAGATION & LOSS SURFACE)",
        )

        # Preset Badge
        p_badge = QRectF(self.WIDTH - 162.0, 7.0, 148.0, 22.0)
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.setPen(QPen(QColor("#818cf8"), 1.0))
        painter.drawRoundedRect(p_badge, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#c7d2fe"))
        painter.drawText(p_badge, Qt.AlignmentFlag.AlignCenter, self._preset.name_vi[:24])

        # 3. Preset Buttons & Controls
        preset_items = [
            ("XOR_PROBLEM", "XOR"),
            ("CONCENTRIC_CIRCLES", "Đồng tâm"),
            ("TWO_MOONS", "Trăng khuyết"),
            ("LINEAR_SEPARABLE", "Tuyến tính"),
            ("SPIRAL_TWIN", "Xoắn ốc"),
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

        # Train Toggle
        r_play = self._btn_rects["toggle_train"]
        painter.setBrush(QBrush(QColor("#065f46") if self._is_training else QColor("#334155")))
        painter.setPen(QPen(QColor("#10b981") if self._is_training else QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_play, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, "▶ Huấn luyện" if not self._is_training else "⏸ Tạm dừng")

        # Step Button
        r_step = self._btn_rects["step_train"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(r_step, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_step, Qt.AlignmentFlag.AlignCenter, "+10 E")

        # LR steppers
        r_lrd = self._btn_rects["lr_down"]
        painter.drawRoundedRect(r_lrd, 3.0, 3.0)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_lrd, Qt.AlignmentFlag.AlignCenter, "η-")

        r_lru = self._btn_rects["lr_up"]
        painter.drawRoundedRect(r_lru, 3.0, 3.0)
        painter.setPen(QColor("#ef4444"))
        painter.drawText(r_lru, Qt.AlignmentFlag.AlignCenter, "η+")

        # Activation button
        r_act = self._btn_rects["toggle_activation"]
        painter.setBrush(QBrush(QColor("#2e1065")))
        painter.setPen(QPen(QColor("#9333ea"), 1.0))
        painter.drawRoundedRect(r_act, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#d8b4fe"))
        painter.drawText(r_act, Qt.AlignmentFlag.AlignCenter, self._activation_name)

        # Reset button
        r_rst = self._btn_rects["reset_weights"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(r_rst, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(r_rst, Qt.AlignmentFlag.AlignCenter, "Reset")

        # Stamp button
        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#831843")))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # ---------------------------------------------------------------------
        # 4. Left Panel: 2D Decision Boundary Probability Heatmap
        # ---------------------------------------------------------------------
        bx, by, bw, bh = self._boundary_rect.x(), self._boundary_rect.y(), self._boundary_rect.width(), self._boundary_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._boundary_rect, 6.0, 6.0)

        # Render 2D decision boundary image from self._grid_z
        # Generate QImage directly
        img = QImage(self._grid_res, self._grid_res, QImage.Format.Format_ARGB32)
        for r_i in range(self._grid_res):
            for c_i in range(self._grid_res):
                val = float(self._grid_z[r_i, c_i])  # 0.0 (blue) to 1.0 (orange)
                # Map to RGB
                r_c = int(249 * val + 30 * (1.0 - val))
                g_c = int(115 * val + 58 * (1.0 - val))
                b_c = int(22 * val + 138 * (1.0 - val))
                img.setPixelColor(c_i, self._grid_res - 1 - r_i, QColor(r_c, g_c, b_c, 160))

        painter.drawImage(QRectF(bx + 1.0, by + 1.0, bw - 2.0, bh - 2.0), img)

        # Draw axes
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        painter.setPen(QPen(QColor("#475569"), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(bx, cy), QPointF(bx + bw, cy))
        painter.drawLine(QPointF(cx, by), QPointF(cx, by + bh))

        # Render training data points
        for i in range(len(self._X)):
            px = cx + (self._X[i, 0] / 1.5) * (bw / 2.0)
            py = cy - (self._X[i, 1] / 1.5) * (bh / 2.0)
            is_class_1 = bool(self._Y[i, 0] > 0.5)

            pt_col = QColor("#f97316") if is_class_1 else QColor("#38bdf8")
            painter.setBrush(QBrush(pt_col))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            if is_class_1:
                painter.drawRect(QRectF(px - 3.5, py - 3.5, 7.0, 7.0))
            else:
                painter.drawEllipse(QPointF(px, py), 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(bx + 8.0, by + 16.0), "BIÊN QUYẾT ĐỊNH 2D (DECISION BOUNDARY)")

        # ---------------------------------------------------------------------
        # 5. Center Panel: Network Topology & Synaptic Weight Vectors
        # ---------------------------------------------------------------------
        tx, ty, tw, th = self._topology_rect.x(), self._topology_rect.y(), self._topology_rect.width(), self._topology_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._topology_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(tx + 8.0, ty + 16.0), "CẤU TRÚC MẠNG NƠ-RON (2-6-1)")

        # Layer coordinates
        in_x = tx + 32.0
        hid_x = tx + tw / 2.0
        out_x = tx + tw - 32.0

        in_ys = [ty + th * 0.35, ty + th * 0.65]
        hid_ys = [ty + 34.0 + i * (th - 50.0) / 5.0 for i in range(6)]
        out_ys = [ty + th * 0.5]

        # Draw Synaptic Connections (Input -> Hidden)
        for i_idx, iy in enumerate(in_ys):
            for h_idx, hy in enumerate(hid_ys):
                w_val = float(self._w1[i_idx, h_idx])
                w_thick = max(0.5, min(3.5, abs(w_val) * 1.5))
                w_col = QColor("#38bdf8") if w_val > 0 else QColor("#f43f5e")
                painter.setPen(QPen(w_col, w_thick))
                painter.drawLine(QPointF(in_x, iy), QPointF(hid_x, hy))

        # Draw Synaptic Connections (Hidden -> Output)
        for h_idx, hy in enumerate(hid_ys):
            for o_idx, oy in enumerate(out_ys):
                w_val = float(self._w2[h_idx, o_idx])
                w_thick = max(0.5, min(3.5, abs(w_val) * 1.5))
                w_col = QColor("#38bdf8") if w_val > 0 else QColor("#f43f5e")
                painter.setPen(QPen(w_col, w_thick))
                painter.drawLine(QPointF(hid_x, hy), QPointF(out_x, oy))

        # Render Nodes
        # Input Layer
        for i_idx, iy in enumerate(in_ys):
            painter.setBrush(QBrush(QColor("#0ea5e9")))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.drawEllipse(QPointF(in_x, iy), 9.0, 9.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(in_x - 9.0, iy - 9.0, 18.0, 18.0), Qt.AlignmentFlag.AlignCenter, f"x{i_idx+1}")

        # Hidden Layer
        for h_idx, hy in enumerate(hid_ys):
            painter.setBrush(QBrush(QColor("#8b5cf6")))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.drawEllipse(QPointF(hid_x, hy), 8.0, 8.0)
            painter.setFont(QFont("Segoe UI", 5, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(hid_x - 8.0, hy - 8.0, 16.0, 16.0), Qt.AlignmentFlag.AlignCenter, f"h{h_idx+1}")

        # Output Layer
        for o_idx, oy in enumerate(out_ys):
            painter.setBrush(QBrush(QColor("#f97316")))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(QPointF(out_x, oy), 10.0, 10.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(out_x - 10.0, oy - 10.0, 20.0, 20.0), Qt.AlignmentFlag.AlignCenter, "ŷ")

        # ---------------------------------------------------------------------
        # 6. Right Panel: Real-time Loss Curve & Metrics Ledger
        # ---------------------------------------------------------------------
        lx, ly, lw, lh = self._loss_rect.x(), self._loss_rect.y(), self._loss_rect.width(), self._loss_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._loss_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(lx + 8.0, ly + 16.0), "ĐỒ THỊ HÀM MẤT MÁT (LOSS)")

        # Draw Loss History Graph Box
        g_box = QRectF(lx + 8.0, ly + 24.0, lw - 16.0, 75.0)
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRect(g_box)

        # Plot loss curve
        if len(self._loss_history) > 1:
            l_path = QPainterPath()
            max_l = max(1.0, max(self._loss_history))
            min_l = 0.0
            n_pts = len(self._loss_history)
            for p_i, l_val in enumerate(self._loss_history):
                gx = g_box.left() + (p_i / (n_pts - 1)) * g_box.width()
                norm_y = (l_val - min_l) / (max_l - min_l + 1e-6)
                gy = g_box.bottom() - norm_y * g_box.height()
                if p_i == 0:
                    l_path.moveTo(gx, gy)
                else:
                    l_path.lineTo(gx, gy)

            painter.setPen(QPen(QColor("#ef4444"), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(l_path)

        # Metrics text
        metrics = [
            (f"• Epoch: {self._epoch}", QColor("#f8fafc")),
            (f"• Mất mát (BCE): {self._loss:.4f}", QColor("#ef4444") if self._loss > 0.3 else QColor("#10b981")),
            (f"• Độ chính xác: {self._accuracy * 100:.1f}%", QColor("#10b981") if self._accuracy > 0.85 else QColor("#fbbf24")),
            (f"• Tốc độ học η: {self._learning_rate:.2f}", QColor("#38bdf8")),
            (f"• Kích hoạt: {self._activation_name}", QColor("#a78bfa")),
        ]

        my = ly + 115.0
        for m_txt, m_col in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(m_col)
            painter.drawText(QPointF(lx + 8.0, my), m_txt)
            my += 16.0

        # Accuracy Progress Bar
        bar_r = QRectF(lx + 8.0, ly + lh - 22.0, lw - 16.0, 14.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(bar_r, 3.0, 3.0)

        fill_w = max(0.0, (lw - 16.0) * self._accuracy)
        fill_r = QRectF(lx + 8.0, ly + lh - 22.0, fill_w, 14.0)
        painter.setBrush(QBrush(QColor("#10b981") if self._accuracy > 0.85 else QColor("#f59e0b")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(fill_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(bar_r, Qt.AlignmentFlag.AlignCenter, f"Accuracy: {self._accuracy * 100:.0f}%")

        # ---------------------------------------------------------------------
        # 7. Bottom Panel: Rumelhart Backprop & Hinton Chain Rule Principles
        # ---------------------------------------------------------------------
        bot_y = self._boundary_rect.bottom() + 10.0
        bot_h = self.HEIGHT - bot_y - 10.0
        bot_rect = QRectF(14.0, bot_y, self.WIDTH - 28.0, bot_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(bot_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(22.0, bot_y + 16.0), "NGUYÊN LÝ LAN TRUYỀN NGƯỢC (RUMELHART, HINTON & WILLIAMS, 1986):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        b_desc = (
            f"• Quy tắc đạo hàm chuỗi (Chain Rule): δ^[l] = (W^[l+1]T δ^[l+1]) ⊙ g'(z^[l]). Gradient lỗi lan truyền từ đầu ra ngược về đầu vào.\n"
            f"• Tối ưu Gradient Descent kết hợp Momentum: v_t = β v_(t-1) + (1-β) ∇L, W ← W - η v_t giúp vượt qua cực tiểu địa phương.\n"
            f"• Bộ dữ liệu: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, bot_y + 22.0, bot_rect.width() - 20.0, 58.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            b_desc,
        )
