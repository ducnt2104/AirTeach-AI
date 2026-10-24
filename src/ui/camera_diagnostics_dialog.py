"""
AirTeach AI - Camera Diagnostics & Optical Quality Monitor
==========================================================
Displays real-time diagnostic telemetry, live preview with skeleton overlay,
optical quality measurements (brightness, contrast, clipping, sharpness),
lighting condition classification, and controls for camera enhancement modes.

Zero Fake Data: All metrics are computed in real time by AdaptiveVisionPipeline.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.adaptive_vision_pipeline import CameraDiagnosticsData
from src.ui.camera_widget import CameraWidget

logger = logging.getLogger(__name__)


class CameraDiagnosticsDialog(QDialog):
    """
    Diagnostic dashboard for camera hardware, image quality, and tracking performance.
    """

    lighting_mode_changed = Signal(str)
    calibration_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chẩn đoán Camera & Chất lượng Thị giác máy tính — AirTeach AI")
        self.setMinimumSize(780, 560)
        self.setModal(False)
        self._build_ui()

    def _build_ui(self) -> None:
        root_v = QVBoxLayout(self)
        root_v.setContentsMargins(16, 16, 16, 16)
        root_v.setSpacing(12)

        # Header
        header = QLabel("🔬 Bảng điều khiển Chẩn đoán Camera & Thị giác máy tính")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        root_v.addWidget(header)

        # Main horizontal split: Left is live preview, Right is telemetry cards
        main_h = QHBoxLayout()
        main_h.setSpacing(16)

        # Left Column: Camera Preview
        preview_box = QGroupBox("Hình ảnh thời gian thực (Live Preview)")
        preview_layout = QVBoxLayout(preview_box)
        self.cam_widget = CameraWidget()
        self.cam_widget.setMinimumSize(360, 270)
        preview_layout.addWidget(self.cam_widget)

        # Mode controls below preview
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Chế độ xử lý:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Tự động thích ứng (Adaptive)", "adaptive")
        self.combo_mode.addItem("Phòng thiếu sáng (Low Light)", "low_light")
        self.combo_mode.addItem("Ngược sáng (Balanced CLAHE)", "balanced")
        self.combo_mode.addItem("Ưu tiên tốc độ (Motion Priority)", "motion_priority")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.combo_mode, stretch=1)
        preview_layout.addLayout(mode_row)

        btn_calib = QPushButton("🎯 Mở Trình hiệu chỉnh Cử chỉ (Wizard)")
        btn_calib.setStyleSheet("background: #e8f0fe; color: #1a73e8; font-weight: 600; padding: 6px;")
        btn_calib.clicked.connect(self._on_calib_clicked)
        preview_layout.addWidget(btn_calib)

        main_h.addWidget(preview_box, stretch=3)

        # Right Column: Real-time Telemetry Cards
        telemetry_box = QGroupBox("Chỉ số đo lường thực tế (Hardware & Vision Metrics)")
        telemetry_form = QFormLayout(telemetry_box)
        telemetry_form.setSpacing(8)

        # Optical Metrics
        self.lbl_fps = QLabel("— FPS")
        self.lbl_fps.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        telemetry_form.addRow("Tốc độ khung hình (FPS):", self.lbl_fps)

        self.lbl_resolution = QLabel("— x —")
        self.lbl_resolution.setFont(QFont("Consolas", 10))
        telemetry_form.addRow("Độ phân giải khung hình:", self.lbl_resolution)

        self.lbl_condition = QLabel("—")
        self.lbl_condition.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #e8f0fe; color: #1a73e8; font-weight: bold;")
        telemetry_form.addRow("Điều kiện ánh sáng phát hiện:", self.lbl_condition)

        self.lbl_brightness = QLabel("— / 255")
        telemetry_form.addRow("Độ sáng trung bình (Luminance):", self.lbl_brightness)

        self.lbl_sharpness = QLabel("— (Laplacian σ²)")
        telemetry_form.addRow("Chỉ số độ sắc nét (Focus/Blur):", self.lbl_sharpness)

        self.lbl_shadow_clip = QLabel("0.0%")
        telemetry_form.addRow("Tỷ lệ vùng tối (Under-exposed):", self.lbl_shadow_clip)

        self.lbl_highlight_clip = QLabel("0.0%")
        telemetry_form.addRow("Tỷ lệ vùng cháy sáng (Glare):", self.lbl_highlight_clip)

        # Hand Tracking Metrics
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        telemetry_form.addRow(sep)

        self.lbl_hands = QLabel("0 bàn tay")
        self.lbl_hands.setFont(QFont("Consolas", 10))
        telemetry_form.addRow("Bàn tay phát hiện:", self.lbl_hands)

        self.lbl_confidence = QLabel("0.00")
        self.lbl_confidence.setFont(QFont("Consolas", 10))
        telemetry_form.addRow("Độ tin cậy nhận diện:", self.lbl_confidence)

        self.lbl_gesture = QLabel("NONE")
        self.lbl_gesture.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #202124; color: #00ff88; font-weight: bold; font-family: Consolas;")
        telemetry_form.addRow("Trạng thái cử chỉ (FSM):", self.lbl_gesture)

        self.lbl_draw_mode = QLabel("INDEX_UP (Air Pen)")
        telemetry_form.addRow("Phương thức vẽ hiện tại:", self.lbl_draw_mode)

        self.lbl_latency = QLabel("0.0 ms")
        self.lbl_latency.setFont(QFont("Consolas", 10))
        telemetry_form.addRow("Độ trễ xử lý (Latency):", self.lbl_latency)

        self.lbl_dropped = QLabel("0 khung hình")
        telemetry_form.addRow("Khung hình bị mất (Drops):", self.lbl_dropped)

        main_h.addWidget(telemetry_box, stretch=2)
        root_v.addLayout(main_h, stretch=1)

        # Footer Notice & Close Button
        footer_h = QHBoxLayout()
        tip_label = QLabel("💡 Lưu ý: Các chỉ số được đo trực tiếp từ cảm biến camera để hỗ trợ giảng dạy tối ưu.")
        tip_label.setStyleSheet("color: #5f6368; font-size: 11px;")
        footer_h.addWidget(tip_label)

        footer_h.addStretch(1)
        btn_close = QPushButton("Đóng")
        btn_close.clicked.connect(self.close)
        footer_h.addWidget(btn_close)
        root_v.addLayout(footer_h)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(object)
    def update_diagnostics(self, diag: CameraDiagnosticsData) -> None:
        """Update telemetry UI labels with real measured metrics from background worker."""
        self.lbl_fps.setText(f"{diag.fps:.1f} FPS")
        self.lbl_resolution.setText(f"{diag.width} × {diag.height} px")
        self.lbl_condition.setText(diag.condition_name)

        # Condition styling
        if diag.condition_name == "LOW_LIGHT":
            self.lbl_condition.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #feefe3; color: #b06000; font-weight: bold;")
        elif diag.condition_name == "BACKLIGHT":
            self.lbl_condition.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #fce8e6; color: #c5221f; font-weight: bold;")
        elif diag.condition_name == "NORMAL":
            self.lbl_condition.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #e6f4ea; color: #137333; font-weight: bold;")
        else:
            self.lbl_condition.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #e8f0fe; color: #1a73e8; font-weight: bold;")

        self.lbl_brightness.setText(f"{diag.mean_brightness:.1f} / 255")
        sharpness_badge = "✅ Tốt" if diag.sharpness_score >= 45.0 else "⚠️ Nhòe/Mờ"
        self.lbl_sharpness.setText(f"{diag.sharpness_score:.1f} ({sharpness_badge})")
        self.lbl_shadow_clip.setText(f"{diag.shadow_clipping_pct:.1f}%")
        self.lbl_highlight_clip.setText(f"{diag.highlight_clipping_pct:.1f}%")

        self.lbl_hands.setText(f"{diag.hands_detected} bàn tay")
        self.lbl_confidence.setText(f"{diag.tracking_confidence:.2f}")

        # Gesture badge
        self.lbl_gesture.setText(diag.gesture_state_name)
        if diag.gesture_state_name == "DRAW":
            self.lbl_gesture.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #d93025; color: #ffffff; font-weight: bold; font-family: Consolas;")
        elif diag.gesture_state_name == "POINT":
            self.lbl_gesture.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #188038; color: #ffffff; font-weight: bold; font-family: Consolas;")
        else:
            self.lbl_gesture.setStyleSheet("padding: 2px 6px; border-radius: 4px; background: #202124; color: #00ff88; font-weight: bold; font-family: Consolas;")

        self.lbl_draw_mode.setText(diag.draw_mode_name)
        self.lbl_latency.setText(f"{diag.pipeline_latency_ms:.1f} ms")
        self.lbl_dropped.setText(f"{diag.frame_dropped_count} frames")

    def _on_mode_changed(self) -> None:
        mode = self.combo_mode.currentData()
        self.lighting_mode_changed.emit(mode)

    def _on_calib_clicked(self) -> None:
        self.calibration_requested.emit()
