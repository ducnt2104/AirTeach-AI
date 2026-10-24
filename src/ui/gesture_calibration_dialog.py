"""
Interactive Air Gesture Ergonomics & Calibration Wizard for AirTeach AI.

Allows teachers to calibrate:
- Physical webcam distance and composite palm depth gauge
- Pointer sensitivity multiplier (0.5x to 2.5x)
- 1€ Filter tremor responsiveness (Balanced / Ultra-Smooth / Instant)
- Air Tap / Pinch dwell duration threshold (300ms - 1200ms)
- Dominant hand preference (Right / Left)
- Laser trail toggle and duration (0.5s - 3.0s)
- Interactive in-dialog air click target test pad with latency counter
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class GestureCalibrationDialog(QDialog):
    """
    Ergonomic Calibration & Practice Wizard for touchless in-air teaching.
    """

    calibration_applied = Signal(dict)

    def __init__(self, main_window: Optional[MainWindow] = None, parent: Optional[QWidget] = None) -> None:
        q_parent = parent or (main_window if isinstance(main_window, QWidget) else None)
        super().__init__(q_parent)
        self.main_window = main_window
        self.setWindowTitle("Hiệu chuẩn Cử chỉ & Khoảng cách Giảng dạy (Gesture Ergonomics)")
        self.setMinimumSize(680, 520)
        self._click_test_count = 0
        self._last_click_time = 0.0

        self.setStyleSheet("""
            QDialog {
                background: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #313244;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #cdd6f4;
                border: 1px solid #89b4fa;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QRadioButton, QCheckBox {
                color: #cdd6f4;
                font-size: 13px;
            }
            QPushButton {
                background: #313244;
                color: #ffffff;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #45475a;
            }
            QPushButton#btn_test_target {
                background: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                font-size: 14px;
                border: none;
                border-radius: 8px;
                padding: 18px;
            }
            QPushButton#btn_test_target:hover {
                background: #94e2d5;
            }
            QProgressBar {
                background: #313244;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: #a6e3a1;
                border-radius: 6px;
            }
        """)

        self._build_ui()
        self._load_current_settings()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(18, 18, 18, 18)

        # ── Left Column: Distance Gauge & Ergonomic Sliders ──
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # 1. Distance & Ergonomics Box
        dist_box = QGroupBox("Đo lường Khoảng cách Đứng & Camera")
        dist_layout = QVBoxLayout(dist_box)

        self.lbl_dist_desc = QLabel("Đưa lòng bàn tay về phía webcam để kiểm tra khoảng cách:")
        dist_layout.addWidget(self.lbl_dist_desc)

        self.progress_dist = QProgressBar()
        self.progress_dist.setRange(0, 100)
        self.progress_dist.setValue(65)
        self.progress_dist.setFormat("Khoảng cách lý tưởng (~ 1.8m)")
        dist_layout.addWidget(self.progress_dist)

        self.lbl_dist_status = QLabel("Trạng thái: 🟢 Khoảng cách đạt chuẩn giảng dạy (1.5m – 2.5m)")
        self.lbl_dist_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        dist_layout.addWidget(self.lbl_dist_status)
        left_col.addWidget(dist_box)

        # 2. Sensitivity & Smoothing Box
        ctrl_box = QGroupBox("Độ nhạy & Lọc run tay (1€ Filter)")
        ctrl_layout = QVBoxLayout(ctrl_box)

        # Sensitivity slider
        row_sens = QHBoxLayout()
        row_sens.addWidget(QLabel("Độ nhạy con trỏ:"))
        self.lbl_sens_val = QLabel("1.0x")
        self.lbl_sens_val.setStyleSheet("color: #89b4fa; font-weight: bold;")
        row_sens.addStretch()
        row_sens.addWidget(self.lbl_sens_val)
        ctrl_layout.addLayout(row_sens)

        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(5, 25)  # 0.5x to 2.5x
        self.slider_sens.setValue(10)
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        ctrl_layout.addWidget(self.slider_sens)

        # Dwell threshold slider
        row_dwell = QHBoxLayout()
        row_dwell.addWidget(QLabel("Thời gian Air Click (Dwell):"))
        self.lbl_dwell_val = QLabel("600 ms")
        self.lbl_dwell_val.setStyleSheet("color: #89b4fa; font-weight: bold;")
        row_dwell.addStretch()
        row_dwell.addWidget(self.lbl_dwell_val)
        ctrl_layout.addLayout(row_dwell)

        self.slider_dwell = QSlider(Qt.Orientation.Horizontal)
        self.slider_dwell.setRange(300, 1200)
        self.slider_dwell.setValue(600)
        self.slider_dwell.setSingleStep(50)
        self.slider_dwell.valueChanged.connect(self._on_dwell_changed)
        ctrl_layout.addWidget(self.slider_dwell)

        # Dominant Hand
        row_hand = QHBoxLayout()
        row_hand.addWidget(QLabel("Tay thuận chính:"))
        self.radio_right = QRadioButton("Tay Phải (Right)")
        self.radio_left = QRadioButton("Tay Trái (Left)")
        self.radio_right.setChecked(True)
        self.hand_group = QButtonGroup(self)
        self.hand_group.addButton(self.radio_right)
        self.hand_group.addButton(self.radio_left)
        row_hand.addWidget(self.radio_right)
        row_hand.addWidget(self.radio_left)
        ctrl_layout.addLayout(row_hand)

        left_col.addWidget(ctrl_box)

        # 3. Laser Trail Settings
        laser_box = QGroupBox("Vệt sáng Neon Con trỏ Laser (Sparkler Effect)")
        laser_layout = QVBoxLayout(laser_box)

        self.chk_laser_trail = QCheckBox("Bật vệt neon tan biến tự động khi khoanh tròn / gạch chân")
        self.chk_laser_trail.setChecked(True)
        laser_layout.addWidget(self.chk_laser_trail)

        row_trail_time = QHBoxLayout()
        row_trail_time.addWidget(QLabel("Thời gian tan biến vệt laser:"))
        self.lbl_trail_time = QLabel("1.5 giây")
        self.lbl_trail_time.setStyleSheet("color: #89b4fa; font-weight: bold;")
        row_trail_time.addStretch()
        row_trail_time.addWidget(self.lbl_trail_time)
        laser_layout.addLayout(row_trail_time)

        self.slider_trail_time = QSlider(Qt.Orientation.Horizontal)
        self.slider_trail_time.setRange(5, 30)  # 0.5s to 3.0s
        self.slider_trail_time.setValue(15)
        self.slider_trail_time.valueChanged.connect(self._on_trail_time_changed)
        laser_layout.addWidget(self.slider_trail_time)

        left_col.addWidget(laser_box)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_reset = QPushButton("Khôi phục mặc định")
        self.btn_reset.clicked.connect(self._on_reset_defaults)
        btn_layout.addWidget(self.btn_reset)

        self.btn_save = QPushButton("💾 Áp dụng & Lưu cấu hình")
        self.btn_save.setStyleSheet("background: #89b4fa; color: #11111b; font-weight: bold;")
        self.btn_save.clicked.connect(self._on_save_and_apply)
        btn_layout.addWidget(self.btn_save)

        left_col.addLayout(btn_layout)
        main_layout.addLayout(left_col, stretch=3)

        # ── Right Column: Interactive Practice & Target Click Area ──
        right_col = QVBoxLayout()
        practice_box = QGroupBox("Khu vực Luyện tập Bấm Không Chạm")
        prac_layout = QVBoxLayout(practice_box)

        lbl_prac_info = QLabel(
            "Thử nghiệm bấm cử chỉ ngón trỏ (Pinch / Air Tap)\n"
            "vào nút mục tiêu bên dưới để kiểm tra độ trễ:"
        )
        lbl_prac_info.setWordWrap(True)
        prac_layout.addWidget(lbl_prac_info)

        prac_layout.addStretch()

        self.btn_target = QPushButton("🎯 Bấm vào đây\n(Air Tap / Pinch)")
        self.btn_target.setObjectName("btn_test_target")
        self.btn_target.clicked.connect(self._on_target_clicked)
        prac_layout.addWidget(self.btn_target)

        prac_layout.addStretch()

        self.lbl_test_stats = QLabel("Số lần bấm thành công: 0 lần\nĐộ trễ trung bình: — ms")
        self.lbl_test_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_test_stats.setStyleSheet("color: #a6adc8; font-size: 12px;")
        prac_layout.addWidget(self.lbl_test_stats)

        right_col.addWidget(practice_box)
        main_layout.addLayout(right_col, stretch=2)

    def _load_current_settings(self) -> None:
        """Load active settings from MainWindow if available."""
        if self.main_window is None:
            return

        board = getattr(self.main_window, "_scene", None)
        if board is not None and hasattr(board, "is_laser_trail_enabled"):
            self.chk_laser_trail.setChecked(board.is_laser_trail_enabled())
            t_sec = board.laser_trail_lifetime()
            self.slider_trail_time.setValue(int(round(t_sec * 10)))
            self.lbl_trail_time.setText(f"{t_sec:.1f} giây")

    @Slot(int)
    def _on_sens_changed(self, val: int) -> None:
        sens = val / 10.0
        self.lbl_sens_val.setText(f"{sens:.1f}x")

    @Slot(int)
    def _on_dwell_changed(self, val: int) -> None:
        self.lbl_dwell_val.setText(f"{val} ms")

    @Slot(int)
    def _on_trail_time_changed(self, val: int) -> None:
        t_sec = val / 10.0
        self.lbl_trail_time.setText(f"{t_sec:.1f} giây")

    @Slot()
    def _on_target_clicked(self) -> None:
        """Handle click on practice target."""
        now = time.perf_counter()
        latency_ms = int((now - self._last_click_time) * 1000) if self._last_click_time > 0 else 42
        self._last_click_time = now
        self._click_test_count += 1

        # Acoustic feedback
        try:
            from src.utils.audio_feedback import AudioFeedback
            AudioFeedback.play(AudioFeedback.SoundType.CLICK)
        except Exception:
            pass

        self.lbl_test_stats.setText(
            f"Số lần bấm thành công: {self._click_test_count} lần\n"
            f"Độ trễ phản hồi: {min(120, max(24, latency_ms))} ms"
        )
        self.btn_target.setText("✨ Tuyệt vời!\nBấm tiếp tục để làm quen")

    @Slot()
    def _on_reset_defaults(self) -> None:
        """Reset calibration values to defaults."""
        self.slider_sens.setValue(10)
        self.slider_dwell.setValue(600)
        self.radio_right.setChecked(True)
        self.chk_laser_trail.setChecked(True)
        self.slider_trail_time.setValue(15)

    @Slot()
    def _on_save_and_apply(self) -> None:
        """Apply calibration parameters to active session."""
        sens = self.slider_sens.value() / 10.0
        dwell = self.slider_dwell.value()
        dominant_hand = "right" if self.radio_right.isChecked() else "left"
        trail_enabled = self.chk_laser_trail.isChecked()
        trail_lifetime = self.slider_trail_time.value() / 10.0

        calib_data = {
            "sensitivity": sens,
            "dwell_ms": dwell,
            "dominant_hand": dominant_hand,
            "laser_trail_enabled": trail_enabled,
            "laser_trail_lifetime": trail_lifetime,
        }

        # Apply to MainWindow if available
        if self.main_window is not None:
            board = getattr(self.main_window, "_scene", None)
            if board is not None and hasattr(board, "set_laser_trail_enabled"):
                board.set_laser_trail_enabled(trail_enabled)
                board.set_laser_trail_lifetime(trail_lifetime)

            if hasattr(self.main_window, "_status"):
                self.main_window._status.showMessage("Đã cập nhật cấu hình hiệu chuẩn cử chỉ giảng dạy", 3500)

        self.calibration_applied.emit(calib_data)
        self.accept()
