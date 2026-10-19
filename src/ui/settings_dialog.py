"""
AirTeach AI - Settings Dialog
==============================
QDialog for user-configurable settings:
  - Gemini API Key (update + validation)
  - Active camera index
  - Drawing tool defaults (ink color, min/max width)
  - Perspective calibration launcher
  - Stroke beautifier parameters (RDP epsilon, 1€ filter beta)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _read_env_key(key: str, default: str = "") -> str:
    """Read a single key from .env without importing dotenv (faster)."""
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line[len(key) + 1:].strip().strip('"').strip("'")
    return default


def _write_env_key(key: str, value: str) -> None:
    """Update (or insert) a key=value pair in .env."""
    lines: list[str] = []
    found = False
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SettingsDialog(QDialog):
    """
    Modal settings dialog.

    Signals:
        api_key_changed(str): emitted when the user saves a new Gemini API key.
        camera_index_changed(int): emitted when a new camera index is selected.
        calibration_requested(): emitted when the user clicks "Calibrate Perspective".
    """

    api_key_changed         = Signal(str)
    camera_index_changed    = Signal(int)
    calibration_requested   = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cài đặt AirTeach AI")
        self.setMinimumWidth(480)
        self.setModal(True)
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel("⚙️  Cài đặt")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        root.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._tab_ai(),      "🤖 AI & API")
        tabs.addTab(self._tab_camera(),  "📷 Camera")
        tabs.addTab(self._tab_drawing(), "✏️ Vẽ & Làm nét")
        root.addWidget(tabs)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ── Tab: AI & API ─────────────────────────────────────────────────────────

    def _tab_ai(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)

        # API Key
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("AIzaSy…  hoặc  AQ.Ab8…")
        current_key = _read_env_key("GEMINI_API_KEY")
        self._api_key_edit.setText(current_key)
        self._api_key_edit.setToolTip(
            "Nhập Gemini API Key từ Google AI Studio (aistudio.google.com)."
        )

        show_key_btn = QPushButton("👁")
        show_key_btn.setFixedWidth(32)
        show_key_btn.setCheckable(True)
        show_key_btn.toggled.connect(
            lambda checked: self._api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )

        key_row = QHBoxLayout()
        key_row.addWidget(self._api_key_edit)
        key_row.addWidget(show_key_btn)
        form.addRow("Gemini API Key:", key_row)

        # Model name
        self._model_edit = QLineEdit()
        self._model_edit.setText(_read_env_key("GEMINI_MODEL", "gemini-2.5-flash"))
        self._model_edit.setPlaceholderText("gemini-2.5-flash")
        form.addRow("Gemini Model:", self._model_edit)

        # Test connection button
        test_btn = QPushButton("🔗 Kiểm tra kết nối API")
        test_btn.clicked.connect(self._on_test_api)
        form.addRow("", test_btn)

        self._api_status_label = QLabel("")
        form.addRow("Trạng thái:", self._api_status_label)

        return w

    # ── Tab: Camera ───────────────────────────────────────────────────────────

    def _tab_camera(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)

        self._camera_index_spin = QSpinBox()
        self._camera_index_spin.setRange(0, 9)
        self._camera_index_spin.setValue(0)
        self._camera_index_spin.setToolTip("Camera index (0 = mặc định)")
        form.addRow("Chỉ số camera:", self._camera_index_spin)

        # CLAHE toggle
        self._clahe_check = QCheckBox("Kích hoạt tăng cường sáng tự động (CLAHE)")
        self._clahe_check.setChecked(True)
        form.addRow("", self._clahe_check)

        # Calibration
        calib_btn = QPushButton("📐 Hiệu chỉnh phối cảnh (Perspective)")
        calib_btn.clicked.connect(self._on_calibrate)
        form.addRow("Phối cảnh:", calib_btn)

        return w

    # ── Tab: Drawing ──────────────────────────────────────────────────────────

    def _tab_drawing(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)

        # Ink color picker
        self._ink_color = "#202124"
        self._ink_color_btn = QPushButton()
        self._ink_color_btn.setFixedWidth(80)
        self._update_color_btn()
        self._ink_color_btn.clicked.connect(self._on_pick_color)
        form.addRow("Màu nét mực:", self._ink_color_btn)

        # Min / max width
        self._min_width_spin = QDoubleSpinBox()
        self._min_width_spin.setRange(0.5, 5.0)
        self._min_width_spin.setSingleStep(0.5)
        self._min_width_spin.setValue(1.5)
        form.addRow("Độ dày nét tối thiểu:", self._min_width_spin)

        self._max_width_spin = QDoubleSpinBox()
        self._max_width_spin.setRange(1.0, 20.0)
        self._max_width_spin.setSingleStep(0.5)
        self._max_width_spin.setValue(6.0)
        form.addRow("Độ dày nét tối đa:", self._max_width_spin)

        # RDP epsilon
        self._rdp_spin = QDoubleSpinBox()
        self._rdp_spin.setRange(0.5, 20.0)
        self._rdp_spin.setSingleStep(0.5)
        self._rdp_spin.setValue(2.5)
        self._rdp_spin.setToolTip(
            "Ngưỡng RDP – giá trị nhỏ hơn giữ nhiều điểm hơn (nét chi tiết hơn)."
        )
        form.addRow("Ngưỡng đơn giản hóa RDP (ε):", self._rdp_spin)

        # 1€ Filter beta
        self._beta_spin = QDoubleSpinBox()
        self._beta_spin.setRange(0.01, 2.0)
        self._beta_spin.setSingleStep(0.05)
        self._beta_spin.setDecimals(3)
        self._beta_spin.setValue(0.350)
        self._beta_spin.setToolTip(
            "Tham số β của bộ lọc 1€ – giá trị lớn hơn → ít trễ hơn khi vẽ nhanh."
        )
        form.addRow("Bộ lọc 1€ – beta (β):", self._beta_spin)

        # Shape snapping toggle
        self._snap_check = QCheckBox("Tự động nắn hình học (Shape Snapping)")
        self._snap_check.setChecked(True)
        form.addRow("", self._snap_check)

        return w

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_pick_color(self) -> None:
        from PySide6.QtGui import QColor  # noqa: PLC0415
        col = QColorDialog.getColor(QColor(self._ink_color), self, "Chọn màu nét mực")
        if col.isValid():
            self._ink_color = col.name()
            self._update_color_btn()

    def _update_color_btn(self) -> None:
        self._ink_color_btn.setStyleSheet(
            f"background:{self._ink_color}; border:1px solid #ccc; border-radius:4px;"
        )

    def _on_test_api(self) -> None:
        key = self._api_key_edit.text().strip()
        if not key:
            self._api_status_label.setText("❌ Chưa nhập API Key.")
            return
        self._api_status_label.setText("⏳ Đang kiểm tra…")
        try:
            from google import genai  # noqa: PLC0415
            client = genai.Client(api_key=key)
            # Minimal ping: list models
            _ = list(client.models.list())
            self._api_status_label.setText("✅ Kết nối thành công!")
        except Exception as exc:  # noqa: BLE001
            self._api_status_label.setText(f"❌ Lỗi: {exc}")

    def _on_calibrate(self) -> None:
        self.calibration_requested.emit()
        QMessageBox.information(
            self,
            "Hiệu chỉnh phối cảnh",
            "Hãy chỉ 4 góc của bảng theo thứ tự:\n"
            "Trên-trái → Trên-phải → Dưới-phải → Dưới-trái\n\n"
            "Cửa sổ hiệu chỉnh sẽ mở sau khi đóng hộp thoại này.",
        )

    def _on_save(self) -> None:
        # Save API key and model to .env
        new_key = self._api_key_edit.text().strip()
        new_model = self._model_edit.text().strip() or "gemini-2.5-flash"
        if new_key:
            _write_env_key("GEMINI_API_KEY", new_key)
            _write_env_key("GEMINI_MODEL", new_model)
            self.api_key_changed.emit(new_key)

        # Camera index
        self.camera_index_changed.emit(self._camera_index_spin.value())

        self.accept()

    # ── Getters for main window ───────────────────────────────────────────────

    @property
    def ink_color(self) -> str:
        return self._ink_color

    @property
    def min_stroke_width(self) -> float:
        return self._min_width_spin.value()

    @property
    def max_stroke_width(self) -> float:
        return self._max_width_spin.value()

    @property
    def rdp_epsilon(self) -> float:
        return self._rdp_spin.value()

    @property
    def one_euro_beta(self) -> float:
        return self._beta_spin.value()

    @property
    def shape_snapping_enabled(self) -> bool:
        return self._snap_check.isChecked()

    @property
    def clahe_enabled(self) -> bool:
        return self._clahe_check.isChecked()
