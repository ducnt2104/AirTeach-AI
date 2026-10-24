"""
AirTeach AI - Settings Dialog (Modern Teacher Workstation)
=========================================================
Comprehensive per-teacher settings dialog supporting:
  1. AI & API Settings (Per-Teacher Keyring Storage, Connection Test, Model Selection, Feature Toggles)
  2. Camera & Vision Diagnostics (Camera index, Adaptive Lighting mode, Preprocessing)
  3. Gesture & Drawing (Air Pen vs Precision Pinch mode, Stroke Beautifier, Shape Snapping)
  4. Privacy & Workspace (Profile info, Data privacy policy, Key redaction)

Zero UI Blocking: Connection tests run on background worker threads.
Zero Plain-Text Secrets: All credentials are saved in OS Keyring with per-teacher namespace.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.services.credential_service import TeacherAISettings, TeacherCredentialService

logger = logging.getLogger(__name__)


class _ApiTestWorkerSignals(QObject):
    finished = Signal(bool, str)


class _ApiTestRunnable(QRunnable):
    """Asynchronous worker to test API keys without blocking Qt UI thread."""

    def __init__(self, api_key: str, model_name: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.signals = _ApiTestWorkerSignals()

    @Slot()
    def run(self) -> None:
        success, message = TeacherCredentialService.test_gemini_connection(
            self.api_key, self.model_name
        )
        self.signals.finished.emit(success, message)


class SettingsDialog(QDialog):
    """
    Workstation Settings dialog with per-teacher account isolation.

    Signals:
        api_key_changed(str): emitted when teacher updates their API key.
        settings_updated(): emitted when any teacher setting changes.
        camera_index_changed(int): emitted when camera index changes.
        calibration_requested(): emitted when perspective/gesture calibration is requested.
    """

    api_key_changed = Signal(str)
    settings_updated = Signal()
    camera_index_changed = Signal(int)
    calibration_requested = Signal()

    def __init__(
        self,
        teacher_id: str = "",
        teacher_name: str = "Giáo viên",
        credential_service: Optional[TeacherCredentialService] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        if isinstance(teacher_id, QWidget):
            parent = teacher_id
            teacher_id = ""
        super().__init__(parent)
        self.teacher_id = teacher_id
        self.teacher_name = teacher_name
        self.cred_service = credential_service or TeacherCredentialService()

        # Load teacher's existing settings & credentials
        self._current_settings: TeacherAISettings = self.cred_service.get_settings(self.teacher_id)
        self._current_api_key = self.cred_service.get_api_key(self.teacher_id)

        self.setWindowTitle(f"Cài đặt AirTeach AI — {self.teacher_name}")
        self.setMinimumSize(680, 520)
        self.setModal(True)

        self._ink_color = "#202124"
        self._is_editing_key = False
        self._thread_pool = QThreadPool.globalInstance()

        self._build_ui()
        self._populate_current_settings()

    def _build_ui(self) -> None:
        root_v = QVBoxLayout(self)
        root_v.setContentsMargins(16, 16, 16, 16)
        root_v.setSpacing(12)

        # Header with Teacher Profile Badge
        header_h = QHBoxLayout()
        header_title = QLabel("⚙️ Cài đặt hệ thống & Trợ giảng AI")
        header_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        header_h.addWidget(header_title)

        header_h.addStretch(1)
        profile_badge = QLabel(f"👤 Hồ sơ: {self.teacher_name}")
        profile_badge.setStyleSheet(
            "background: #e8f0fe; color: #1a73e8; padding: 4px 10px; border-radius: 12px; font-weight: 600;"
        )
        header_h.addWidget(profile_badge)
        root_v.addLayout(header_h)

        # Main Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d0d7de;
                background: #ffffff;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #f6f8fa;
                border: 1px solid #d0d7de;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
                color: #1a73e8;
                font-weight: 600;
            }
        """)

        self.tabs.addTab(self._build_ai_tab(), "🤖 AI & API Settings")
        self.tabs.addTab(self._build_gesture_tab(), "✋ Cử chỉ & Bút vẽ")
        self.tabs.addTab(self._build_camera_tab(), "📷 Camera & Thị giác")
        self.tabs.addTab(self._build_privacy_tab(), "🔒 Quyền riêng tư & Bảo mật")

        root_v.addWidget(self.tabs, stretch=1)

        # Dialog Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.button(QDialogButtonBox.StandardButton.Save).setText("Lưu cài đặt")
        btn_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Hủy")
        btn_box.accepted.connect(self._on_save_all)
        btn_box.rejected.connect(self.reject)
        root_v.addWidget(btn_box)

    # ── Tab 1: AI & API Settings ──────────────────────────────────────────────

    def _build_ai_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Section: API Key configuration
        key_group = QGroupBox("Khóa Google Gemini API (Theo tài khoản cá nhân)")
        key_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        key_form = QFormLayout(key_group)
        key_form.setSpacing(10)

        # Provider
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Google Gemini (Khuyên dùng)", "Local Offline Only"])
        key_form.addRow("Nhà cung cấp AI:", self.provider_combo)

        # API Key input row with Show/Hide, Edit, and Delete
        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("AIzaSy… Nhập khóa từ Google AI Studio")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_row.addWidget(self.api_key_edit, stretch=1)

        self.btn_toggle_key = QPushButton("👁 Hiện")
        self.btn_toggle_key.setCheckable(True)
        self.btn_toggle_key.setFixedWidth(65)
        self.btn_toggle_key.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self.btn_toggle_key)

        self.btn_delete_key = QPushButton("🗑 Xóa key")
        self.btn_delete_key.setStyleSheet("color: #d93025;")
        self.btn_delete_key.clicked.connect(self._on_delete_key)
        key_row.addWidget(self.btn_delete_key)
        key_form.addRow("Khóa API cá nhân:", key_row)

        # Model selection
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ])
        self.model_combo.setEditable(True)
        model_row.addWidget(self.model_combo, stretch=1)

        # Test Connection button
        self.btn_test_conn = QPushButton("🔗 Kiểm tra kết nối")
        self.btn_test_conn.setStyleSheet("background: #f1f3f4; font-weight: 500;")
        self.btn_test_conn.clicked.connect(self._on_test_connection)
        model_row.addWidget(self.btn_test_conn)
        key_form.addRow("Mô hình AI:", model_row)

        # Connection Status Badge
        self.status_badge = QLabel("Chưa kiểm tra kết nối")
        self.status_badge.setStyleSheet(
            "padding: 4px 8px; border-radius: 4px; background: #f1f3f4; color: #5f6368;"
        )
        key_form.addRow("Trạng thái:", self.status_badge)

        layout.addWidget(key_group)

        # Section: AI Feature Toggles
        feat_group = QGroupBox("Các tính năng AI trợ giảng được kích hoạt")
        feat_layout = QVBoxLayout(feat_group)
        feat_layout.setSpacing(6)

        self.chk_assistant = QCheckBox("Trợ lý đàm thoại & gợi ý bài giảng trên lớp (Lesson Assistant)")
        self.chk_doc_summary = QCheckBox("Tóm tắt tài liệu & slide bài giảng (Document Summarizer)")
        self.chk_quiz = QCheckBox("Tạo câu hỏi trắc nghiệm & đề ôn tập đa cấp độ (Quiz Generator)")
        self.chk_explainer = QCheckBox("Giải thích trực quan các khái niệm STEM khó (Concept Explainer)")
        self.chk_lesson_plan = QCheckBox("Gợi ý kế hoạch bài dạy chuẩn 45 phút (Lesson Plan Assistant)")
        self.chk_smart_ink = QCheckBox("Hỗ trợ diễn giải công thức toán & văn bản OCR (Smart Ink Assistant)")

        for chk in [
            self.chk_assistant,
            self.chk_doc_summary,
            self.chk_quiz,
            self.chk_explainer,
            self.chk_lesson_plan,
            self.chk_smart_ink,
        ]:
            chk.setChecked(True)
            feat_layout.addWidget(chk)

        layout.addWidget(feat_group)
        layout.addStretch(1)
        return w

    # ── Tab 2: Gesture & Drawing ──────────────────────────────────────────────

    def _build_gesture_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Gesture Drawing Mode Selection (Part V: Mode A vs Mode B)
        mode_group = QGroupBox("Chế độ kích hoạt nét vẽ (DRAW Mode)")
        mode_layout = QVBoxLayout(mode_group)

        mode_row = QHBoxLayout()
        mode_label = QLabel("Phương thức vẽ:")
        self.draw_mode_combo = QComboBox()
        self.draw_mode_combo.addItem("Chế độ A — Air Pen (Giơ ngón trỏ tự nhiên để viết)", "index_up")
        self.draw_mode_combo.addItem("Chế độ B — Precision Pinch (Chụm ngón trỏ và ngón cái để viết)", "precision_pinch")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.draw_mode_combo, stretch=1)
        mode_layout.addLayout(mode_row)

        self.mode_desc_label = QLabel(
            "• Chế độ Air Pen: Phù hợp giáo viên mới làm quen. Chỉ cần giơ ngón trỏ lên và vẽ thoải mái trong không khí.\n"
            "• Chế độ Precision Pinch: Dành cho viết chữ và vẽ nét kỹ thuật độ chính xác cao. Chụm ngón tay để hạ bút."
        )
        self.mode_desc_label.setStyleSheet("color: #5f6368; font-size: 11px; margin-top: 4px;")
        mode_layout.addWidget(self.mode_desc_label)

        # Pinch Sensitivity (normalized distance)
        pinch_row = QHBoxLayout()
        pinch_row.addWidget(QLabel("Độ nhạy Pinch (khoảng cách kích hoạt):"))
        self.pinch_spin = QDoubleSpinBox()
        self.pinch_spin.setRange(0.15, 0.45)
        self.pinch_spin.setSingleStep(0.02)
        self.pinch_spin.setValue(0.28)
        pinch_row.addWidget(self.pinch_spin)
        mode_layout.addLayout(pinch_row)

        layout.addWidget(mode_group)

        # Multi-Hand & Air Control Settings
        multi_hand_group = QGroupBox("Điều khiển Đa bàn tay & Tương tác Không gian (Multi-Hand Air Control)")
        mh_form = QFormLayout(multi_hand_group)
        mh_form.setSpacing(10)

        # Coordination Mode
        self.multi_hand_mode_combo = QComboBox()
        self.multi_hand_mode_combo.addItem("Trợ lý giảng dạy (Tay thuận vẽ, tay phụ mở Menu & Đổi công cụ)", "draw_and_tool")
        self.multi_hand_mode_combo.addItem("Vẽ hai tay đồng thời (Hai tay vẽ nét độc lập)", "dual_draw")
        self.multi_hand_mode_combo.addItem("Vẽ và Tẩy nhanh (Tay thuận vẽ, tay phụ làm Tẩy xóa)", "draw_and_erase")
        self.multi_hand_mode_combo.addItem("Độc lập (Tự động theo vai trò hai tay)", "independent")
        mh_form.addRow("Chế độ phối hợp 2 tay:", self.multi_hand_mode_combo)

        # Dominant Hand
        self.dominant_hand_combo = QComboBox()
        self.dominant_hand_combo.addItem("Tay phải (Right-handed)", "Right")
        self.dominant_hand_combo.addItem("Tay trái (Left-handed)", "Left")
        mh_form.addRow("Tay thuận giảng dạy:", self.dominant_hand_combo)

        # Air Click Trigger
        self.air_click_combo = QComboBox()
        self.air_click_combo.addItem("Pinch Click (Chụm ngón trỏ và ngón cái)", "pinch")
        self.air_click_combo.addItem("Air Tap (Nhấn tới trong không gian)", "air_tap")
        self.air_click_combo.addItem("Dwell Selection (Giữ con trỏ ổn định 600ms)", "dwell")
        self.air_click_combo.addItem("Hybrid (Kết hợp Tap, Pinch & Dwell)", "hybrid")
        mh_form.addRow("Cử chỉ kích hoạt Menu / Nhấn:", self.air_click_combo)

        # Eraser default radius
        eraser_row = QHBoxLayout()
        self.eraser_radius_spin = QSpinBox()
        self.eraser_radius_spin.setRange(15, 80)
        self.eraser_radius_spin.setValue(32)
        eraser_row.addWidget(self.eraser_radius_spin)
        eraser_row.addWidget(QLabel("px (bán kính vùng tẩy)"))
        eraser_row.addStretch(1)
        mh_form.addRow("Bán kính tẩy xóa:", eraser_row)

        # Floating Menu Layout
        self.menu_layout_combo = QComboBox()
        self.menu_layout_combo.addItem("Radial Wheel (Bánh xe xoay tròn 360° - Tối ưu cử chỉ)", "radial_wheel")
        self.menu_layout_combo.addItem("Floating Panel (Bảng nổi đứng - Gọn gàng)", "floating_panel")
        mh_form.addRow("Giao diện Menu nổi:", self.menu_layout_combo)

        layout.addWidget(multi_hand_group)

        # Stroke Beautifier Settings
        stroke_group = QGroupBox("Bộ lọc làm mượt nét 3 tầng (Stroke Beautifier)")
        stroke_form = QFormLayout(stroke_group)
        stroke_form.setSpacing(10)

        # Color picker
        color_row = QHBoxLayout()
        self.ink_color_preview = QPushButton()
        self.ink_color_preview.setFixedSize(60, 24)
        self._update_color_btn()
        self.ink_color_preview.clicked.connect(self._on_pick_color)
        color_row.addWidget(self.ink_color_preview)
        color_row.addWidget(QLabel("Nhấn vào ô để đổi màu mực mặc định"))
        color_row.addStretch(1)
        stroke_form.addRow("Màu nét mực mặc định:", color_row)

        # Pen width limits
        width_row = QHBoxLayout()
        self.min_width_spin = QDoubleSpinBox()
        self.min_width_spin.setRange(0.5, 5.0)
        self.min_width_spin.setValue(1.5)
        width_row.addWidget(QLabel("Tối thiểu:"))
        width_row.addWidget(self.min_width_spin)

        self.max_width_spin = QDoubleSpinBox()
        self.max_width_spin.setRange(1.0, 20.0)
        self.max_width_spin.setValue(6.0)
        width_row.addWidget(QLabel("Tối đa:"))
        width_row.addWidget(self.max_width_spin)
        width_row.addStretch(1)
        stroke_form.addRow("Độ dày nét bút (px):", width_row)

        # Smoothing preset combo
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Tiêu chuẩn cân bằng (Khuyên dùng)", "balanced")
        self.preset_combo.addItem("Siêu mượt — Giảm rung tay (Thư pháp / Công thức)", "smooth")
        self.preset_combo.addItem("Siêu nhạy — Phản xạ tức thì (Phác thảo nhanh)", "instant")
        self.preset_combo.addItem("Tùy chỉnh thông số thủ công", "custom")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        stroke_form.addRow("Chế độ làm mượt nét vẽ:", self.preset_combo)

        # 1€ Filter beta
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.01, 2.0)
        self.beta_spin.setSingleStep(0.05)
        self.beta_spin.setValue(0.35)
        self.beta_spin.setToolTip("Tham số β của bộ lọc 1€ – tăng β để giảm độ trễ khi vung tay nhanh.")
        self.beta_spin.valueChanged.connect(self._on_manual_spin_changed)
        stroke_form.addRow("Bộ lọc 1€ Filter (Beta):", self.beta_spin)

        # RDP epsilon
        self.rdp_spin = QDoubleSpinBox()
        self.rdp_spin.setRange(0.5, 10.0)
        self.rdp_spin.setSingleStep(0.5)
        self.rdp_spin.setValue(2.5)
        self.rdp_spin.valueChanged.connect(self._on_manual_spin_changed)
        stroke_form.addRow("Ngưỡng làm mịn RDP (ε):", self.rdp_spin)

        # Shape snapping
        self.chk_shape_snap = QCheckBox("Tự động nắn hình học chuẩn (Shape Snapping)")
        self.chk_shape_snap.setChecked(True)
        stroke_form.addRow("", self.chk_shape_snap)

        layout.addWidget(stroke_group)

        # Grid settings
        grid_group = QGroupBox("Cấu hình lưới bảng vẽ & Hệ trục Oxy")
        grid_form = QFormLayout(grid_group)
        grid_form.setSpacing(10)

        self.grid_mode_combo = QComboBox()
        self.grid_mode_combo.addItem("Không hiển thị (None)", "none")
        self.grid_mode_combo.addItem("Lưới chấm (Dots)", "dots")
        self.grid_mode_combo.addItem("Lưới ô vuông (Grid)", "grid")
        self.grid_mode_combo.addItem("Hệ trục tọa độ Oxy (Cartesian)", "cartesian")
        grid_form.addRow("Chế độ hiển thị lưới:", self.grid_mode_combo)

        self.grid_spacing_spin = QSpinBox()
        self.grid_spacing_spin.setRange(20, 150)
        self.grid_spacing_spin.setSingleStep(10)
        self.grid_spacing_spin.setValue(40)
        self.grid_spacing_spin.setSuffix(" px")
        grid_form.addRow("Khoảng cách ô lưới:", self.grid_spacing_spin)

        layout.addWidget(grid_group)

        # Audio Feedback Settings
        audio_group = QGroupBox("Âm thanh & Phản hồi Tương tác (Acoustic Gesture Feedback)")
        audio_form = QFormLayout(audio_group)
        audio_form.setSpacing(10)

        self.audio_feedback_check = QCheckBox("Bật âm thanh phản hồi cử chỉ (Click, Snap, Tẩy, Chuyển trang)")
        from src.utils.audio_feedback import AudioFeedbackManager
        audio_mgr = AudioFeedbackManager.instance()
        self.audio_feedback_check.setChecked(audio_mgr.enabled)
        audio_form.addRow("", self.audio_feedback_check)

        audio_vol_row = QHBoxLayout()
        self.audio_volume_spin = QSpinBox()
        self.audio_volume_spin.setRange(0, 100)
        self.audio_volume_spin.setSingleStep(5)
        self.audio_volume_spin.setValue(int(round(audio_mgr.volume * 100)))
        self.audio_volume_spin.setSuffix(" %")
        audio_vol_row.addWidget(self.audio_volume_spin)
        audio_vol_row.addStretch(1)
        audio_form.addRow("Âm lượng phản hồi:", audio_vol_row)

        test_row = QHBoxLayout()
        self.audio_test_type_combo = QComboBox()
        self.audio_test_type_combo.addItems([
            "✨ Nắn hình học (Snap)",
            "🖱️ Chạm không chạm (Air Click)",
            "📄 Lật trang (Page Turn)",
            "🧹 Xóa nét (Erase)",
            "🔔 Chuông đếm giờ (Timer Chime)",
            "↩️ Hoàn tác (Undo)",
        ])
        test_row.addWidget(self.audio_test_type_combo, stretch=2)

        self.btn_test_audio = QPushButton("🔊 Thử âm thanh")
        self.btn_test_audio.clicked.connect(self._on_test_audio)
        test_row.addWidget(self.btn_test_audio, stretch=1)
        audio_form.addRow("Kiểm tra âm thanh:", test_row)

        layout.addWidget(audio_group)

        # Laser Pointer & Glow Trail Customization
        laser_group = QGroupBox("Chỉ thị Laser & Vệt Sáng Tương Tác (Laser Pointer & Glow Trail)")
        laser_form = QFormLayout(laser_group)
        laser_form.setSpacing(10)

        self.laser_color_combo = QComboBox()
        self.laser_color_combo.addItem("🔴 Đỏ Ruby (#ef4444)", "#ef4444")
        self.laser_color_combo.addItem("🟢 Xanh Lục Bảo (#10b981)", "#10b981")
        self.laser_color_combo.addItem("🔵 Xanh Cyan Dạ Quang (#06b6d4)", "#06b6d4")
        self.laser_color_combo.addItem("🟡 Vàng Hổ Phách (#f59e0b)", "#f59e0b")
        self.laser_color_combo.addItem("🟣 Tím Dạ Quang (#8b5cf6)", "#8b5cf6")
        laser_form.addRow("Màu tia laser:", self.laser_color_combo)

        self.laser_size_combo = QComboBox()
        self.laser_size_combo.addItem("Nhỏ (8 px — Viết chi tiết)", 8.0)
        self.laser_size_combo.addItem("Tiêu chuẩn (12 px — Chuẩn bài giảng)", 12.0)
        self.laser_size_combo.addItem("Lớn (18 px — Hội trường & Máy chiếu)", 18.0)
        self.laser_size_combo.setCurrentIndex(1)
        laser_form.addRow("Kích thước đầu trỏ:", self.laser_size_combo)

        self.laser_trail_check = QCheckBox("Bật hiệu ứng vệt sáng pháo hoa dạ quang (Laser Sparkler Trail)")
        self.laser_trail_check.setChecked(True)
        laser_form.addRow("", self.laser_trail_check)

        layout.addWidget(laser_group)
        layout.addStretch(1)
        return w

    # ── Tab 3: Camera & Vision Diagnostics ────────────────────────────────────

    def _build_camera_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        cam_group = QGroupBox("Cấu hình thiết bị Webcam & Tiền xử lý ánh sáng")
        cam_form = QFormLayout(cam_group)
        cam_form.setSpacing(10)

        # Camera Index
        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 9)
        self.cam_index_spin.setValue(0)
        cam_form.addRow("Chỉ số cổng Webcam (Index):", self.cam_index_spin)

        # Camera Lighting Mode (Part IV)
        self.lighting_mode_combo = QComboBox()
        self.lighting_mode_combo.addItem("Tự động thích ứng (Adaptive Lighting - Khuyên dùng)", "adaptive")
        self.lighting_mode_combo.addItem("Chế độ phòng thiếu sáng (Low Light Enhancement)", "low_light")
        self.lighting_mode_combo.addItem("Chế độ ngược sáng & cân bằng (Balanced CLAHE)", "balanced")
        self.lighting_mode_combo.addItem("Ưu tiên tốc độ khung hình (Motion Priority)", "motion_priority")
        cam_form.addRow("Chế độ xử lý môi trường:", self.lighting_mode_combo)

        # CLAHE enable
        self.chk_clahe = QCheckBox("Kích hoạt cân bằng tương phản thích ứng (CLAHE)")
        self.chk_clahe.setChecked(True)
        cam_form.addRow("", self.chk_clahe)

        # Mirror Preview
        self.chk_mirror = QCheckBox("Lật gương hình ảnh webcam (Mirror preview)")
        self.chk_mirror.setChecked(True)
        cam_form.addRow("", self.chk_mirror)

        layout.addWidget(cam_group)

        # Calibration Actions
        calib_group = QGroupBox("Hiệu chỉnh tọa độ không gian")
        calib_layout = QHBoxLayout(calib_group)
        calib_btn = QPushButton("📐 Hiệu chỉnh phối cảnh 4 góc (Perspective)")
        calib_btn.clicked.connect(self._on_calibrate_perspective)
        calib_layout.addWidget(calib_btn)
        layout.addWidget(calib_group)

        layout.addStretch(1)
        return w

    # ── Tab 4: Privacy & Security ─────────────────────────────────────────────

    def _build_privacy_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        privacy_box = QGroupBox("Chính sách bảo mật & Quyền riêng tư (Zero Cloud Leakage)")
        privacy_layout = QVBoxLayout(privacy_box)
        privacy_layout.setSpacing(8)

        info_text = QLabel(
            "AirTeach AI cam kết bảo vệ dữ liệu giảng dạy và quyền riêng tư của giáo viên:\n\n"
            "1. <b>Bảo mật khóa API:</b> Khóa API của từng giáo viên được lưu trữ độc lập bằng "
            "trình quản lý bảo mật của hệ điều hành (Windows Credential Manager). Khóa không bao giờ "
            "lưu plain-text trong database hoặc tệp bài giảng .airteach.\n\n"
            "2. <b>Không gửi hình ảnh webcam lên đám mây:</b> Toàn bộ thuật toán theo dõi bàn tay (MediaPipe) "
            "và nhận diện khuôn mặt (YuNet/SFace) chạy 100% cục bộ trên máy tính của bạn.\n\n"
            "3. <b>Quyền kiểm soát AI:</b> Dữ liệu bài giảng chỉ được gửi tới Gemini API khi bạn "
            "chủ động nhấn nút yêu cầu trợ lý AI. Hệ thống chỉ gửi tối thiểu vùng nội dung được chọn."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3c4043; line-height: 1.4;")
        privacy_layout.addWidget(info_text)

        layout.addWidget(privacy_box)
        layout.addStretch(1)
        return w

    # ── Population & Actions ──────────────────────────────────────────────────

    def _populate_current_settings(self) -> None:
        """Populate UI widgets with teacher's saved settings."""
        # Key masking
        if self._current_api_key:
            masked = TeacherCredentialService.mask_api_key(self._current_api_key)
            self.api_key_edit.setText(masked)
            self.api_key_edit.setEnabled(False)
            self.btn_toggle_key.setText("Thay đổi")
            self._is_editing_key = False
            self.status_badge.setText("✅ Đã cấu hình khóa bảo mật (Windows Keyring)")
            self.status_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 4px; background: #e6f4ea; color: #137333; font-weight: 500;"
            )
        else:
            self.api_key_edit.setText("")
            self.api_key_edit.setEnabled(True)
            self.btn_toggle_key.setText("👁 Hiện")
            self._is_editing_key = True
            self.status_badge.setText("⚠️ Chưa cấu hình khóa API (Đang chạy chế độ ngoại tuyến)")
            self.status_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 4px; background: #fef7e0; color: #b06000; font-weight: 500;"
            )

        # Model
        idx = self.model_combo.findText(self._current_settings.model_name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setEditText(self._current_settings.model_name)

        # Feature checkboxes
        self.chk_assistant.setChecked(self._current_settings.feature_lesson_assistant)
        self.chk_doc_summary.setChecked(self._current_settings.feature_doc_summary)
        self.chk_quiz.setChecked(self._current_settings.feature_quiz_gen)
        self.chk_explainer.setChecked(self._current_settings.feature_concept_explainer)
        self.chk_lesson_plan.setChecked(self._current_settings.feature_lesson_plan)
        self.chk_smart_ink.setChecked(self._current_settings.feature_smart_ink)

        # Gesture mode
        draw_idx = self.draw_mode_combo.findData(self._current_settings.draw_mode)
        if draw_idx >= 0:
            self.draw_mode_combo.setCurrentIndex(draw_idx)

        # Pinch threshold
        self.pinch_spin.setValue(self._current_settings.pinch_threshold)

        # Lighting mode
        light_idx = self.lighting_mode_combo.findData(self._current_settings.camera_lighting_mode)
        if light_idx >= 0:
            self.lighting_mode_combo.setCurrentIndex(light_idx)

        # Multi-hand
        m_mode = getattr(self._current_settings, "multi_hand_mode", "draw_and_tool")
        idx = self.multi_hand_mode_combo.findData(m_mode)
        if idx >= 0:
            self.multi_hand_mode_combo.setCurrentIndex(idx)

        d_hand = getattr(self._current_settings, "dominant_hand", "Right")
        idx = self.dominant_hand_combo.findData(d_hand)
        if idx >= 0:
            self.dominant_hand_combo.setCurrentIndex(idx)

        c_type = getattr(self._current_settings, "air_click_type", "pinch")
        idx = self.air_click_combo.findData(c_type)
        if idx >= 0:
            self.air_click_combo.setCurrentIndex(idx)

        e_rad = getattr(self._current_settings, "eraser_radius", 32.0)
        self.eraser_radius_spin.setValue(int(e_rad))

        l_mode = getattr(self._current_settings, "menu_layout_mode", "radial_wheel")
        idx = self.menu_layout_combo.findData(l_mode)
        if idx >= 0:
            self.menu_layout_combo.setCurrentIndex(idx)

    def _toggle_key_visibility(self, checked: bool) -> None:
        """Handle Toggle / Change Key action."""
        if not self._is_editing_key:
            # Switch into edit mode so teacher can type new key
            self.api_key_edit.setEnabled(True)
            self.api_key_edit.clear()
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self.api_key_edit.setFocus()
            self.btn_toggle_key.setText("👁 Ẩn")
            self._is_editing_key = True
        else:
            if checked:
                self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
                self.btn_toggle_key.setText("👁 Ẩn")
            else:
                self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
                self.btn_toggle_key.setText("👁 Hiện")

    def _on_delete_key(self) -> None:
        """Confirm and delete teacher's API key."""
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa khóa API",
            f"Bạn có chắc chắn muốn xóa khóa API của tài khoản '{self.teacher_name}'?\n"
            "Các tính năng trợ lý AI trực tuyến sẽ chuyển sang chế độ ngoại tuyến.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.cred_service.delete_api_key(self.teacher_id)
            self._current_api_key = ""
            self.api_key_edit.setEnabled(True)
            self.api_key_edit.clear()
            self._is_editing_key = True
            self.status_badge.setText("Đã xóa khóa thành công.")
            self.status_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 4px; background: #fce8e6; color: #c5221f;"
            )
            self.api_key_changed.emit("")

    def _on_test_connection(self) -> None:
        """Test API key asynchronously on background QThreadPool."""
        # If not editing, test current saved key
        key_to_test = self.api_key_edit.text().strip()
        if not self._is_editing_key:
            key_to_test = self._current_api_key

        if not key_to_test or "•" in key_to_test:
            key_to_test = self._current_api_key

        if not key_to_test:
            self.status_badge.setText("❌ Vui lòng nhập khóa API trước khi kiểm tra.")
            self.status_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 4px; background: #fce8e6; color: #c5221f;"
            )
            return

        model = self.model_combo.currentText().strip() or "gemini-2.5-flash"
        self.status_badge.setText("⏳ Đang kiểm tra kết nối tới Google AI Studio…")
        self.status_badge.setStyleSheet(
            "padding: 4px 8px; border-radius: 4px; background: #e8f0fe; color: #1a73e8;"
        )
        self.btn_test_conn.setEnabled(False)

        worker = _ApiTestRunnable(key_to_test, model)
        worker.signals.finished.connect(self._on_test_connection_finished)
        self._thread_pool.start(worker)

    def _on_test_connection_finished(self, success: bool, message: str) -> None:
        self.btn_test_conn.setEnabled(True)
        if success:
            self.status_badge.setText(f"✅ {message}")
            self.status_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 4px; background: #e6f4ea; color: #137333; font-weight: 500;"
            )
        else:
            self.status_badge.setText(f"❌ {message}")
            self.status_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 4px; background: #fce8e6; color: #c5221f;"
            )

    def _on_preset_changed(self, idx: int) -> None:
        data = self.preset_combo.currentData()
        self.beta_spin.blockSignals(True)
        self.rdp_spin.blockSignals(True)
        if data == "balanced":
            self.beta_spin.setValue(0.35)
            self.rdp_spin.setValue(2.5)
        elif data == "smooth":
            self.beta_spin.setValue(0.15)
            self.rdp_spin.setValue(3.5)
        elif data == "instant":
            self.beta_spin.setValue(0.70)
            self.rdp_spin.setValue(1.0)
        self.beta_spin.blockSignals(False)
        self.rdp_spin.blockSignals(False)

    def _on_manual_spin_changed(self) -> None:
        b = self.beta_spin.value()
        r = self.rdp_spin.value()
        if abs(b - 0.35) < 1e-3 and abs(r - 2.5) < 1e-3:
            target_idx = 0
        elif abs(b - 0.15) < 1e-3 and abs(r - 3.5) < 1e-3:
            target_idx = 1
        elif abs(b - 0.70) < 1e-3 and abs(r - 1.0) < 1e-3:
            target_idx = 2
        else:
            target_idx = 3
        if self.preset_combo.currentIndex() != target_idx:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(target_idx)
            self.preset_combo.blockSignals(False)

    def _on_pick_color(self) -> None:
        col = QColorDialog.getColor(QColor(self._ink_color), self, "Chọn màu nét mực")
        if col.isValid():
            self._ink_color = col.name()
            self._update_color_btn()

    def _update_color_btn(self) -> None:
        self.ink_color_preview.setStyleSheet(
            f"background:{self._ink_color}; border:1px solid #ccc; border-radius:4px;"
        )

    def _on_test_audio(self) -> None:
        sound_type = "snap"
        if hasattr(self, "audio_test_type_combo"):
            txt = self.audio_test_type_combo.currentText()
            if "Snap" in txt:
                sound_type = "snap"
            elif "Click" in txt:
                sound_type = "click"
            elif "Page" in txt or "Lật" in txt:
                sound_type = "page"
            elif "Erase" in txt or "Xóa" in txt:
                sound_type = "erase"
            elif "Timer" in txt or "Chuông" in txt:
                sound_type = "timer"
            elif "Undo" in txt or "Hoàn tác" in txt:
                sound_type = "undo"
        self.play_test_sound(sound_type)

    def play_test_sound(self, sound_type: str = "snap") -> None:
        """Play a preview acoustic gesture feedback sound at currently configured test volume."""
        from src.utils.audio_feedback import AudioFeedbackManager
        mgr = AudioFeedbackManager.instance()
        old_enabled = mgr.enabled
        try:
            mgr.volume = self.audio_volume_spin.value() / 100.0
            mgr.enabled = True  # Always allow test preview even if main toggle is off
            st = sound_type.lower()
            if "click" in st:
                mgr.play_click()
            elif "snap" in st:
                mgr.play_snap()
            elif "page" in st:
                mgr.play_page_turn()
            elif "erase" in st:
                mgr.play_erase()
            elif "timer" in st or "chime" in st:
                mgr.play_timer_chime()
            elif "undo" in st:
                mgr.play_undo()
            else:
                mgr.play_snap()
        finally:
            # Restore dialog's pending state
            mgr.enabled = self.audio_feedback_check.isChecked()

    def _on_calibrate_perspective(self) -> None:
        self.calibration_requested.emit()
        QMessageBox.information(
            self,
            "Hiệu chỉnh phối cảnh",
            "Cửa sổ hiệu chỉnh 4 góc sẽ mở ngay sau khi đóng hộp thoại này.",
        )

    def _on_save_all(self) -> None:
        """Persist all teacher settings and credentials."""
        # 1. Update API Key if modified
        if self._is_editing_key:
            new_key = self.api_key_edit.text().strip()
            if new_key and not ("•" in new_key):
                self.cred_service.set_api_key(self.teacher_id, new_key)
                self._current_api_key = new_key
                self.api_key_changed.emit(new_key)

        # 2. Update AI Settings
        model = self.model_combo.currentText().strip() or "gemini-2.5-flash"
        draw_mode = self.draw_mode_combo.currentData() or "index_up"
        lighting_mode = self.lighting_mode_combo.currentData() or "adaptive"
        pinch_val = self.pinch_spin.value()
        m_mode = self.multi_hand_mode_combo.currentData() or "draw_and_tool"
        d_hand = self.dominant_hand_combo.currentData() or "Right"
        c_type = self.air_click_combo.currentData() or "pinch"
        e_rad = float(self.eraser_radius_spin.value())
        l_mode = self.menu_layout_combo.currentData() or "radial_wheel"

        settings = TeacherAISettings(
            teacher_id=self.teacher_id,
            provider="gemini",
            model_name=model,
            feature_lesson_assistant=self.chk_assistant.isChecked(),
            feature_doc_summary=self.chk_doc_summary.isChecked(),
            feature_quiz_gen=self.chk_quiz.isChecked(),
            feature_concept_explainer=self.chk_explainer.isChecked(),
            feature_lesson_plan=self.chk_lesson_plan.isChecked(),
            feature_smart_ink=self.chk_smart_ink.isChecked(),
            draw_mode=draw_mode,
            camera_lighting_mode=lighting_mode,
            pinch_threshold=pinch_val,
            multi_hand_mode=m_mode,
            dominant_hand=d_hand,
            air_click_type=c_type,
            eraser_radius=e_rad,
            menu_layout_mode=l_mode,
        )
        self.cred_service.save_settings(settings)
        self._current_settings = settings

        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            mgr = AudioFeedbackManager.instance()
            mgr.enabled = self.audio_feedback_check.isChecked()
            mgr.volume = self.audio_volume_spin.value() / 100.0
        except Exception:
            pass

        # 3. Emit signals
        self.camera_index_changed.emit(self.cam_index_spin.value())
        self.settings_updated.emit()
        self.accept()

    # ── Public Getters ────────────────────────────────────────────────────────

    @property
    def ink_color(self) -> str:
        return self._ink_color

    @property
    def min_stroke_width(self) -> float:
        return self.min_width_spin.value()

    @property
    def max_stroke_width(self) -> float:
        return self.max_width_spin.value()

    @property
    def smoothing_preset(self) -> str:
        return self.preset_combo.currentData() or "balanced"

    @property
    def rdp_epsilon(self) -> float:
        return self.rdp_spin.value()

    @property
    def one_euro_beta(self) -> float:
        return self.beta_spin.value()

    @property
    def shape_snapping_enabled(self) -> bool:
        return self.chk_shape_snap.isChecked()

    @property
    def clahe_enabled(self) -> bool:
        return self.chk_clahe.isChecked()

    @property
    def mirror_camera(self) -> bool:
        return self.chk_mirror.isChecked()

    @property
    def draw_mode(self) -> str:
        return self.draw_mode_combo.currentData() or "index_up"

    @property
    def camera_lighting_mode(self) -> str:
        return self.lighting_mode_combo.currentData() or "adaptive"

    @property
    def multi_hand_mode(self) -> str:
        return self.multi_hand_mode_combo.currentData() or "draw_and_tool"

    @property
    def dominant_hand(self) -> str:
        return self.dominant_hand_combo.currentData() or "Right"

    @property
    def air_click_type(self) -> str:
        return self.air_click_combo.currentData() or "pinch"

    @property
    def eraser_radius(self) -> float:
        return float(self.eraser_radius_spin.value())

    @property
    def menu_layout_mode(self) -> str:
        return self.menu_layout_combo.currentData() or "radial_wheel"

    @property
    def grid_mode(self) -> str:
        return self.grid_mode_combo.currentData() or "none"

    def set_grid_mode(self, mode: str) -> None:
        idx = self.grid_mode_combo.findData(mode)
        if idx >= 0:
            self.grid_mode_combo.setCurrentIndex(idx)

    @property
    def grid_spacing(self) -> int:
        return self.grid_spacing_spin.value()

    @property
    def audio_feedback_enabled(self) -> bool:
        return self.audio_feedback_check.isChecked()

    @property
    def audio_volume(self) -> float:
        return self.audio_volume_spin.value() / 100.0

    def set_grid_spacing(self, spacing: int) -> None:
        self.grid_spacing_spin.setValue(int(spacing))

    @property
    def laser_color(self) -> str:
        return self.laser_color_combo.currentData() or "#ef4444"

    def set_laser_color(self, color: str) -> None:
        idx = self.laser_color_combo.findData(color)
        if idx >= 0:
            self.laser_color_combo.setCurrentIndex(idx)

    @property
    def laser_radius(self) -> float:
        return float(self.laser_size_combo.currentData() or 12.0)

    def set_laser_radius(self, radius: float) -> None:
        idx = self.laser_size_combo.findData(float(radius))
        if idx >= 0:
            self.laser_size_combo.setCurrentIndex(idx)

    @property
    def laser_trail_enabled(self) -> bool:
        return self.laser_trail_check.isChecked()

    def set_laser_trail_enabled(self, enabled: bool) -> None:
        self.laser_trail_check.setChecked(bool(enabled))
