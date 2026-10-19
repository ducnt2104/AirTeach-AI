"""
AirTeach AI - Teacher Login Window
===================================
Real-time biometric authentication portal:
  - Live webcam stream with OpenCV YuNet / SFace face bounding boxes
  - Automatic facial authentication when cosine similarity >= 0.363
  - Anti-spoofing liveness check (eye/nose relative displacement over 15 frames)
  - Interactive PIN keypad (0–9, Clear, Enter) for low-light environments
  - Teacher Enrollment modal: collects 5 real frames, full name, subject, and PBKDF2-hashed PIN
  - Demo / Guest mode bypass

Zero-mock policy: 100% production code, real camera I/O on background QThread.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.services.face_profile_service import (
    COSINE_THRESHOLD,
    ENROLL_FRAMES,
    LIVENESS_FRAMES,
    LivenessChecker,
    TeacherProfileService,
)

logger = logging.getLogger(__name__)


# ─── Background Camera Worker ─────────────────────────────────────────────────

class _LoginCameraWorker(QObject):
    """Webcam streaming worker running on a background QThread."""

    frame_ready             = Signal(np.ndarray, object)  # (frame_bgr, faces)
    face_authenticated      = Signal(object)              # sqlite3.Row
    enroll_sample_captured  = Signal(int, int)            # (current, total)
    enroll_finished         = Signal(list)                # list of 5 frames
    error_occurred          = Signal(str)

    def __init__(
        self,
        service: TeacherProfileService,
        camera_index: int = 0,
    ) -> None:
        super().__init__()
        self._service = service
        self._camera_index = camera_index
        self._running = True
        self._liveness = LivenessChecker(required_frames=LIVENESS_FRAMES)

        # Enrollment mode flag
        self._is_enrolling: bool = False
        self._enroll_frames: list[np.ndarray] = []

    def set_enroll_mode(self, enrolling: bool) -> None:
        self._is_enrolling = enrolling
        self._enroll_frames.clear()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            self.error_occurred.emit(f"Không thể mở camera index={self._camera_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Contrast enhancement for classrooms
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

            # Detect faces
            faces = self._service._models.detect_faces(enhanced)

            self.frame_ready.emit(enhanced, faces)

            if faces is not None and len(faces) > 0:
                best_face = faces[np.argmax(faces[:, 14])]
                confidence = float(best_face[14])

                if self._is_enrolling:
                    if confidence >= 0.5:
                        self._enroll_frames.append(enhanced.copy())
                        self.enroll_sample_captured.emit(len(self._enroll_frames), ENROLL_FRAMES)
                        if len(self._enroll_frames) >= ENROLL_FRAMES:
                            self.enroll_finished.emit(list(self._enroll_frames))
                            self._is_enrolling = False
                else:
                    # Auto login mode
                    self._liveness.update(best_face)
                    if self._liveness.is_ready():
                        matched_teacher = self._service.authenticate_by_face(enhanced, self._liveness)
                        if matched_teacher is not None:
                            self.face_authenticated.emit(matched_teacher)
                            self._running = False

        cap.release()


# ─── Enrollment Sub-Dialog ────────────────────────────────────────────────────

class TeacherEnrollDialog(QDialog):
    """Dialog collecting teacher details and triggering frame capture."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Đăng ký Giáo viên Mới")
        self.setFixedSize(460, 320)
        self.setModal(True)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        header = QLabel("👤 Đăng ký Hồ sơ Giáo viên Mới")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #1a73e8;")
        layout.addWidget(header)

        # Name
        layout.addWidget(QLabel("Họ và Tên Giáo viên:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("VD: Thầy Nguyễn Văn An")
        self._name_edit.setStyleSheet("padding: 7px; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(self._name_edit)

        # Subject
        layout.addWidget(QLabel("Môn học phụ trách:"))
        self._subject_edit = QLineEdit()
        self._subject_edit.setPlaceholderText("VD: Toán học / Vật lý")
        self._subject_edit.setStyleSheet("padding: 7px; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(self._subject_edit)

        # Backup PIN
        layout.addWidget(QLabel("Mã PIN bảo mật dự phòng (4–6 số):"))
        self._pin_edit = QLineEdit()
        self._pin_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._pin_edit.setMaxLength(6)
        self._pin_edit.setPlaceholderText("Nhập 4–6 chữ số")
        self._pin_edit.setStyleSheet("padding: 7px; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(self._pin_edit)

        # Instructions
        info = QLabel("💡 Sau khi bấm 'Bắt đầu chụp', hãy nhìn thẳng vào camera và quay nhẹ sang hai bên để lấy 5 mẫu khuôn mặt.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #5f6368; font-size: 11px;")
        layout.addWidget(info)

        layout.addStretch()

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        cancel_btn = QPushButton("Hủy")
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("Bắt đầu chụp")
        start_btn.setStyleSheet("background: #1a73e8; color: white; font-weight: bold; padding: 7px 16px; border-radius: 4px;")
        start_btn.clicked.connect(self._validate)

        btn_box.addWidget(cancel_btn)
        btn_box.addWidget(start_btn)
        layout.addLayout(btn_box)

    def _validate(self) -> None:
        if not self._name_edit.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập Họ và Tên giáo viên.")
            return
        pin = self._pin_edit.text().strip()
        if not (pin.isdigit() and 4 <= len(pin) <= 6):
            QMessageBox.warning(self, "Lỗi", "Mã PIN phải từ 4 đến 6 chữ số.")
            return
        self.accept()

    @property
    def full_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def subject(self) -> str:
        return self._subject_edit.text().strip()

    @property
    def pin(self) -> str:
        return self._pin_edit.text().strip()


# ─── Main Login Window ────────────────────────────────────────────────────────

class LoginWindow(QWidget):
    """
    Teacher Authentication Window with Face Recognition and PIN Pad.
    
    Signals:
        login_successful(object): Emitted with sqlite3.Row of authenticated teacher.
        demo_mode_requested(): Emitted when entering demo mode.
    """

    login_successful     = Signal(object)
    demo_mode_requested  = Signal()

    def __init__(
        self,
        service: TeacherProfileService,
        camera_index: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.camera_index = camera_index

        self._pin_buffer: str = ""
        self._cam_thread: Optional[QThread] = None
        self._cam_worker: Optional[_LoginCameraWorker] = None

        self.setWindowTitle("AirTeach AI — Xác thực Giáo viên")
        self.setMinimumSize(960, 600)
        self.setStyleSheet("background-color: #f8f9fa;")

        self._build_ui()
        self._start_camera()
        self.refresh_teacher_dropdown()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(28, 20, 28, 20)
        root_layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        logo_title = QLabel("✋ AirTeach AI")
        logo_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        logo_title.setStyleSheet("color: #1a73e8;")
        header.addWidget(logo_title)

        header.addStretch()

        demo_btn = QPushButton("🚀 Vào chế độ Dạy thử / Demo")
        demo_btn.setStyleSheet(
            "background: #e8f0fe; color: #1a73e8; border: 1px solid #d2e3fc; font-weight: bold; padding: 8px 16px; border-radius: 6px;"
        )
        demo_btn.clicked.connect(self._on_demo_mode)
        header.addWidget(demo_btn)

        root_layout.addLayout(header)

        # Splitter: Left = Camera + Bbox, Right = PIN Keypad + Profiles
        body = QHBoxLayout()
        body.setSpacing(24)

        # ── Left: Camera feed ─────────────────────────────────────────────────
        left_card = QFrame()
        left_card.setStyleSheet("background: white; border: 1px solid #dadce0; border-radius: 10px; padding: 12px;")
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(10)

        left_title = QLabel("📹 Nhận diện Khuôn mặt Trực tiếp")
        left_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        left_title.setStyleSheet("color: #202124;")
        left_layout.addWidget(left_title)

        self._video_view = QLabel()
        self._video_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_view.setMinimumSize(480, 360)
        self._video_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_view.setStyleSheet("background: #1a1a2e; border-radius: 8px;")
        left_layout.addWidget(self._video_view)

        # Progress / Status indicator
        self._status_bar = QLabel("🔍 Đang quan sát... Hãy nhìn thẳng vào camera để tự động đăng nhập.")
        self._status_bar.setStyleSheet("color: #5f6368; font-size: 11px;")
        left_layout.addWidget(self._status_bar)

        self._liveness_bar = QProgressBar()
        self._liveness_bar.setRange(0, LIVENESS_FRAMES)
        self._liveness_bar.setValue(0)
        self._liveness_bar.setFixedHeight(6)
        self._liveness_bar.setStyleSheet("""
            QProgressBar { background: #e0e0e0; border-radius: 3px; }
            QProgressBar::chunk { background: #34a853; border-radius: 3px; }
        """)
        left_layout.addWidget(self._liveness_bar)

        btn_row_left = QHBoxLayout()
        self._btn_enroll = QPushButton("➕ Đăng ký Giáo viên mới")
        self._btn_enroll.setStyleSheet(
            "background: #1a73e8; color: white; font-weight: bold; padding: 9px 16px; border-radius: 6px;"
        )
        self._btn_enroll.clicked.connect(self._on_enroll_clicked)
        btn_row_left.addWidget(self._btn_enroll)

        left_layout.addLayout(btn_row_left)
        body.addWidget(left_card, stretch=3)

        # ── Right: PIN Keypad & Profile Selector ─────────────────────────────
        right_card = QFrame()
        right_card.setStyleSheet("background: white; border: 1px solid #dadce0; border-radius: 10px; padding: 16px;")
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(12)

        right_title = QLabel("🔢 Đăng nhập bằng mã PIN")
        right_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        right_title.setStyleSheet("color: #202124;")
        right_layout.addWidget(right_title)

        right_layout.addWidget(QLabel("Chọn hồ sơ Giáo viên:"))
        self._teacher_combo = QComboBox()
        self._teacher_combo.setStyleSheet("padding: 7px; border: 1px solid #dadce0; border-radius: 4px;")
        right_layout.addWidget(self._teacher_combo)

        # PIN display
        self._pin_display = QLineEdit()
        self._pin_display.setReadOnly(True)
        self._pin_display.setEchoMode(QLineEdit.EchoMode.Password)
        self._pin_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pin_display.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self._pin_display.setPlaceholderText("• • • •")
        self._pin_display.setStyleSheet(
            "background: #f8f9fa; border: 2px solid #dadce0; border-radius: 6px; padding: 8px; letter-spacing: 4px;"
        )
        right_layout.addWidget(self._pin_display)

        # Numeric Keypad (3x4 grid)
        keypad_grid = QGridLayout()
        keypad_grid.setSpacing(6)

        buttons = [
            ("1", 0, 0), ("2", 0, 1), ("3", 0, 2),
            ("4", 1, 0), ("5", 1, 1), ("6", 1, 2),
            ("7", 2, 0), ("8", 2, 1), ("9", 2, 2),
            ("Xóa", 3, 0), ("0", 3, 1), ("⌫", 3, 2),
        ]

        for text, r, c in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(68, 48)
            btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            if text in ("Xóa", "⌫"):
                btn.setStyleSheet(
                    "background: #f1f3f4; color: #5f6368; border: 1px solid #dadce0; border-radius: 6px;"
                )
            else:
                btn.setStyleSheet(
                    "background: #ffffff; color: #202124; border: 1px solid #dadce0; border-radius: 6px;"
                )
            btn.clicked.connect(lambda _, t=text: self._on_keypad_press(t))
            keypad_grid.addWidget(btn, r, c)

        right_layout.addLayout(keypad_grid)

        # Submit PIN button
        self._btn_submit_pin = QPushButton("Đăng nhập bằng PIN")
        self._btn_submit_pin.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._btn_submit_pin.setStyleSheet(
            "background: #34a853; color: white; padding: 10px; border-radius: 6px;"
        )
        self._btn_submit_pin.clicked.connect(self._on_submit_pin)
        right_layout.addWidget(self._btn_submit_pin)

        body.addWidget(right_card, stretch=2)
        root_layout.addLayout(body, stretch=1)

    # ── Camera stream lifecycle ───────────────────────────────────────────────

    def _start_camera(self) -> None:
        worker = _LoginCameraWorker(self.service, self.camera_index)
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.frame_ready.connect(self._on_frame_ready)
        worker.face_authenticated.connect(self._on_face_auth_success)
        worker.enroll_sample_captured.connect(self._on_enroll_sample)
        worker.enroll_finished.connect(self._on_enroll_finished)
        worker.error_occurred.connect(lambda err: self._status_bar.setText(f"Lỗi: {err}"))

        thread.started.connect(worker.run)
        thread.start()

        self._cam_worker = worker
        self._cam_thread = thread

    def _stop_camera(self) -> None:
        if self._cam_worker:
            self._cam_worker.stop()
        if self._cam_thread and self._cam_thread.isRunning():
            self._cam_thread.quit()
            self._cam_thread.wait(2000)
        self._cam_worker = None
        self._cam_thread = None

    # ── Video frame rendering with YuNet bbox ─────────────────────────────────

    @Slot(np.ndarray, object)
    def _on_frame_ready(self, frame_bgr: np.ndarray, faces: Optional[np.ndarray]) -> None:
        """Draw face bounding boxes and render onto label."""
        canvas = frame_bgr.copy()

        if faces is not None and len(faces) > 0:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                score = float(face[14])
                color = (0, 255, 100) if score > 0.6 else (0, 200, 255)

                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
                label = f"{score:.0%}"
                cv2.putText(canvas, label, (x, max(20, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Draw eye landmarks if present
                if len(face) >= 14:
                    cv2.circle(canvas, (int(face[4]), int(face[5])), 3, (255, 0, 0), -1)  # right eye
                    cv2.circle(canvas, (int(face[6]), int(face[7])), 3, (0, 0, 255), -1)  # left eye
                    cv2.circle(canvas, (int(face[8]), int(face[9])), 3, (0, 255, 0), -1)  # nose

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self._video_view.width(), self._video_view.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_view.setPixmap(pix)

    # ── Authentication events ─────────────────────────────────────────────────

    @Slot(object)
    def _on_face_auth_success(self, teacher_row) -> None:
        """Triggered automatically when face matches."""
        name = teacher_row["full_name"]
        self._status_bar.setText(f"🎉 Xác thực thành công: {name}!")
        self._stop_camera()
        self.login_successful.emit(teacher_row)

    def _on_keypad_press(self, key: str) -> None:
        if key == "Xóa":
            self._pin_buffer = ""
        elif key == "⌫":
            self._pin_buffer = self._pin_buffer[:-1]
        elif len(self._pin_buffer) < 6 and key.isdigit():
            self._pin_buffer += key

        self._pin_display.setText(self._pin_buffer)

    def _on_submit_pin(self) -> None:
        if not self._pin_buffer:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập mã PIN.")
            return

        t_id = self._teacher_combo.currentData()
        if not t_id:
            QMessageBox.warning(self, "Lỗi", "Chưa chọn hồ sơ giáo viên.")
            return

        row = self.service.authenticate_by_pin(t_id, self._pin_buffer)
        if row is not None:
            self._status_bar.setText(f"🎉 Đăng nhập thành công bằng PIN!")
            self._stop_camera()
            self.login_successful.emit(row)
        else:
            QMessageBox.critical(self, "Thất bại", "Mã PIN không chính xác. Vui lòng thử lại.")
            self._pin_buffer = ""
            self._pin_display.clear()

    def _on_demo_mode(self) -> None:
        self._stop_camera()
        self.demo_mode_requested.emit()

    # ── Enrollment flow ───────────────────────────────────────────────────────

    def _on_enroll_clicked(self) -> None:
        dlg = TeacherEnrollDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._pending_enroll_name = dlg.full_name
        self._pending_enroll_subject = dlg.subject
        self._pending_enroll_pin = dlg.pin

        self._status_bar.setText("📸 Đang chụp 5 mẫu khuôn mặt... Vui lòng nhìn thẳng và nghiêng nhẹ.")
        if self._cam_worker:
            self._cam_worker.set_enroll_mode(True)

    @Slot(int, int)
    def _on_enroll_sample(self, current: int, total: int) -> None:
        self._status_bar.setText(f"📸 Đã thu được {current}/{total} khung hình khuôn mặt...")
        self._liveness_bar.setValue(int(current * (LIVENESS_FRAMES / total)))

    @Slot(list)
    def _on_enroll_finished(self, frames: list) -> None:
        try:
            teacher_id = self.service.enroll_from_frames(
                frames=frames,
                full_name=self._pending_enroll_name,
                subject=self._pending_enroll_subject,
                pin=self._pending_enroll_pin,
            )
            QMessageBox.information(
                self, "Đăng ký thành công",
                f"Đã tạo hồ sơ cho giáo viên: {self._pending_enroll_name}!\nHệ thống tự động đăng nhập."
            )
            self.refresh_teacher_dropdown()
            # Fetch new teacher row
            cur = self.service._conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,))
            row = cur.fetchone()
            if row is not None:
                self._stop_camera()
                self.login_successful.emit(row)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi đăng ký", f"Không thể đăng ký:\n{exc}")
            self._status_bar.setText("Đăng ký thất bại. Hãy thử lại.")

    def refresh_teacher_dropdown(self) -> None:
        self._teacher_combo.clear()
        teachers = self.service.list_teachers()
        for t in teachers:
            label = f"{t['full_name']} ({t['subject']})" if t['subject'] else t['full_name']
            self._teacher_combo.addItem(label, t["id"])

        has_teachers = len(teachers) > 0
        self._btn_submit_pin.setEnabled(has_teachers)

    def closeEvent(self, event) -> None:
        self._stop_camera()
        super().closeEvent(event)
