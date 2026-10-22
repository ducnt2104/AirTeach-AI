"""
AirTeach AI - Face Authentication Dialog
==========================================
Two-mode QDialog:
  1. LOGIN  — Shows live webcam feed, runs continuous face detection,
              shows countdown liveness check, falls back to PIN entry.
  2. ENROLL — Collects teacher info (name, subject, PIN), captures
              5 face frames at varied angles, stores profile.

All camera + face model inference runs on background QThread (FaceAuthWorker).
UI thread only receives QImages (pre-converted BGR → RGB → QImage) via Signal.
"""
from __future__ import annotations

import logging
import uuid
from enum import Enum, auto
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.services.face_profile_service import (
    LivenessChecker,
    TeacherProfileService,
    LIVENESS_FRAMES,
    ENROLL_FRAMES,
)

logger = logging.getLogger(__name__)


class DialogMode(Enum):
    LOGIN  = auto()
    ENROLL = auto()


# ── Background worker ─────────────────────────────────────────────────────────

class FaceAuthWorker(QObject):
    """
    Background QThread worker that:
      - Opens webcam with cv2.VideoCapture
      - Applies CLAHE on L channel
      - Runs YuNet detection + SFace authentication each frame
      - Emits frame_ready (QImage) for display
      - Emits auth_success (teacher_row dict) on match
      - Emits liveness_progress (int frame_count) for progress bar
      - In ENROLL mode: collects frames for enrollment
    """

    frame_ready        = Signal(QImage)
    liveness_progress  = Signal(int)
    auth_success       = Signal(object)      # sqlite3.Row or dict
    enroll_frames_collected = Signal(list)   # list of np.ndarray
    error_occurred     = Signal(str)

    def __init__(
        self,
        service: TeacherProfileService,
        mode: DialogMode,
        camera_index: int = 0,
        enroll_count: int = ENROLL_FRAMES,
    ) -> None:
        super().__init__()
        self._service       = service
        self._mode          = mode
        self._camera_index  = camera_index
        self._enroll_count  = enroll_count
        self._running       = True
        self._liveness      = LivenessChecker(required_frames=LIVENESS_FRAMES)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        if self._camera_index < 0:
            return

        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            self.error_occurred.emit(f"Không thể mở camera index={self._camera_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        collected_frames: list[np.ndarray] = []

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            # CLAHE on L channel (backlight correction)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            frame_enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

            # Emit frame for display (BGR → RGB → QImage)
            rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qi)

            if self._mode == DialogMode.LOGIN:
                # Update liveness
                faces = self._service._models.detect_faces(frame_enhanced)
                if faces is not None and len(faces) > 0:
                    best = faces[np.argmax(faces[:, 14])]
                    self._liveness.update(best)
                    self.liveness_progress.emit(len(self._liveness._nose_history))

                    if self._liveness.is_ready():
                        teacher = self._service.authenticate_by_face(
                            frame_enhanced, self._liveness
                        )
                        if teacher is not None:
                            self.auth_success.emit(teacher)
                            self._running = False

            elif self._mode == DialogMode.ENROLL:
                faces = self._service._models.detect_faces(frame_enhanced)
                if faces is not None and len(faces) > 0:
                    best_score = float(faces[:, 14].max())
                    # Only collect high-confidence detections
                    if best_score > 0.4:
                        collected_frames.append(frame_enhanced.copy())
                        self.liveness_progress.emit(len(collected_frames))
                        if len(collected_frames) >= self._enroll_count:
                            self.enroll_frames_collected.emit(collected_frames)
                            self._running = False

        cap.release()


# ── Main Dialog ───────────────────────────────────────────────────────────────

class FaceAuthDialog(QDialog):
    """
    Teacher authentication / enrollment dialog.

    Args:
        service:        TeacherProfileService instance.
        mode:           DialogMode.LOGIN or DialogMode.ENROLL.
        camera_index:   Webcam index (default 0).
        parent:         Parent widget.

    Signals:
        teacher_authenticated(object): emitted with sqlite3.Row on successful login.
        teacher_enrolled(str):         emitted with new teacher UUID on enrollment.
    """

    teacher_authenticated = Signal(object)
    teacher_enrolled      = Signal(str)

    def __init__(
        self,
        service: TeacherProfileService,
        mode: DialogMode = DialogMode.LOGIN,
        camera_index: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._service       = service
        self._mode          = mode
        self._camera_index  = camera_index
        self._worker: Optional[FaceAuthWorker] = None
        self._thread: Optional[QThread]        = None
        self._collected_frames: list[np.ndarray] = []

        title = "Đăng ký Giáo viên" if mode == DialogMode.ENROLL else "Xác thực Giáo viên"
        self.setWindowTitle(title)
        self.setMinimumSize(640, 560)
        self.setModal(True)
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # Title
        lbl = QLabel(
            "📷 Đặt khuôn mặt vào khung hình và giữ yên trong vài giây"
            if self._mode == DialogMode.LOGIN
            else "📷 Xoay nhẹ đầu theo các hướng khác nhau khi đèn nhấp nháy"
        )
        lbl.setFont(QFont("Segoe UI", 11))
        lbl.setWordWrap(True)
        root.addWidget(lbl)

        # Camera preview
        self._cam_label = QLabel("Đang khởi động camera…")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet(
            "background:#1a1a2e; color:#8d99ae; border:2px solid #3d5a80;"
            "border-radius:8px; min-height:240px;"
        )
        self._cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._cam_label, stretch=2)

        # Liveness progress
        prog_lbl = QLabel(
            "Tiến trình kiểm tra sống:" if self._mode == DialogMode.LOGIN
            else "Thu thập khung hình khuôn mặt:"
        )
        root.addWidget(prog_lbl)
        self._progress = QProgressBar()
        self._progress.setRange(0, LIVENESS_FRAMES if self._mode == DialogMode.LOGIN else ENROLL_FRAMES)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        # Enroll form (only in ENROLL mode)
        if self._mode == DialogMode.ENROLL:
            self._enroll_group = self._build_enroll_form()
            root.addWidget(self._enroll_group)

        # PIN fallback (LOGIN mode only)
        if self._mode == DialogMode.LOGIN:
            self._pin_group = self._build_pin_fallback()
            root.addWidget(self._pin_group)

        # Buttons
        btn_row = QHBoxLayout()
        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(self._stop_and_close)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _build_enroll_form(self) -> QGroupBox:
        box = QGroupBox("Thông tin giáo viên")
        form_layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Họ và tên:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Nguyễn Thị Lan")
        row1.addWidget(self._name_edit)
        form_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Môn dạy:"))
        self._subject_edit = QLineEdit()
        self._subject_edit.setPlaceholderText("Toán, Lý, Hóa…")
        row2.addWidget(self._subject_edit)
        form_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Mã PIN (4-6 số):"))
        self._pin_edit = QLineEdit()
        self._pin_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._pin_edit.setMaxLength(6)
        self._pin_edit.setPlaceholderText("••••")
        row3.addWidget(self._pin_edit)
        form_layout.addLayout(row3)

        self._enroll_btn = QPushButton("✅ Hoàn tất đăng ký")
        self._enroll_btn.setEnabled(False)  # enabled after frames collected
        self._enroll_btn.clicked.connect(self._on_do_enroll)
        self._enroll_btn.setStyleSheet(
            "QPushButton { background:#34a853; color:white; border:none; "
            "border-radius:6px; padding:8px 16px; font-size:12px; }"
            "QPushButton:hover { background:#2d9249; }"
            "QPushButton:disabled { background:#bdc1c6; }"
        )
        form_layout.addWidget(self._enroll_btn)
        return box

    def _build_pin_fallback(self) -> QGroupBox:
        box = QGroupBox("Đăng nhập bằng PIN dự phòng")
        layout = QVBoxLayout(box)

        teachers = self._service.list_teachers()
        if not teachers:
            layout.addWidget(QLabel("Chưa có giáo viên nào được đăng ký."))
            return box

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Tên giáo viên:"))
        self._pin_name_edit = QLineEdit()
        self._pin_name_edit.setPlaceholderText("Nhập đúng họ tên đầy đủ")
        row1.addWidget(self._pin_name_edit)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Mã PIN:"))
        self._pin_input = QLineEdit()
        self._pin_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._pin_input.setMaxLength(6)
        row2.addWidget(self._pin_input)
        layout.addLayout(row2)

        pin_btn = QPushButton("🔑 Đăng nhập PIN")
        pin_btn.clicked.connect(self._on_pin_login)
        pin_btn.setStyleSheet(
            "QPushButton { background:#fbbc04; color:#202124; border:none; "
            "border-radius:6px; padding:6px 14px; }"
        )
        layout.addWidget(pin_btn)
        return box

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._start_worker()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_worker()
        super().closeEvent(event)

    def _stop_and_close(self) -> None:
        self._stop_worker()
        self.reject()

    def _start_worker(self) -> None:
        if self._camera_index < 0:
            logger.info("Camera index is negative (%d); FaceAuthDialog worker not started.", self._camera_index)
            return

        worker = FaceAuthWorker(
            service=self._service,
            mode=self._mode,
            camera_index=self._camera_index,
            enroll_count=ENROLL_FRAMES,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.frame_ready.connect(self._on_frame, Qt.ConnectionType.QueuedConnection)
        worker.liveness_progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        worker.auth_success.connect(self._on_auth_success, Qt.ConnectionType.QueuedConnection)
        worker.enroll_frames_collected.connect(
            self._on_frames_collected, Qt.ConnectionType.QueuedConnection
        )
        worker.error_occurred.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        thread.started.connect(worker.run)
        thread.start()
        self._worker = worker
        self._thread = thread

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(1000):
                self._thread.terminate()
                self._thread.wait(500)
        self._worker = None
        self._thread = None

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)

    def reject(self) -> None:
        self._stop_worker()
        super().reject()

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_frame(self, qi: QImage) -> None:
        pixmap = QPixmap.fromImage(qi)
        scaled = pixmap.scaled(
            self._cam_label.width(), self._cam_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cam_label.setPixmap(scaled)

    def _on_progress(self, count: int) -> None:
        self._progress.setValue(count)

    def _on_auth_success(self, teacher_row) -> None:
        self._stop_worker()
        QMessageBox.information(
            self,
            "Xác thực thành công",
            f"Chào mừng, {teacher_row['full_name']}! 👋",
        )
        self.teacher_authenticated.emit(teacher_row)
        self.accept()

    def _on_frames_collected(self, frames: list) -> None:
        self._collected_frames = frames
        if self._mode == DialogMode.ENROLL:
            self._enroll_btn.setEnabled(True)
            QMessageBox.information(
                self, "Sẵn sàng đăng ký",
                f"Đã thu thập {len(frames)} khung hình khuôn mặt.\n"
                "Hãy nhập thông tin và bấm 'Hoàn tất đăng ký'."
            )

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Lỗi camera", msg)

    # ── Enrollment ────────────────────────────────────────────────────────────

    def _on_do_enroll(self) -> None:
        name    = self._name_edit.text().strip()
        subject = self._subject_edit.text().strip()
        pin     = self._pin_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập họ và tên.")
            return
        if not (4 <= len(pin) <= 6 and pin.isdigit()):
            QMessageBox.warning(self, "Mã PIN không hợp lệ", "PIN phải gồm 4–6 chữ số.")
            return
        if not self._collected_frames:
            QMessageBox.warning(self, "Chưa có ảnh", "Vui lòng để camera thu thập khuôn mặt trước.")
            return

        try:
            teacher_id = self._service.enroll_from_frames(
                frames=self._collected_frames,
                full_name=name,
                subject=subject,
                pin=pin,
            )
            QMessageBox.information(
                self, "Đăng ký thành công",
                f"Hồ sơ giáo viên '{name}' đã được tạo.\n"
                f"ID: {teacher_id}"
            )
            self.teacher_enrolled.emit(teacher_id)
            self.accept()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi đăng ký", str(exc))

    # ── PIN login ─────────────────────────────────────────────────────────────

    def _on_pin_login(self) -> None:
        name = self._pin_name_edit.text().strip()
        pin  = self._pin_input.text().strip()
        if not name or not pin:
            return
        row = self._service.authenticate_by_name_and_pin(name, pin)
        if row is not None:
            self._on_auth_success(row)
        else:
            QMessageBox.warning(
                self, "Sai thông tin",
                "Tên giáo viên hoặc mã PIN không chính xác."
            )
