"""
AirTeach AI - Practical Teacher AI Workbench
============================================
Provides practical, teacher-controlled AI tools for lesson preparation and live teaching:
  1. Lesson Outline Generator (Dàn ý bài giảng có cấu trúc)
  2. Document & Slide Summarizer (Tóm tắt kiến thức trọng tâm)
  3. Differentiated Quiz Generator (Soạn câu hỏi trắc nghiệm đa cấp độ)
  4. STEM Concept Explainer (Diễn giải khái niệm trực quan kèm ví dụ đời thực)
  5. 45-Minute Lesson Plan Assistant (Kế hoạch bài dạy sư phạm)

Zero UI Blocking: Generation runs in background QThreadPool workers.
Teacher Control: Content is previewed and editable before inserting into lesson canvas.
Account Isolated: Uses the active teacher's private API key and settings.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.services.credential_service import TeacherCredentialService
from src.services.gemini_assistant import GeminiAssistant

logger = logging.getLogger(__name__)


class _WorkbenchWorkerSignals(QObject):
    chunk_received = Signal(str)
    finished = Signal(str)
    error_occurred = Signal(str)


class _WorkbenchTaskRunnable(QRunnable):
    """Executes AI tasks asynchronously on a background thread."""

    def __init__(self, task_func, *args, **kwargs) -> None:
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.signals = _WorkbenchWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            # Pass streaming chunk callback
            def on_chunk(chunk_text: str):
                self.signals.chunk_received.emit(chunk_text)

            res = self.task_func(*self.args, on_chunk=on_chunk, **self.kwargs)
            self.signals.finished.emit(res)
        except Exception as exc:
            redacted_err = TeacherCredentialService.redact_text(str(exc))
            self.signals.error_occurred.emit(redacted_err)


class AIWorkbenchWidget(QWidget):
    """
    Pedagogical AI Workspace widget designed for teachers.
    Can be placed inside a dock or displayed as a workspace view.

    Signals:
        insert_text_to_board(str): emitted when teacher approves inserting AI content into board.
    """

    insert_text_to_board = Signal(str)

    def __init__(
        self,
        assistant: GeminiAssistant,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.assistant = assistant
        self._thread_pool = QThreadPool.globalInstance()
        self._build_ui()

    def _build_ui(self) -> None:
        root_v = QVBoxLayout(self)
        root_v.setContentsMargins(12, 12, 12, 12)
        root_v.setSpacing(10)

        # Header
        header_h = QHBoxLayout()
        header_title = QLabel("✨ Trạm soạn thảo & Trợ giảng AI (AI Workbench)")
        header_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header_h.addWidget(header_title)

        self.online_status_badge = QLabel("Đang tải…")
        self.online_status_badge.setStyleSheet(
            "padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold;"
        )
        self._update_status_badge()
        header_h.addStretch(1)
        header_h.addWidget(self.online_status_badge)
        root_v.addLayout(header_h)

        # Main splitter (Top/Left: Task Selector & Parameters, Bottom/Right: Preview & Approval)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Control Panel: Task Form ──────────────────────────────────────────
        ctrl_card = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(8)

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Chọn tác vụ trợ giảng:"))
        self.task_selector = QComboBox()
        self.task_selector.addItems([
            "1. Lập dàn ý bài giảng chi tiết (Lesson Outline)",
            "2. Tóm tắt nội dung bài học / slide (Summarizer)",
            "3. Soạn câu hỏi trắc nghiệm & đề ôn tập (Quiz Gen)",
            "4. Diễn giải trực quan khái niệm STEM khó (Explainer)",
            "5. Soạn kế hoạch bài dạy chuẩn 45 phút (Lesson Plan)",
        ])
        self.task_selector.currentIndexChanged.connect(self._on_task_changed)
        task_row.addWidget(self.task_selector, stretch=1)
        ctrl_layout.addLayout(task_row)

        # Dynamic parameter fields
        self.param_title = QLabel("Chủ đề / Tên bài học:")
        ctrl_layout.addWidget(self.param_title)

        self.input_main = QLineEdit()
        self.input_main.setPlaceholderText("Ví dụ: Định luật bảo toàn cơ năng, Tích phân và ứng dụng, Quang hợp...")
        ctrl_layout.addWidget(self.input_main)

        # Context / Reference text area (e.g. for summarizer or quiz generator)
        self.context_title = QLabel("Văn bản tham khảo hoặc trích dẫn từ bài giảng (tùy chọn):")
        ctrl_layout.addWidget(self.context_title)

        self.input_context = QTextEdit()
        self.input_context.setMaximumHeight(85)
        self.input_context.setPlaceholderText("Dán nội dung tài liệu hoặc văn bản cần tóm tắt / tạo câu hỏi vào đây…")
        ctrl_layout.addWidget(self.input_context)

        # Action Buttons
        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("🚀 Thực hiện tác vụ AI")
        self.btn_generate.setStyleSheet(
            "background: #1a73e8; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px;"
        )
        self.btn_generate.clicked.connect(self._on_generate_clicked)
        btn_row.addWidget(self.btn_generate)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate spinner
        self.progress_bar.setVisible(False)
        btn_row.addWidget(self.progress_bar)
        btn_row.addStretch(1)

        ctrl_layout.addLayout(btn_row)
        splitter.addWidget(ctrl_card)

        # ── Preview, Editing & Approval Area ──────────────────────────────────
        preview_card = QWidget()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(0, 8, 0, 0)
        preview_layout.setSpacing(6)

        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("📝 Kết quả sinh bởi AI (Xem trước & Kiểm duyệt của giáo viên):"))
        preview_header.addStretch(1)

        self.btn_insert = QPushButton("📥 Chèn vào Bảng")
        self.btn_insert.setStyleSheet("background: #0f9d58; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.btn_insert.clicked.connect(self._on_insert_clicked)

        self.btn_copy = QPushButton("📋 Sao chép")
        self.btn_copy.clicked.connect(self._on_copy_clicked)

        self.btn_clear = QPushButton("Xóa")
        self.btn_clear.clicked.connect(self._on_clear_clicked)

        preview_header.addWidget(self.btn_insert)
        preview_header.addWidget(self.btn_copy)
        preview_header.addWidget(self.btn_clear)
        preview_layout.addLayout(preview_header)

        self.preview_edit = QTextEdit()
        self.preview_edit.setFont(QFont("Segoe UI", 10))
        self.preview_edit.setPlaceholderText("Kết quả từ trợ lý AI sẽ xuất hiện tại đây. Giáo viên có thể chỉnh sửa tự do trước khi chèn vào bài học…")
        preview_layout.addWidget(self.preview_edit)

        splitter.addWidget(preview_card)
        root_v.addWidget(splitter, stretch=1)

    # ── State & UI Sync ───────────────────────────────────────────────────────

    def _update_status_badge(self) -> None:
        if self.assistant.is_online():
            self.online_status_badge.setText(f"🟢 Online ({self.assistant.model_name()})")
            self.online_status_badge.setStyleSheet(
                "background: #e6f4ea; color: #137333; padding: 2px 8px; border-radius: 10px; font-weight: bold;"
            )
        else:
            self.online_status_badge.setText("🟠 Offline (Bảo lưu dữ liệu)")
            self.online_status_badge.setStyleSheet(
                "background: #feefe3; color: #b06000; padding: 2px 8px; border-radius: 10px; font-weight: bold;"
            )

    def refresh_teacher_status(self) -> None:
        """Call when active teacher changes."""
        self._update_status_badge()

    def _on_task_changed(self, idx: int) -> None:
        if idx == 0:  # Lesson Outline
            self.param_title.setText("Tên môn học & Chủ đề bài học:")
            self.input_main.setPlaceholderText("Ví dụ: Vật lý 12 - Sóng cơ và sự truyền sóng cơ")
            self.context_title.setText("Yêu cầu bổ sung hoặc mục tiêu cụ thể (tùy chọn):")
        elif idx == 1:  # Summarizer
            self.param_title.setText("Tiêu đề phần tóm tắt:")
            self.input_main.setPlaceholderText("Tóm tắt lý thuyết trọng tâm")
            self.context_title.setText("Nội dung bài học hoặc đoạn văn bản cần tóm tắt (Bắt buộc):")
        elif idx == 2:  # Quiz Gen
            self.param_title.setText("Chủ đề kiểm tra / Đề mục:")
            self.input_main.setPlaceholderText("Ví dụ: Câu hỏi ôn tập chương 1 - Di truyền học")
            self.context_title.setText("Đoạn kiến thức cần bám sát để soạn câu hỏi:")
        elif idx == 3:  # Concept Explainer
            self.param_title.setText("Khái niệm cần giải thích:")
            self.input_main.setPlaceholderText("Ví dụ: Lực hướng tâm, Hiệu ứng nhà kính, Đệ quy...")
            self.context_title.setText("Đối tượng hoặc bối cảnh minh họa:")
        elif idx == 4:  # Lesson Plan
            self.param_title.setText("Tên bài dạy (Giáo án 45 phút):")
            self.input_main.setPlaceholderText("Ví dụ: Bài 10 - Ba định luật Newton")
            self.context_title.setText("Thiết bị dạy học hoặc lưu ý đặc thù:")

    # ── Action Execution ──────────────────────────────────────────────────────

    def _on_generate_clicked(self) -> None:
        idx = self.task_selector.currentIndex()
        topic = self.input_main.text().strip()
        context = self.input_context.toPlainText().strip()

        if not topic and not context:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập chủ đề bài học hoặc nội dung văn bản cần xử lý.")
            return

        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.preview_edit.clear()

        # Build runner
        if idx == 0:  # Outline
            runner = _WorkbenchTaskRunnable(
                self.assistant.generate_lesson_outline,
                subject="",
                topic=topic,
            )
        elif idx == 1:  # Summarize
            text_to_sum = context if context else topic
            runner = _WorkbenchTaskRunnable(
                self.assistant.summarize_document_content,
                text_content=text_to_sum,
            )
        elif idx == 2:  # Quiz
            text_quiz = f"{topic}\n{context}"
            runner = _WorkbenchTaskRunnable(
                self.assistant.generate_quiz_questions,
                content=text_quiz,
            )
        elif idx == 3:  # Concept Explainer
            runner = _WorkbenchTaskRunnable(
                self.assistant.explain_concept,
                concept=topic,
            )
        else:  # Lesson Plan
            runner = _WorkbenchTaskRunnable(
                self.assistant.generate_lesson_plan,
                topic=topic,
            )

        runner.signals.chunk_received.connect(self._on_chunk)
        runner.signals.finished.connect(self._on_finished)
        runner.signals.error_occurred.connect(self._on_error)
        self._thread_pool.start(runner)

    def _on_chunk(self, chunk: str) -> None:
        self.preview_edit.insertPlainText(chunk)

    def _on_finished(self, full_text: str) -> None:
        self.btn_generate.setEnabled(True)
        self.progress_bar.setVisible(False)
        if not self.preview_edit.toPlainText().strip():
            self.preview_edit.setPlainText(full_text)

    def _on_error(self, err: str) -> None:
        self.btn_generate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.preview_edit.setPlainText(f"❌ {err}")

    def _on_insert_clicked(self) -> None:
        text = self.preview_edit.toPlainText().strip()
        if not text:
            return
        self.insert_text_to_board.emit(text)
        QMessageBox.information(
            self,
            "Đã chèn nội dung",
            "Nội dung đã được chèn vào bảng dạy học thành công!",
        )

    def _on_copy_clicked(self) -> None:
        from PySide6.QtGui import QGuiApplication
        text = self.preview_edit.toPlainText().strip()
        if text:
            QGuiApplication.clipboard().setText(text)
            QMessageBox.information(self, "Đã sao chép", "Nội dung đã được sao chép vào bộ nhớ tạm.")

    def _on_clear_clicked(self) -> None:
        self.preview_edit.clear()
