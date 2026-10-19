"""
AirTeach AI - Chat Panel (Gemini AI Assistant UI)
==================================================
QWidget chat panel with:
  - Scrollable conversation history (QTextEdit read-only)
  - Input bar (QLineEdit + Send button)
  - "Explain selected region" button (sends canvas snapshot to Gemini Vision)
  - Streaming text display: new tokens append in real-time
  - Offline mode banner when Gemini is unavailable
"""
from __future__ import annotations

import logging
from typing import Optional, Callable

from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QColor, QTextCursor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ── Worker for background Gemini call ─────────────────────────────────────────

class _ChatWorker(QObject):
    """Runs a GeminiAssistant call on a background thread."""

    chunk_received  = Signal(str)     # partial streaming token
    finished        = Signal()        # call complete
    error_occurred  = Signal(str)     # error message

    def __init__(
        self,
        assistant,                          # GeminiAssistant instance
        history: list[dict],
        user_message: str,
        page_text: str,
    ) -> None:
        super().__init__()
        self._assistant   = assistant
        self._history     = history
        self._user_msg    = user_message
        self._page_text   = page_text

    def run(self) -> None:
        try:
            self._assistant.chat_with_teacher(
                history=self._history,
                user_message=self._user_msg,
                current_page_text=self._page_text,
                on_chunk=self._on_chunk,
            )
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()

    def _on_chunk(self, token: str) -> None:
        self.chunk_received.emit(token)


class _ExplainWorker(QObject):
    """Sends a canvas region image to Gemini Vision."""

    chunk_received = Signal(str)
    finished       = Signal()
    error_occurred = Signal(str)

    def __init__(self, assistant, image_bytes: bytes, prompt: str) -> None:
        super().__init__()
        self._assistant    = assistant
        self._image_bytes  = image_bytes
        self._prompt       = prompt

    def run(self) -> None:
        try:
            self._assistant.explain_canvas_region(
                cropped_image_bytes=self._image_bytes,
                prompt=self._prompt,
                on_chunk=self._on_chunk,
            )
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()

    def _on_chunk(self, token: str) -> None:
        self.chunk_received.emit(token)


# ── Chat Panel Widget ─────────────────────────────────────────────────────────

class ChatPanel(QWidget):
    """
    Collapsible chat panel for the Gemini AI assistant.

    Args:
        assistant: A GeminiAssistant instance.
        parent:    Parent widget.

    Signals:
        request_canvas_snapshot: emitted when "Explain Region" is pressed;
            the main window should call receive_canvas_snapshot() with PNG bytes.
    """

    request_canvas_snapshot = Signal()

    def __init__(self, assistant, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._assistant      = assistant
        self._history: list[dict] = []
        self._current_page_text: str = ""
        self._pending_ai_reply: str = ""
        self._chat_thread: Optional[QThread] = None
        self._chat_worker: Optional[_ChatWorker | _ExplainWorker] = None

        self.setMinimumWidth(300)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Header
        header = QLabel("🤖 AirTeach AI Assistant")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #1a73e8;")
        root.addWidget(header)

        # Offline banner
        self._offline_banner = QLabel(
            "⚠️  Offline mode — using local guide only"
        )
        self._offline_banner.setStyleSheet(
            "background:#fff3cd; color:#856404; border-radius:4px; padding:4px 8px;"
        )
        self._offline_banner.setVisible(not self._assistant.is_online)
        root.addWidget(self._offline_banner)

        # Conversation history
        self._history_view = QTextEdit()
        self._history_view.setReadOnly(True)
        self._history_view.setStyleSheet(
            "background:#ffffff; border:1px solid #dadce0; border-radius:6px; padding:4px;"
        )
        self._history_view.setFont(QFont("Segoe UI", 11))
        root.addWidget(self._history_view, stretch=1)

        # "Explain canvas region" button
        explain_btn = QPushButton("🔍 Explain Selected Region")
        explain_btn.setToolTip("Send the selected canvas area to Gemini for explanation")
        explain_btn.clicked.connect(self._on_explain_region)
        explain_btn.setStyleSheet(self._btn_style("#34a853"))
        root.addWidget(explain_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#dadce0;")
        root.addWidget(sep)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self._input_box = QLineEdit()
        self._input_box.setPlaceholderText("Hỏi AI trợ giảng...")
        self._input_box.setFont(QFont("Segoe UI", 11))
        self._input_box.setStyleSheet(
            "border:1px solid #dadce0; border-radius:6px; padding:6px 10px;"
        )
        self._input_box.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input_box, stretch=1)

        send_btn = QPushButton("Gửi")
        send_btn.setFixedWidth(64)
        send_btn.clicked.connect(self._on_send)
        send_btn.setStyleSheet(self._btn_style("#1a73e8", text_color="white"))
        input_row.addWidget(send_btn)
        root.addLayout(input_row)

    @staticmethod
    def _btn_style(bg: str, text_color: str = "#202124") -> str:
        return (
            f"QPushButton {{ background:{bg}; color:{text_color}; "
            f"border:none; border-radius:6px; padding:6px 12px; font-size:11px; }}"
            f"QPushButton:hover {{ background:{bg}dd; }}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_current_page_text(self, text: str) -> None:
        """Update context text from the currently visible document page."""
        self._current_page_text = text

    def receive_canvas_snapshot(self, png_bytes: bytes) -> None:
        """
        Called by the main window after request_canvas_snapshot was emitted.
        Sends the image to Gemini Vision for explanation.
        """
        self._append_message("Tôi", "[Vùng bảng đã khoanh chọn]", is_user=True)
        self._start_explain_thread(png_bytes, "Hãy giải thích nội dung vùng bảng này.")

    def update_online_status(self, is_online: bool) -> None:
        self._offline_banner.setVisible(not is_online)

    # ── Send message ──────────────────────────────────────────────────────────

    def _on_send(self) -> None:
        text = self._input_box.text().strip()
        if not text:
            return
        self._input_box.clear()
        self._append_message("Tôi", text, is_user=True)
        self._history.append({"role": "user", "content": text})
        self._start_chat_thread(text)

    def _on_explain_region(self) -> None:
        self.request_canvas_snapshot.emit()

    # ── Worker thread management ──────────────────────────────────────────────

    def _start_chat_thread(self, user_msg: str) -> None:
        if self._chat_thread is not None and self._chat_thread.isRunning():
            return  # still processing previous request
        self._pending_ai_reply = ""
        worker = _ChatWorker(
            assistant=self._assistant,
            history=list(self._history[:-1]),
            user_message=user_msg,
            page_text=self._current_page_text,
        )
        self._launch_worker(worker)

    def _start_explain_thread(self, png_bytes: bytes, prompt: str) -> None:
        if self._chat_thread is not None and self._chat_thread.isRunning():
            return
        self._pending_ai_reply = ""
        worker = _ExplainWorker(
            assistant=self._assistant,
            image_bytes=png_bytes,
            prompt=prompt,
        )
        self._launch_worker(worker)

    def _launch_worker(self, worker: QObject) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.chunk_received.connect(self._on_chunk, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_ai_done, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.error_occurred.connect(self._on_ai_error, Qt.ConnectionType.QueuedConnection)
        thread.started.connect(worker.run)
        self._chat_thread = thread
        self._chat_worker = worker
        thread.start()
        # Pre-insert empty AI message placeholder
        self._append_message("AI", "", is_user=False, streaming=True)

    # ── Streaming callbacks ───────────────────────────────────────────────────

    def _on_chunk(self, token: str) -> None:
        """Append streaming token to the last AI message bubble."""
        self._pending_ai_reply += token
        cursor = self._history_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self._history_view.setTextCursor(cursor)
        self._history_view.ensureCursorVisible()

    def _on_ai_done(self) -> None:
        """Finalize AI reply and add to history."""
        if self._pending_ai_reply:
            self._history.append({"role": "model", "content": self._pending_ai_reply})
            self._pending_ai_reply = ""
        # Add newline separator
        self._history_view.append("")

    def _on_ai_error(self, msg: str) -> None:
        self._history_view.append(
            f'<span style="color:#ea4335;">⚠️ Lỗi: {msg}</span>'
        )

    # ── History rendering ─────────────────────────────────────────────────────

    def _append_message(
        self,
        sender: str,
        text: str,
        is_user: bool,
        streaming: bool = False,
    ) -> None:
        color = "#1a73e8" if is_user else "#34a853"
        prefix = f'<b style="color:{color};">{sender}:</b> '
        self._history_view.append(prefix + text)
        if streaming:
            # Move cursor to end so streaming tokens append correctly
            cursor = self._history_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._history_view.setTextCursor(cursor)
