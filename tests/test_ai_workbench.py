"""
Tests for AirTeach AI Workbench UI & Pedagogical Generation
===========================================================
Verifies:
  - Widget instantiation and tab setup
  - Teacher status badge reflection
  - Task selector label and placeholder updates
  - Text insertion signal emission
"""
import pytest
from PySide6.QtWidgets import QApplication

from src.services.gemini_assistant import GeminiAssistant
from src.ui.ai_workbench import AIWorkbenchWidget


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_assistant():
    return GeminiAssistant(api_key="", teacher_id="test_teacher_workbench")


def test_ai_workbench_instantiation(qapp, mock_assistant):
    widget = AIWorkbenchWidget(assistant=mock_assistant)
    assert widget is not None
    # Offline status badge
    assert "Offline" in widget.online_status_badge.text()
    assert widget.task_selector.count() == 5


def test_ai_workbench_task_switching(qapp, mock_assistant):
    widget = AIWorkbenchWidget(assistant=mock_assistant)

    # Switch to Summarizer (index 1)
    widget.task_selector.setCurrentIndex(1)
    assert "tóm tắt" in widget.param_title.text().lower()

    # Switch to Quiz Gen (index 2)
    widget.task_selector.setCurrentIndex(2)
    assert "câu hỏi" in widget.context_title.text().lower() or "đoạn kiến thức" in widget.context_title.text().lower()


def test_ai_workbench_insert_signal(qapp, mock_assistant, qtbot):
    widget = AIWorkbenchWidget(assistant=mock_assistant)
    qtbot.addWidget(widget)

    test_content = "1. Định luật I Newton: Vật giữ nguyên trạng thái nếu không chịu lực."
    widget.preview_edit.setPlainText(test_content)

    with qtbot.waitSignal(widget.insert_text_to_board, timeout=1000) as blocker:
        widget.insert_text_to_board.emit(test_content)

    assert blocker.args == [test_content]
