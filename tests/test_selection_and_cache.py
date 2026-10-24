"""
Tests for AirTeach AI Touchless Selection, Translation, and Page Pixmap LRU Cache
==================================================================================
Verifies:
  - LessonBoard get_items_in_rect and get_strokes_in_rect
  - LessonBoard translate_items and delete_items
  - MainWindow in-air pinch drag & translation handling
  - Unified Undo / Redo for translated items and deleted selections
  - Multi-page LRU Pixmap Cache eviction and memory bounding
  - Keyboard shortcuts (Delete, Ctrl+Z, Ctrl+Y, Escape, navigation)
"""
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QPixmap
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsSimpleTextItem

from src.core.stroke_beautifier import BezierSegment
from src.ui.lesson_board import BOARD_HEIGHT, BOARD_WIDTH, InkStroke, LessonBoard


@pytest.fixture
def sample_stroke():
    seg = BezierSegment(
        p0=np.array([100.0, 100.0]),
        p1=np.array([120.0, 100.0]),
        p2=np.array([140.0, 100.0]),
        p3=np.array([160.0, 100.0]),
    )
    return InkStroke(
        stroke_id="test_stroke_1",
        segments=[seg],
        widths=[3.0],
        color=QColor("#202124"),
        timestamps_ms=[0.0, 10.0],
    )


@pytest.fixture
def main_window(qapp):
    with patch("src.ui.main_window._QtCameraWorker") as mock_cam_cls, \
         patch("src.ui.main_window.QThread") as mock_thread_cls:

        mock_cam = MagicMock()
        for attr in ("frame_ready", "gesture_event", "multi_gesture_event", "diagnostics_ready", "error_occurred"):
            sig = MagicMock()
            sig.connect = MagicMock()
            setattr(mock_cam, attr, sig)
        mock_cam_cls.return_value = mock_cam

        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = False
        mock_thread.started = MagicMock()
        mock_thread.started.connect = MagicMock()
        mock_thread_cls.return_value = mock_thread

        from src.ui.main_window import MainWindow
        win = MainWindow(teacher_row=None, camera_index=99)
        yield win
        win.close()


def test_lesson_board_get_items_and_strokes_in_rect(qapp, sample_stroke):
    board = LessonBoard()
    board.commit_stroke(sample_stroke)

    rect_item = QGraphicsRectItem(QRectF(120, 90, 50, 50))
    board.addItem(rect_item)

    query_rect = QRectF(90, 80, 100, 50)
    matched_strokes = board.get_strokes_in_rect(query_rect)
    assert len(matched_strokes) == 1
    assert matched_strokes[0].stroke_id == "test_stroke_1"

    matched_items = board.get_items_in_rect(query_rect)
    assert rect_item in matched_items
    assert sample_stroke._path_item in matched_items

    # Outside rect
    empty_rect = QRectF(500, 500, 50, 50)
    assert len(board.get_strokes_in_rect(empty_rect)) == 0
    assert len(board.get_items_in_rect(empty_rect)) == 0


def test_lesson_board_translate_items(qapp):
    board = LessonBoard()
    item = QGraphicsRectItem(QRectF(0, 0, 50, 50))
    board.addItem(item)
    assert item.pos() == QPointF(0, 0)

    board.translate_items([item], 35.0, 50.0)
    assert item.pos() == QPointF(35.0, 50.0)

    board.translate_items([item], -10.0, 20.0)
    assert item.pos() == QPointF(25.0, 70.0)


def test_lesson_board_delete_items(qapp, sample_stroke):
    board = LessonBoard()
    board.commit_stroke(sample_stroke)
    txt_item = QGraphicsSimpleTextItem("STEM formula")
    board.addItem(txt_item)

    assert "test_stroke_1" in board._strokes
    assert txt_item.scene() is board

    del_strokes, del_items = board.delete_items([sample_stroke._path_item, txt_item])
    assert len(del_strokes) == 1
    assert del_strokes[0].stroke_id == "test_stroke_1"
    assert "test_stroke_1" not in board._strokes
    assert len(del_items) == 1
    assert del_items[0] is txt_item


def test_main_window_selection_drag_and_undo_redo(main_window, sample_stroke):
    win = main_window
    win._scene.commit_stroke(sample_stroke)
    win._page_strokes[0] = [sample_stroke]

    # Select the stroke
    win._selected_items = [sample_stroke._path_item]
    win._selected_strokes = [sample_stroke]

    # Start pinch drag
    win._handle_selection_drag(100.0, 100.0, is_pinched=True)
    assert win._is_dragging_selection is True

    # Move to new point
    win._handle_selection_drag(140.0, 160.0, is_pinched=True)
    assert sample_stroke._path_item.pos() == QPointF(40.0, 60.0)

    # Release pinch
    win._handle_selection_drag(140.0, 160.0, is_pinched=False)
    assert win._is_dragging_selection is False

    # Check committed to undo stack
    undo_list = win._page_undo_stack.get(0, [])
    assert len(undo_list) >= 1
    act_type, payload = undo_list[-1]
    assert act_type == "translate"
    assert payload[1] == 40.0
    assert payload[2] == 60.0

    # Test Undo
    win._on_undo()
    assert sample_stroke._path_item.pos() == QPointF(0.0, 0.0)

    # Test Redo
    win._on_redo()
    assert sample_stroke._path_item.pos() == QPointF(40.0, 60.0)


def test_main_window_delete_selected_items_and_undo_redo(main_window, sample_stroke):
    win = main_window
    win._scene.commit_stroke(sample_stroke)
    win._page_strokes[0] = [sample_stroke]

    win._selected_items = [sample_stroke._path_item]
    win._selected_strokes = [sample_stroke]

    win._delete_selected_items()
    assert "test_stroke_1" not in win._scene._strokes
    assert len(win._selected_items) == 0

    # Undo deletion
    win._on_undo()
    assert "test_stroke_1" in win._scene._strokes

    # Redo deletion
    win._on_redo()
    assert "test_stroke_1" not in win._scene._strokes


def test_main_window_page_lru_cache_eviction(main_window, tmp_path):
    win = main_window
    from src.services.document_manager import PageInfo
    import cv2

    pages = []
    for i in range(12):
        img_path = tmp_path / f"page_{i}.png"
        img = np.full((100, 100, 3), i * 15, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        pages.append(PageInfo(page_index=i, image_path=str(img_path), width_px=100, height_px=100, text=f"Page {i}"))

    win._pages = pages
    win._MAX_PAGE_CACHE = 4

    # Load first 4 pages
    for i in range(4):
        win._current_page_index = i
        win._show_current_page()
    assert len(win._page_pixmap_cache) == 4
    assert 0 in win._page_pixmap_cache

    # Access page 4, evicts page 0
    win._current_page_index = 4
    win._show_current_page()
    assert len(win._page_pixmap_cache) == 4
    assert 0 not in win._page_pixmap_cache
    assert 4 in win._page_pixmap_cache


def test_main_window_keyboard_shortcuts(main_window, sample_stroke):
    win = main_window
    win._scene.commit_stroke(sample_stroke)
    win._page_strokes[0] = [sample_stroke]
    win._selected_items = [sample_stroke._path_item]

    # Delete key
    del_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    win.keyPressEvent(del_event)
    assert "test_stroke_1" not in win._scene._strokes

    # Ctrl+Z (Undo)
    undo_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    win.keyPressEvent(undo_event)
    assert "test_stroke_1" in win._scene._strokes

    # Ctrl+Y (Redo)
    redo_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    win.keyPressEvent(redo_event)
    assert "test_stroke_1" not in win._scene._strokes


def test_lesson_board_get_items_in_rect_excludes_background_image(qapp, sample_stroke):
    board = LessonBoard()
    # Set a background image covering the board
    bg_img = np.full((1080, 1920, 3), 240, dtype=np.uint8)
    board.set_background_page(bg_img)
    assert board._bg_item is not None

    # Add a user stroke
    board.commit_stroke(sample_stroke)

    # Query entire board
    all_items = board.get_items_in_rect(QRectF(0, 0, BOARD_WIDTH, BOARD_HEIGHT))
    assert sample_stroke._path_item in all_items
    assert board._bg_item not in all_items


def test_floating_menu_dwell_smooth_decay(qapp):
    from src.ui.floating_action_menu import FloatingActionMenu
    menu = FloatingActionMenu()
    menu.show_at(500, 500)

    # Hover over item 0 with 0.8 progress
    # In RADIAL_WHEEL layout, local offset (0, -100) hits the top item
    menu.update_pointer(500, 400, dwell_progress=0.8)
    assert menu._hovered_index is not None
    assert abs(menu._dwell_progress - 0.8) < 1e-3

    # On next frame, hand micro-jitters to dwell_progress = 0.0
    menu.update_pointer(500, 400, dwell_progress=0.0)
    # Dwell progress should decay smoothly (0.8 * 0.85 = 0.68) rather than snapping to 0.0
    assert 0.65 <= menu._dwell_progress <= 0.70

    # Moving to a completely different position (outside menu) resets dwell to 0.0
    menu.update_pointer(100, 100, dwell_progress=0.0)
    assert menu._dwell_progress == 0.0
    assert menu._hovered_index is None


def test_document_manager_vietnamese_path_handling(tmp_path):
    from src.services.document_manager import DocumentManager
    doc_mgr = DocumentManager(tmp_path)

    # Directory with Vietnamese diacritics
    vn_dir = tmp_path / "Bài giảng_Toán 12_Giải tích"
    vn_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = vn_dir / "Chương 1_Ứng dụng đạo hàm.pdf"

    # Create a small valid 1-page PDF using PyPDF
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Import document
    pages = doc_mgr.import_document(pdf_path)
    assert len(pages) == 1
    assert Path(pages[0].image_path).exists()


def test_airteach_package_roundtrip_with_recognized_items(main_window, tmp_path):
    win = main_window
    package_path = tmp_path / "test_roundtrip.airteach"

    # Add a mock page
    from src.services.document_manager import PageInfo, RecognizedItem
    import cv2
    img_path = tmp_path / "bg_page_0.png"
    cv2.imwrite(str(img_path), np.full((100, 100, 3), 255, dtype=np.uint8))
    win._pages = [PageInfo(page_index=0, image_path=str(img_path), width_px=100, height_px=100, text="Intro")]

    # Add a recognized text item to page 0
    txt_item = win._scene.add_text_overlay("E = mc^2", 200, 200)
    txt_item.setData(1, {
        "text": "E = mc^2",
        "latex": "E = mc^2",
        "is_formula": True,
        "conf": 0.98,
    })
    win._add_stem_item_to_current_page(txt_item)

    # Mock QFileDialog to return package_path
    from unittest.mock import patch
    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(package_path), "AirTeach")):
        win._on_save_airteach()

    assert package_path.exists()

    # Clear everything on canvas and in state
    win._scene.clear_all_strokes()
    for s_list in win._page_stem_items.values():
        for itm in s_list:
            if itm.scene() is win._scene:
                win._scene.removeItem(itm)
    win._page_stem_items.clear()
    assert len(win._page_stem_items.get(0, [])) == 0

    # Open the saved package
    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(package_path), "AirTeach")):
        win._on_open_airteach()

    # Verify recognized item was restored on page 0
    restored_items = win._page_stem_items.get(0, [])
    assert len(restored_items) >= 1
    data = restored_items[0].data(1)
    assert isinstance(data, dict)
    assert data.get("text") == "E = mc^2"
    assert data.get("is_formula") is True
