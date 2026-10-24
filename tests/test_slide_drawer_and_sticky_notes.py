"""
Unit tests for Slide Thumbnail Drawer and Acrylic Floating Sticky Notes.

Validates:
- SlideThumbnailDrawer thumbnail generation, row selection, and action signals
- StickyNoteItem dimensions, themes, collapse toggling, serialization, and rendering
- LessonBoard sticky note lifecycle management
- MainWindow multi-page drawer integration, blank slide creation, slide duplication/deletion,
  and page-bounded sticky note persistence
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.services.document_manager import PageInfo
from src.ui.lesson_board import BOARD_HEIGHT, BOARD_WIDTH, LessonBoard
from src.ui.slide_thumbnail_drawer import SlideThumbnailDrawer
from src.ui.sticky_note_item import STICKY_THEMES, StickyNoteItem


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── 1. SlideThumbnailDrawer Tests ─────────────────────────────────────────────

def test_slide_drawer_initialization(qapp):
    """Test SlideThumbnailDrawer default state."""
    drawer = SlideThumbnailDrawer()
    assert drawer.list_widget.count() == 0
    assert not drawer.btn_del.isEnabled()
    assert not drawer.btn_dup.isEnabled()
    assert drawer.btn_add.isEnabled()


def test_slide_drawer_set_pages(qapp, tmp_path):
    """Test setting pages populates list items with correct titles."""
    drawer = SlideThumbnailDrawer()

    # Create dummy slide images
    p1 = tmp_path / "page_1.png"
    p2 = tmp_path / "page_2.png"
    QImage(160, 90, QImage.Format.Format_RGB32).save(str(p1), "PNG")
    QImage(160, 90, QImage.Format.Format_RGB32).save(str(p2), "PNG")

    pages = [
        PageInfo(page_index=0, image_path=str(p1), width_px=1920, height_px=1080, text="Slide 1"),
        PageInfo(page_index=1, image_path=str(p2), width_px=1920, height_px=1080, text="Slide 2"),
    ]

    drawer.set_pages(pages, current_index=0)
    assert drawer.list_widget.count() == 2
    assert drawer.list_widget.item(0).text() == "Trang 1"
    assert drawer.list_widget.item(1).text() == "Trang 2"
    assert drawer.list_widget.currentRow() == 0
    assert drawer.btn_del.isEnabled()
    assert drawer.btn_dup.isEnabled()


def test_slide_drawer_selection_signal(qapp, tmp_path):
    """Test clicking a slide item emits page_selected with target index."""
    drawer = SlideThumbnailDrawer()
    p1 = tmp_path / "page_1.png"
    p2 = tmp_path / "page_2.png"
    QImage(160, 90, QImage.Format.Format_RGB32).save(str(p1), "PNG")
    QImage(160, 90, QImage.Format.Format_RGB32).save(str(p2), "PNG")

    pages = [
        PageInfo(page_index=0, image_path=str(p1), width_px=1920, height_px=1080, text="Slide 1"),
        PageInfo(page_index=1, image_path=str(p2), width_px=1920, height_px=1080, text="Slide 2"),
    ]
    drawer.set_pages(pages, current_index=0)

    emitted_pages = []
    drawer.page_selected.connect(emitted_pages.append)

    drawer.list_widget.setCurrentRow(1)
    assert len(emitted_pages) == 1
    assert emitted_pages[0] == 1


def test_slide_drawer_action_buttons(qapp):
    """Test toolbar action buttons emit corresponding signals."""
    drawer = SlideThumbnailDrawer()

    add_signals = []
    dup_signals = []
    del_signals = []

    drawer.add_blank_page_requested.connect(lambda: add_signals.append(True))
    drawer.duplicate_page_requested.connect(lambda: dup_signals.append(True))
    drawer.delete_page_requested.connect(lambda: del_signals.append(True))

    drawer.btn_dup.setEnabled(True)
    drawer.btn_del.setEnabled(True)

    drawer.btn_add.click()
    drawer.btn_dup.click()
    drawer.btn_del.click()

    assert len(add_signals) == 1
    assert len(dup_signals) == 1
    assert len(del_signals) == 1


# ── 2. StickyNoteItem Tests ───────────────────────────────────────────────────

def test_sticky_note_themes_integrity():
    """Ensure all sticky note pastel color themes are valid."""
    themes = ["yellow", "green", "blue", "coral", "lavender"]
    for t in themes:
        assert t in STICKY_THEMES
        cfg = STICKY_THEMES[t]
        assert isinstance(cfg["bg"], QColor)
        assert isinstance(cfg["header"], QColor)
        assert isinstance(cfg["border"], QColor)
        assert isinstance(cfg["text"], QColor)


def test_sticky_note_dimensions_and_collapse(qapp):
    """Test sticky note dimensions in expanded and collapsed modes."""
    note = StickyNoteItem(title="Ôn tập", text="Xem lại định luật Ôm", theme="green")
    assert not note.is_collapsed
    assert note.boundingRect().width() > StickyNoteItem.EXPANDED_WIDTH

    # Collapse
    assert note.toggle_collapse() is True
    assert note.is_collapsed is True
    assert note.boundingRect().width() < StickyNoteItem.EXPANDED_WIDTH + 10.0

    # Expand
    assert note.toggle_collapse() is False
    assert note.is_collapsed is False


def test_sticky_note_theme_cycling(qapp):
    """Test cycling through all pastel themes."""
    note = StickyNoteItem(theme="yellow")
    assert note.theme == "yellow"
    assert note.cycle_theme() == "green"
    assert note.cycle_theme() == "blue"
    assert note.cycle_theme() == "coral"
    assert note.cycle_theme() == "lavender"
    assert note.cycle_theme() == "yellow"


def test_sticky_note_serialization(qapp):
    """Test serializing note to dict and reconstructing from dict."""
    note = StickyNoteItem(
        title="Công thức Toán",
        text="S = pi * r^2",
        theme="blue",
        x=250.0,
        y=350.0,
        is_collapsed=True,
    )

    d = note.to_dict()
    assert d["title"] == "Công thức Toán"
    assert d["text"] == "S = pi * r^2"
    assert d["theme"] == "blue"
    assert d["x"] == 250.0
    assert d["y"] == 350.0
    assert d["is_collapsed"] is True

    # Reconstruct
    restored = StickyNoteItem.from_dict(d)
    assert restored.title == note.title
    assert restored.text == note.text
    assert restored.theme == note.theme
    assert restored.x() == note.x()
    assert restored.y() == note.y()
    assert restored.is_collapsed == note.is_collapsed


def test_sticky_note_paint_execution(qapp):
    """Ensure paint renders smoothly without error in both states."""
    note = StickyNoteItem(title="Tiêu đề", text="Nội dung ghi chú chi tiết", theme="coral")
    img = QImage(300, 200, QImage.Format.Format_ARGB32)
    p = QPainter(img)

    # Expanded
    note.paint(p, None)

    # Collapsed
    note.toggle_collapse()
    note.paint(p, None)

    p.end()


# ── 3. LessonBoard Sticky Note Lifecycle Tests ────────────────────────────────

def test_lesson_board_add_and_get_sticky_notes(qapp):
    """Test adding sticky notes to LessonBoard and querying active notes."""
    board = LessonBoard()
    assert len(board.get_sticky_notes()) == 0

    note1 = board.add_sticky_note(title="Ghi chú 1", text="Nội dung 1", x=100.0, y=100.0)
    note2 = board.add_sticky_note(title="Ghi chú 2", text="Nội dung 2", x=300.0, y=100.0)

    notes = board.get_sticky_notes()
    assert len(notes) == 2
    assert note1 in notes
    assert note2 in notes

    # Delete note1
    note1.note_deleted.emit(note1.note_id)
    board.removeItem(note1)
    notes_after = board.get_sticky_notes()
    assert len(notes_after) == 1
    assert note2 in notes_after

    # Clear all
    board.clear_sticky_notes()
    assert len(board.get_sticky_notes()) == 0


# ── 4. MainWindow Integration Tests ───────────────────────────────────────────

def test_main_window_slide_drawer_and_sticky_notes(qapp, tmp_path):
    """Test MainWindow slide dock, jumping, blank page creation, duplication, and deletion."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        # 1. Slide dock exists and has SlideThumbnailDrawer
        assert hasattr(win, "_slide_dock")
        assert hasattr(win, "_slide_drawer")
        assert isinstance(win._slide_drawer, SlideThumbnailDrawer)

        # 2. Toggle slide dock visibility
        was_hidden = win._slide_dock.isHidden()
        win._on_toggle_slide_dock()
        assert win._slide_dock.isHidden() != was_hidden
        win._on_toggle_slide_dock()
        assert win._slide_dock.isHidden() == was_hidden

        # 3. Create dummy pages
        p1 = tmp_path / "page_1.png"
        p2 = tmp_path / "page_2.png"
        QImage(160, 90, QImage.Format.Format_RGB32).save(str(p1), "PNG")
        QImage(160, 90, QImage.Format.Format_RGB32).save(str(p2), "PNG")

        win._pages = [
            PageInfo(page_index=0, image_path=str(p1), width_px=1920, height_px=1080, text="P1"),
            PageInfo(page_index=1, image_path=str(p2), width_px=1920, height_px=1080, text="P2"),
        ]
        win._slide_drawer.set_pages(win._pages, 0)
        assert win._slide_drawer.list_widget.count() == 2

        # 4. Jump to page 1 via _go_to_page
        win._go_to_page(1)
        assert win._current_page_index == 1

        # 5. Add a sticky note on Page 1
        note = win._scene.add_sticky_note(title="Ghi chú Slide 2", text="Quan trọng", x=200, y=200)
        win._page_sticky_notes[win._current_page_index] = [note]

        # 6. Duplicate page 1 -> creates Page 2
        win._on_duplicate_page()
        assert len(win._pages) == 3
        assert win._current_page_index == 2

        # 7. Add blank page
        win._on_add_blank_page()
        assert len(win._pages) == 4
        assert win._current_page_index == 3

        # 8. Delete current page
        win._on_delete_page()
        assert len(win._pages) == 3

        # 9. Jump back to Page 1 and verify sticky note restoration
        win._go_to_page(1)
        active_notes = win._scene.get_sticky_notes()
        assert len(active_notes) >= 1
    finally:
        win.close()
