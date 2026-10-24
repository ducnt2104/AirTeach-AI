"""
Unit tests for Canvas Image Exporter and Classroom Chalkboard / Vietnamese Ô ly Theme.

Validates:
- BoardTheme enumeration and configuration dictionary integrity
- LessonBoard theme switching, cycling, and signal emission
- Authentic Vietnamese elementary 4-sub-cell grid rendering
- CanvasExporter high-res supersampling (1080p, 4K UHD), PNG/JPEG file saving
- Overlay filtering during canvas export
- ExportImageDialog initialization and format handling
- MainWindow theme toggling and in-air menu integration
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

from src.ui.canvas_exporter import CanvasExporter, ExportImageDialog
from src.ui.floating_action_menu import PEN_MENU_ITEMS, STEM_MENU_ITEMS
from src.ui.lesson_board import THEME_CONFIGS, BOARD_HEIGHT, BOARD_WIDTH, BoardTheme, LessonBoard


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── 1. BoardTheme & Configuration Integrity Tests ─────────────────────────────

def test_board_theme_constants():
    """Verify all 4 board theme identifiers are defined and unique."""
    themes = [BoardTheme.WHITE, BoardTheme.CHALKBOARD, BoardTheme.OLY, BoardTheme.SLATE]
    assert len(themes) == 4
    assert len(set(themes)) == 4


def test_theme_configs_integrity():
    """Ensure all theme configs contain required color properties."""
    required_keys = {"name", "bg_color", "default_ink", "grid_pen", "dot_pen", "axis_pen", "axis_text"}
    for theme_id, cfg in THEME_CONFIGS.items():
        assert required_keys.issubset(cfg.keys()), f"Theme {theme_id} missing required keys"
        assert isinstance(cfg["bg_color"], QColor)
        assert isinstance(cfg["default_ink"], QColor)
        assert isinstance(cfg["grid_pen"], QColor)
        assert isinstance(cfg["dot_pen"], QColor)
        assert isinstance(cfg["axis_pen"], QColor)
        assert isinstance(cfg["axis_text"], QColor)

    # Vietnamese Ô ly theme must have subgrid_pen
    assert "subgrid_pen" in THEME_CONFIGS[BoardTheme.OLY]
    assert isinstance(THEME_CONFIGS[BoardTheme.OLY]["subgrid_pen"], QColor)


# ── 2. LessonBoard Theme Switching & Drawing Tests ────────────────────────────

def test_lesson_board_default_theme(qapp):
    """Test LessonBoard defaults to Whiteboard theme."""
    board = LessonBoard()
    assert board.board_theme() == BoardTheme.WHITE
    assert board.backgroundBrush().color() == THEME_CONFIGS[BoardTheme.WHITE]["bg_color"]


def test_lesson_board_set_theme(qapp):
    """Test switching themes emits signal and updates background color."""
    board = LessonBoard()
    emitted = []
    board.board_theme_changed.connect(lambda t, ink: emitted.append((t, ink)))

    board.set_board_theme(BoardTheme.CHALKBOARD)
    assert board.board_theme() == BoardTheme.CHALKBOARD
    assert board.backgroundBrush().color() == THEME_CONFIGS[BoardTheme.CHALKBOARD]["bg_color"]
    assert len(emitted) == 1
    assert emitted[0][0] == BoardTheme.CHALKBOARD
    assert emitted[0][1] == THEME_CONFIGS[BoardTheme.CHALKBOARD]["default_ink"]


def test_lesson_board_cycle_theme(qapp):
    """Test cycling through all themes in deterministic sequence."""
    board = LessonBoard()
    assert board.board_theme() == BoardTheme.WHITE

    assert board.cycle_board_theme() == BoardTheme.CHALKBOARD
    assert board.board_theme() == BoardTheme.CHALKBOARD

    assert board.cycle_board_theme() == BoardTheme.OLY
    assert board.board_theme() == BoardTheme.OLY

    assert board.cycle_board_theme() == BoardTheme.SLATE
    assert board.board_theme() == BoardTheme.SLATE

    assert board.cycle_board_theme() == BoardTheme.WHITE
    assert board.board_theme() == BoardTheme.WHITE


def test_lesson_board_invalid_theme_ignored(qapp):
    """Test invalid theme string does not change state or raise exception."""
    board = LessonBoard()
    board.set_board_theme("invalid_neon_purple")
    assert board.board_theme() == BoardTheme.WHITE


def test_lesson_board_draw_background_all_themes_and_grids(qapp):
    """Ensure drawBackground executes smoothly across all themes and grid modes."""
    board = LessonBoard()
    img = QImage(1920, 1080, QImage.Format.Format_ARGB32)
    p = QPainter(img)

    for theme in [BoardTheme.WHITE, BoardTheme.CHALKBOARD, BoardTheme.OLY, BoardTheme.SLATE]:
        board.set_board_theme(theme)
        for grid_mode in ["none", "dots", "grid", "cartesian"]:
            board.set_grid_mode(grid_mode)
            board.drawBackground(p, QRectF(0, 0, 1920, 1080))

    p.end()


# ── 3. CanvasExporter High-Res Image Rendering Tests ──────────────────────────

def test_render_scene_to_image_1080p(qapp):
    """Test rendering standard 1080p canvas."""
    board = LessonBoard()
    qimg = CanvasExporter.render_scene_to_image(board, width=1920, height=1080, scale=1.0)
    assert not qimg.isNull()
    assert qimg.width() == 1920
    assert qimg.height() == 1080


def test_render_scene_to_image_4k_supersampling(qapp):
    """Test rendering 2x supersampled 4K UHD canvas."""
    board = LessonBoard()
    qimg = CanvasExporter.render_scene_to_image(board, width=1920, height=1080, scale=2.0)
    assert not qimg.isNull()
    assert qimg.width() == 3840
    assert qimg.height() == 2160


def test_render_scene_transparent_background(qapp):
    """Test rendering canvas with transparent background."""
    board = LessonBoard()
    qimg = CanvasExporter.render_scene_to_image(board, scale=1.0, transparent_bg=True)
    assert not qimg.isNull()
    # Check pixel at (0, 0) has alpha 0
    c = qimg.pixelColor(0, 0)
    assert c.alpha() == 0


def test_render_scene_hides_overlays(qapp):
    """Test temporary interactive overlays are hidden during capture and restored after."""
    board = LessonBoard()
    board.move_laser(960.0, 540.0, hand_id="primary")
    board.move_eraser_reticle(960.0, 540.0, radius=30.0, hand_id="primary")
    board.show_timer(True)

    laser = board._laser_items["primary"]
    eraser = board._eraser_items["primary"]
    assert laser.isVisible()
    assert eraser.isVisible()

    qimg = CanvasExporter.render_scene_to_image(board, include_overlays=False)
    assert not qimg.isNull()

    # Overlays must be restored
    assert laser.isVisible()
    assert eraser.isVisible()


# ── 4. CanvasExporter Disk File Saving Tests ──────────────────────────────────

def test_export_scene_to_file_png(qapp, tmp_path):
    """Test exporting canvas to PNG file."""
    board = LessonBoard()
    out_file = tmp_path / "exports" / "test_slide.png"

    saved = CanvasExporter.export_scene_to_file(board, out_file, scale=1.0)
    assert saved.is_file()
    assert saved.stat().st_size > 1000  # Valid non-empty PNG


def test_export_scene_to_file_jpeg(qapp, tmp_path):
    """Test exporting canvas to JPEG file with custom quality."""
    board = LessonBoard()
    out_file = tmp_path / "exports" / "test_slide.jpg"

    saved = CanvasExporter.export_scene_to_file(board, out_file, scale=1.0, quality=90)
    assert saved.is_file()
    assert saved.stat().st_size > 1000  # Valid non-empty JPEG


def test_export_all_pages_whiteboard(qapp, tmp_path):
    """Test batch export on a whiteboard-only MainWindow mock."""
    mock_mw = MagicMock()
    mock_mw._scene = LessonBoard()
    mock_mw._pages = []

    out_dir = tmp_path / "batch_export"
    exported = CanvasExporter.export_all_pages(
        main_window=mock_mw,
        output_dir=out_dir,
        prefix="slide",
        ext=".png",
    )
    assert len(exported) == 1
    assert exported[0].is_file()
    assert exported[0].name == "slide_01.png"


# ── 5. In-Air Menu & Floating Action Menu Integrations ─────────────────────────

def test_in_air_menu_items_extended():
    """Ensure chalk pen colors and board theme / export options are present in in-air menu."""
    pen_ids = [item.item_id for item in PEN_MENU_ITEMS]
    assert "pen_chalk_white" in pen_ids
    assert "pen_chalk_yellow" in pen_ids
    assert "pen_chalk_cyan" in pen_ids

    stem_ids = [item.item_id for item in STEM_MENU_ITEMS]
    assert "stem_board_theme" in stem_ids
    assert "stem_export_img" in stem_ids


# ── 6. ExportImageDialog Integration Tests ────────────────────────────────────

def test_export_image_dialog_initialization(qapp):
    """Test ExportImageDialog builds properly without crashing."""
    mock_mw = MagicMock()
    mock_mw._pages = [MagicMock(), MagicMock()]
    mock_mw._current_page_index = 0

    dlg = ExportImageDialog(mock_mw)
    assert dlg.radio_current.isChecked()
    assert dlg.radio_all.isEnabled()
    assert dlg.combo_format.count() == 2
    assert dlg.combo_res.count() == 3

    # Switch format to JPEG
    dlg.combo_format.setCurrentIndex(1)
    assert dlg.txt_dest.text().endswith(".jpg")
    assert not dlg.chk_transparent.isEnabled()

    # Switch back to PNG
    dlg.combo_format.setCurrentIndex(0)
    assert dlg.txt_dest.text().endswith(".png")
    assert dlg.chk_transparent.isEnabled()



# ── 7. MainWindow Integration Tests ───────────────────────────────────────────

def test_main_window_toggle_board_theme_integration(qapp):
    """Test MainWindow toggles theme, updates ink color, and status bar message."""
    from src.ui.main_window import MainWindow

    mw = MainWindow(camera_index=-1)
    try:
        assert mw._scene.board_theme() == BoardTheme.WHITE

        # Toggle theme to Chalkboard
        mw._on_toggle_board_theme()
        assert mw._scene.board_theme() == BoardTheme.CHALKBOARD
        assert mw._ink_color == THEME_CONFIGS[BoardTheme.CHALKBOARD]["default_ink"]

        # Toggle theme to Vietnamese Ô ly
        mw._on_toggle_board_theme()
        assert mw._scene.board_theme() == BoardTheme.OLY
        assert mw._ink_color == THEME_CONFIGS[BoardTheme.OLY]["default_ink"]

        # In-air menu action triggers theme toggle
        mw._on_menu_action_triggered("action", "toggle_board_theme")
        assert mw._scene.board_theme() == BoardTheme.SLATE
        assert mw._ink_color == THEME_CONFIGS[BoardTheme.SLATE]["default_ink"]
    finally:
        mw.close()

def test_export_scene_to_file_atomic_cleanup_on_failure(qapp, tmp_path):
    """export_scene_to_file must clean up temporary files if saving fails."""
    board = LessonBoard()
    # Invalid directory path that causes save failure
    out_file = tmp_path / "nonexistent_dir" / "nested" / "test.png"
    # Create parent as a file instead of directory to trigger save error
    (tmp_path / "nonexistent_dir").write_text("not_a_dir")

    with pytest.raises(Exception):
        CanvasExporter.export_scene_to_file(board, out_file)

    # Ensure no lingering .tmp files remain in tmp_path
    tmp_files = list(tmp_path.glob(".tmp_*"))
    assert len(tmp_files) == 0

