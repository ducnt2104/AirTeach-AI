"""
Slide Thumbnail Drawer & Carousel Panel for AirTeach AI.

Provides visual thumbnail overview of all document slides and whiteboard pages,
instant random-access jumping between slides, slide addition, duplication,
and deletion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from src.services.document_manager import PageInfo
    from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class SlideThumbnailDrawer(QWidget):
    """
    Visual thumbnail navigation drawer displaying all pages of the active presentation.

    Signals:
        page_selected(int): Emitted when teacher clicks or selects a slide thumbnail.
        add_blank_page_requested(): Emitted when teacher requests a new blank whiteboard slide.
        delete_page_requested(): Emitted when teacher requests deleting the current slide.
        duplicate_page_requested(): Emitted when teacher requests duplicating the active slide.
    """

    page_selected            = Signal(int)
    add_blank_page_requested = Signal()
    delete_page_requested    = Signal()
    duplicate_page_requested = Signal()

    THUMB_WIDTH  = 160
    THUMB_HEIGHT = 90

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(210)
        self.setMaximumWidth(280)
        self._current_index = 0
        self._pages_cache: list[PageInfo] = []
        self._thumb_pixmaps: dict[int, QPixmap] = {}
        self._build_ui()
        self.refresh_thumbnails()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Top Toolbar Header ──
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(6)

        title_lbl = QLabel("📑 Slides Bài Giảng")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #cdd6f4;")
        header_layout.addWidget(title_lbl, stretch=1)

        # Action Buttons
        self.btn_add = QToolButton()
        self.btn_add.setText("➕")
        self.btn_add.setToolTip("Thêm trang trắng mới (Whiteboard Slide)")
        self.btn_add.clicked.connect(self.add_blank_page_requested.emit)
        self._style_tool_btn(self.btn_add)
        header_layout.addWidget(self.btn_add)

        self.btn_dup = QToolButton()
        self.btn_dup.setText("📋")
        self.btn_dup.setToolTip("Nhân bản trang hiện tại")
        self.btn_dup.clicked.connect(self.duplicate_page_requested.emit)
        self._style_tool_btn(self.btn_dup)
        header_layout.addWidget(self.btn_dup)

        self.btn_del = QToolButton()
        self.btn_del.setText("🗑️")
        self.btn_del.setToolTip("Xóa trang hiện tại")
        self.btn_del.clicked.connect(self.delete_page_requested.emit)
        self._style_tool_btn(self.btn_del)
        header_layout.addWidget(self.btn_del)

        layout.addLayout(header_layout)

        # ── Thumbnails List ──
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(self.THUMB_WIDTH, self.THUMB_HEIGHT))
        self.list_widget.setGridSize(QSize(self.THUMB_WIDTH + 24, self.THUMB_HEIGHT + 36))
        self.list_widget.setMovement(QListWidget.Movement.Static)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        self.list_widget.setStyleSheet("""
            QListWidget {
                background: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 4px;
            }
            QListWidget::item {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 2px solid #313244;
                border-radius: 6px;
                padding: 4px;
                margin: 4px;
                font-size: 11px;
                font-weight: 500;
            }
            QListWidget::item:hover {
                background: #313244;
                border-color: #89b4fa;
            }
            QListWidget::item:selected {
                background: #313244;
                color: #ffffff;
                border: 2px solid #89b4fa;
                font-weight: bold;
            }
        """)

        self.list_widget.currentRowChanged.connect(self._on_item_selected)
        layout.addWidget(self.list_widget, stretch=1)

        # Status summary
        self.lbl_info = QLabel("Chưa mở tài liệu (Bảng trắng)")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
        layout.addWidget(self.lbl_info)

    def _style_tool_btn(self, btn: QToolButton) -> None:
        btn.setStyleSheet("""
            QToolButton {
                background: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
                font-size: 12px;
            }
            QToolButton:hover {
                background: #45475a;
                color: #ffffff;
                border-color: #89b4fa;
            }
            QToolButton:pressed {
                background: #89b4fa;
                color: #11111b;
            }
        """)

    @Slot(int)
    def _on_item_selected(self, row: int) -> None:
        if row >= 0 and row != self._current_index:
            self._current_index = row
            self.page_selected.emit(row)

    def set_pages(self, pages: list[PageInfo], current_index: int = 0) -> None:
        """Populate the thumbnail drawer with document pages."""
        self._pages_cache = list(pages)
        self._current_index = max(0, min(current_index, len(pages) - 1)) if pages else 0
        self.refresh_thumbnails()

    def set_current_page(self, index: int) -> None:
        """Highlight active slide thumbnail without triggering selection recursion."""
        if not self._pages_cache:
            return
        idx = max(0, min(index, len(self._pages_cache) - 1))
        self._current_index = idx
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(idx)
        self.list_widget.blockSignals(False)
        self.lbl_info.setText(f"Trang {idx + 1} / {len(self._pages_cache)}")

    def refresh_thumbnails(self) -> None:
        """Re-render thumbnail icons and refresh items in the list."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        total = len(self._pages_cache)
        if total == 0:
            self.lbl_info.setText("Bảng vẽ đơn lẻ (0 trang tài liệu)")
            self.btn_del.setEnabled(False)
            self.btn_dup.setEnabled(False)
            self.list_widget.blockSignals(False)
            return

        self.btn_del.setEnabled(total > 1)
        self.btn_dup.setEnabled(True)

        for i, page in enumerate(self._pages_cache):
            pix = self._get_thumbnail_pixmap(i, page)
            item = QListWidgetItem(QIcon(pix), f"Trang {i + 1}")
            item.setSizeHint(QSize(self.THUMB_WIDTH + 16, self.THUMB_HEIGHT + 28))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
            self.list_widget.addItem(item)

        if 0 <= self._current_index < total:
            self.list_widget.setCurrentRow(self._current_index)
            self.lbl_info.setText(f"Trang {self._current_index + 1} / {total}")

        self.list_widget.blockSignals(False)

    def _get_thumbnail_pixmap(self, idx: int, page: PageInfo) -> QPixmap:
        """Load or create a scaled thumbnail pixmap for a page."""
        if idx in self._thumb_pixmaps:
            return self._thumb_pixmaps[idx]

        img_p = Path(page.image_path)
        if img_p.is_file():
            raw = QPixmap(str(img_p))
            if not raw.isNull():
                thumb = raw.scaled(
                    self.THUMB_WIDTH,
                    self.THUMB_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
                # Crop to exact thumb bounds
                cropped = thumb.copy(
                    max(0, (thumb.width() - self.THUMB_WIDTH) // 2),
                    max(0, (thumb.height() - self.THUMB_HEIGHT) // 2),
                    self.THUMB_WIDTH,
                    self.THUMB_HEIGHT,
                )
                self._thumb_pixmaps[idx] = cropped
                return cropped

        # Fallback placeholder thumbnail
        fallback = QPixmap(self.THUMB_WIDTH, self.THUMB_HEIGHT)
        fallback.fill(QColor("#24273a"))
        p = QPainter(fallback)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("#585b70"))
        p.drawRect(0, 0, self.THUMB_WIDTH - 1, self.THUMB_HEIGHT - 1)
        p.setPen(QColor("#a6adc8"))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRectF(0, 0, self.THUMB_WIDTH, self.THUMB_HEIGHT), Qt.AlignmentFlag.AlignCenter, f"Trang {idx + 1}")
        p.end()

        self._thumb_pixmaps[idx] = fallback
        return fallback

    def invalidate_cache(self) -> None:
        """Clear cached thumbnail pixmaps so they re-render on next refresh."""
        self._thumb_pixmaps.clear()
        self.refresh_thumbnails()
