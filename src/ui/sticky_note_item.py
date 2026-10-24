"""
Acrylic Floating Sticky Note Item for AirTeach AI.

Provides draggable, collapsible, multi-theme acrylic sticky notes
on the whiteboard canvas with rich text notes, reminders, and hints.
"""

from __future__ import annotations

import uuid
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


STICKY_THEMES = {
    "yellow": {
        "name": "Vàng tươi (Classic Yellow)",
        "bg": QColor(255, 249, 196, 240),
        "header": QColor(255, 245, 157, 255),
        "border": QColor(251, 192, 45, 200),
        "text": QColor("#263238"),
        "shadow": QColor(0, 0, 0, 45),
    },
    "green": {
        "name": "Xanh bạc hà (Mint Green)",
        "bg": QColor(232, 245, 233, 240),
        "header": QColor(200, 230, 201, 255),
        "border": QColor(129, 199, 132, 200),
        "text": QColor("#1b5e20"),
        "shadow": QColor(0, 0, 0, 45),
    },
    "blue": {
        "name": "Xanh mây (Sky Blue)",
        "bg": QColor(227, 242, 253, 240),
        "header": QColor(187, 222, 251, 255),
        "border": QColor(100, 181, 246, 200),
        "text": QColor("#0d47a1"),
        "shadow": QColor(0, 0, 0, 45),
    },
    "coral": {
        "name": "Cam đào (Peach Coral)",
        "bg": QColor(251, 233, 231, 240),
        "header": QColor(255, 204, 188, 255),
        "border": QColor(255, 138, 101, 200),
        "text": QColor("#bf360c"),
        "shadow": QColor(0, 0, 0, 45),
    },
    "lavender": {
        "name": "Tím oải hương (Lavender)",
        "bg": QColor(237, 231, 246, 240),
        "header": QColor(209, 196, 233, 255),
        "border": QColor(186, 104, 200, 200),
        "text": QColor("#4a148c"),
        "shadow": QColor(0, 0, 0, 45),
    },
}


class StickyNoteEditDialog(QDialog):
    """Dialog for editing Sticky Note title, content, and color theme."""

    def __init__(self, title: str, text: str, theme: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chỉnh sửa Ghi chú (Sticky Note)")
        self.setMinimumWidth(380)
        self.setStyleSheet("""
            QDialog {
                background: #1e1e2e;
                color: #cdd6f4;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 13px;
                font-weight: bold;
            }
            QLineEdit, QPlainTextEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px;
                color: #ffffff;
                font-size: 13px;
            }
            QPushButton {
                background: #313244;
                color: #ffffff;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background: #45475a;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Tiêu đề:"))
        self.txt_title = QLineEdit(title)
        layout.addWidget(self.txt_title)

        layout.addWidget(QLabel("Nội dung ghi chú:"))
        self.txt_body = QPlainTextEdit(text)
        self.txt_body.setMinimumHeight(120)
        layout.addWidget(self.txt_body)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_data(self) -> tuple[str, str]:
        return self.txt_title.text().strip(), self.txt_body.toPlainText().strip()


class StickyNoteItem(QGraphicsObject):
    """
    Acrylic movable, collapsible sticky note card on the whiteboard canvas.

    Signals:
        note_changed(): Emitted when position, text, or theme changes.
        note_deleted(str): Emitted when user clicks close (✕) button (note_id).
    """

    note_changed = Signal()
    note_deleted = Signal(str)

    EXPANDED_WIDTH   = 240.0
    EXPANDED_HEIGHT  = 160.0
    COLLAPSED_WIDTH  = 160.0
    COLLAPSED_HEIGHT = 36.0
    HEADER_HEIGHT    = 32.0

    def __init__(
        self,
        title: str = "Ghi chú",
        text: str = "",
        theme: str = "yellow",
        x: float = 100.0,
        y: float = 100.0,
        is_collapsed: bool = False,
        note_id: Optional[str] = None,
        parent: Optional[QGraphicsObject] = None,
    ) -> None:
        super().__init__(parent)
        self.note_id = note_id or str(uuid.uuid4())[:8]
        self.title = title or "Ghi chú"
        self.text = text
        self.theme = theme if theme in STICKY_THEMES else "yellow"
        self.is_collapsed = bool(is_collapsed)

        self.setPos(x, y)
        self.setZValue(35)  # Above ink and shapes, below floating action menu
        self.setFlags(
            QGraphicsObject.GraphicsItemFlag.ItemIsMovable |
            QGraphicsObject.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsObject.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    def boundingRect(self) -> QRectF:
        w = self.COLLAPSED_WIDTH if self.is_collapsed else self.EXPANDED_WIDTH
        h = self.COLLAPSED_HEIGHT if self.is_collapsed else self.EXPANDED_HEIGHT
        # Extra 6px padding for drop shadow
        return QRectF(-3.0, -3.0, w + 6.0, h + 6.0)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        w = self.COLLAPSED_WIDTH if self.is_collapsed else self.EXPANDED_WIDTH
        h = self.COLLAPSED_HEIGHT if self.is_collapsed else self.EXPANDED_HEIGHT
        path.addRoundedRect(QRectF(0, 0, w, h), 8.0, 8.0)
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        cfg = STICKY_THEMES.get(self.theme, STICKY_THEMES["yellow"])
        w = self.COLLAPSED_WIDTH if self.is_collapsed else self.EXPANDED_WIDTH
        h = self.COLLAPSED_HEIGHT if self.is_collapsed else self.EXPANDED_HEIGHT

        # 1. Subtle drop shadow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(cfg["shadow"]))
        painter.drawRoundedRect(QRectF(2.0, 3.0, w, h), 8.0, 8.0)

        # 2. Main Note Body
        painter.setPen(QPen(cfg["border"], 1.5))
        painter.setBrush(QBrush(cfg["bg"]))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 8.0, 8.0)

        # 3. Header Ribbon
        header_path = QPainterPath()
        header_path.addRoundedRect(QRectF(0, 0, w, min(h, self.HEADER_HEIGHT)), 8.0, 8.0)
        if not self.is_collapsed:
            # Flatten bottom corners of header
            bottom_cover = QPainterPath()
            bottom_cover.addRect(QRectF(0, self.HEADER_HEIGHT - 6.0, w, 6.0))
            header_path = header_path.united(bottom_cover)

        painter.setBrush(QBrush(cfg["header"]))
        painter.setPen(QPen(cfg["border"], 1.0))
        painter.drawPath(header_path)

        # 4. Pin icon & Title
        painter.setPen(QPen(cfg["text"]))
        painter.setFont(QFont("Segoe UI Emoji", 11))
        painter.drawText(QRectF(8.0, 4.0, 20.0, 24.0), Qt.AlignmentFlag.AlignCenter, "📌")

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title_rect = QRectF(30.0, 4.0, w - 90.0, 24.0)
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            painter.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideRight, int(title_rect.width()))
        )

        # 5. Header Action Buttons (Collapse, Theme, Delete)
        # Collapse toggle: (w - 56, 6, 16, 20)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        collapse_char = "▼" if self.is_collapsed else "▲"
        painter.drawText(QRectF(w - 58.0, 5.0, 16.0, 20.0), Qt.AlignmentFlag.AlignCenter, collapse_char)

        # Theme toggle: (w - 38, 6, 16, 20)
        painter.drawText(QRectF(w - 40.0, 5.0, 16.0, 20.0), Qt.AlignmentFlag.AlignCenter, "🎨")

        # Close button: (w - 18, 6, 16, 20)
        painter.drawText(QRectF(w - 20.0, 5.0, 16.0, 20.0), Qt.AlignmentFlag.AlignCenter, "✕")

        # 6. Body Text (only if expanded)
        if not self.is_collapsed:
            painter.setFont(QFont("Segoe UI", 10))
            body_rect = QRectF(10.0, self.HEADER_HEIGHT + 6.0, w - 20.0, h - self.HEADER_HEIGHT - 12.0)
            if self.text.strip():
                painter.drawText(
                    body_rect,
                    Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                    self.text
                )
            else:
                painter.setPen(QPen(QColor(cfg["text"].red(), cfg["text"].green(), cfg["text"].blue(), 120)))
                painter.drawText(
                    body_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    "(Nhấp đúp để thêm nội dung)"
                )

        painter.restore()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        w = self.COLLAPSED_WIDTH if self.is_collapsed else self.EXPANDED_WIDTH

        # Check Close Button (✕)
        if w - 24.0 <= pos.x() <= w - 4.0 and 2.0 <= pos.y() <= self.HEADER_HEIGHT:
            event.accept()
            self.note_deleted.emit(self.note_id)
            if self.scene() is not None:
                self.scene().removeItem(self)
            return

        # Check Theme Button (🎨)
        if w - 44.0 <= pos.x() < w - 24.0 and 2.0 <= pos.y() <= self.HEADER_HEIGHT:
            event.accept()
            self.cycle_theme()
            return

        # Check Collapse Button (▲ / ▼)
        if w - 64.0 <= pos.x() < w - 44.0 and 2.0 <= pos.y() <= self.HEADER_HEIGHT:
            event.accept()
            self.toggle_collapse()
            return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Double click opens editor dialog."""
        event.accept()
        dlg = StickyNoteEditDialog(self.title, self.text, self.theme)
        if dlg.exec():
            new_title, new_text = dlg.get_data()
            self.title = new_title or "Ghi chú"
            self.text = new_text
            self.prepareGeometryChange()
            self.update()
            self.note_changed.emit()

    def toggle_collapse(self) -> bool:
        """Toggle between collapsed pill badge and expanded card."""
        self.prepareGeometryChange()
        self.is_collapsed = not self.is_collapsed
        self.update()
        self.note_changed.emit()
        return self.is_collapsed

    def cycle_theme(self) -> str:
        """Cycle through pastel sticky themes."""
        themes = list(STICKY_THEMES.keys())
        idx = themes.index(self.theme) if self.theme in themes else 0
        self.theme = themes[(idx + 1) % len(themes)]
        self.update()
        self.note_changed.emit()
        return self.theme

    def to_dict(self) -> dict:
        """Serialize note properties to dictionary."""
        return {
            "note_id": self.note_id,
            "title": self.title,
            "text": self.text,
            "theme": self.theme,
            "x": float(self.x()),
            "y": float(self.y()),
            "is_collapsed": self.is_collapsed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StickyNoteItem:
        """Reconstruct StickyNoteItem from dictionary."""
        return cls(
            title=data.get("title", "Ghi chú"),
            text=data.get("text", ""),
            theme=data.get("theme", "yellow"),
            x=float(data.get("x", 100.0)),
            y=float(data.get("y", 100.0)),
            is_collapsed=bool(data.get("is_collapsed", False)),
            note_id=data.get("note_id"),
        )
