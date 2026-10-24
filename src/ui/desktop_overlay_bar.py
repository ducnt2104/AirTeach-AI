"""
AirTeach AI - Floating Desktop Overlay Toolbar
==============================================
A sleek, draggable acrylic control bar for Transparent Desktop Annotation Mode.
Allows teachers to toggle in-air tools (Pen, Eraser, Laser, Spotlight, Loupe, Timer)
and return seamlessly to the Whiteboard without cluttering external desktop apps.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

logger = logging.getLogger(__name__)


class DesktopOverlayBar(QFrame):
    """
    Floating acrylic HUD toolbar displayed during Desktop Transparent Annotation Mode.
    Stays on top of all windows and allows rapid air gesture tool switching.
    """

    pen_requested = Signal()
    eraser_requested = Signal()
    laser_requested = Signal()
    spotlight_requested = Signal()
    loupe_requested = Signal()
    timer_requested = Signal()
    clear_requested = Signal()
    undo_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._drag_start: Optional[QPoint] = None
        self._active_tool = "pen"

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(8)

        # 1. Drag Handle / Status Badge
        handle = QLabel("⠿ AirTeach")
        handle.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        handle.setStyleSheet("color: #9aa0a6; padding-right: 4px;")
        layout.addWidget(handle)

        # Separator
        layout.addWidget(self._make_separator())

        # 2. Tool buttons
        self.btn_pen = self._make_tool_btn("✏️ Bút", "pen")
        self.btn_pen.clicked.connect(self._on_pen_clicked)
        layout.addWidget(self.btn_pen)

        self.btn_eraser = self._make_tool_btn("🧹 Tẩy", "eraser")
        self.btn_eraser.clicked.connect(self._on_eraser_clicked)
        layout.addWidget(self.btn_eraser)

        self.btn_laser = self._make_tool_btn("🔴 Laser", "laser")
        self.btn_laser.clicked.connect(self._on_laser_clicked)
        layout.addWidget(self.btn_laser)

        self.btn_spotlight = self._make_tool_btn("🔦 Đèn rọi", "spotlight")
        self.btn_spotlight.clicked.connect(self._on_spotlight_clicked)
        layout.addWidget(self.btn_spotlight)

        self.btn_loupe = self._make_tool_btn("🔍 Kính lúp", "loupe")
        self.btn_loupe.clicked.connect(self._on_loupe_clicked)
        layout.addWidget(self.btn_loupe)

        self.btn_timer = self._make_tool_btn("⏱️ Hẹn giờ", "timer")
        self.btn_timer.clicked.connect(self._on_timer_clicked)
        layout.addWidget(self.btn_timer)

        layout.addWidget(self._make_separator())

        # 3. Actions (Undo & Clear)
        btn_undo = QPushButton("↩️")
        btn_undo.setToolTip("Hoàn tác nét vẽ vừa tạo (Ctrl+Z)")
        btn_undo.setStyleSheet(self._btn_style(active=False))
        btn_undo.clicked.connect(self.undo_requested.emit)
        layout.addWidget(btn_undo)

        btn_clear = QPushButton("🗑️")
        btn_clear.setToolTip("Xóa toàn bộ nét vẽ màn hình Desktop")
        btn_clear.setStyleSheet(self._btn_style(active=False))
        btn_clear.clicked.connect(self.clear_requested.emit)
        layout.addWidget(btn_clear)

        layout.addWidget(self._make_separator())

        # 4. Exit / Return to Whiteboard
        btn_exit = QPushButton("🔙 Bảng dạy học")
        btn_exit.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        btn_exit.setToolTip("Thoát chế độ Desktop và quay lại Bảng tương tác (F11)")
        btn_exit.setStyleSheet("""
            QPushButton {
                background: #1a73e8;
                color: #ffffff;
                border: none;
                border-radius: 14px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background: #1557b0;
            }
        """)
        btn_exit.clicked.connect(self.exit_requested.emit)
        layout.addWidget(btn_exit)

        self._update_tool_buttons()

    def _make_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: rgba(255, 255, 255, 0.2);")
        sep.setFixedWidth(2)
        return sep

    def _make_tool_btn(self, label: str, tool_name: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        btn.setProperty("tool_name", tool_name)
        return btn

    def _btn_style(self, active: bool = False) -> str:
        if active:
            return """
                QPushButton {
                    background: rgba(26, 115, 232, 0.85);
                    color: #ffffff;
                    border: 1px solid #4285f4;
                    border-radius: 12px;
                    padding: 4px 10px;
                }
            """
        else:
            return """
                QPushButton {
                    background: rgba(255, 255, 255, 0.10);
                    color: #e8eaed;
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 12px;
                    padding: 4px 10px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.22);
                    color: #ffffff;
                }
            """

    def set_active_tool(self, tool_name: str) -> None:
        """Update active tool indicator on the bar."""
        self._active_tool = tool_name
        self._update_tool_buttons()

    def _update_tool_buttons(self) -> None:
        self.btn_pen.setStyleSheet(self._btn_style(self._active_tool == "pen"))
        self.btn_eraser.setStyleSheet(self._btn_style(self._active_tool == "eraser"))
        self.btn_laser.setStyleSheet(self._btn_style(self._active_tool == "laser"))
        self.btn_spotlight.setStyleSheet(self._btn_style(self._active_tool == "spotlight"))
        self.btn_loupe.setStyleSheet(self._btn_style(self._active_tool == "loupe"))
        self.btn_timer.setStyleSheet(self._btn_style(self._active_tool == "timer"))

    def _on_pen_clicked(self) -> None:
        self.set_active_tool("pen")
        self.pen_requested.emit()

    def _on_eraser_clicked(self) -> None:
        self.set_active_tool("eraser")
        self.eraser_requested.emit()

    def _on_laser_clicked(self) -> None:
        self.set_active_tool("laser")
        self.laser_requested.emit()

    def _on_spotlight_clicked(self) -> None:
        self.set_active_tool("spotlight")
        self.spotlight_requested.emit()

    def _on_loupe_clicked(self) -> None:
        self.set_active_tool("loupe")
        self.loupe_requested.emit()

    def _on_timer_clicked(self) -> None:
        self.timer_requested.emit()

    # ── Paint Acrylic Background ──────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Rounded acrylic card background with subtle glow border
        painter.setBrush(QColor(28, 30, 36, 235))
        painter.setPen(QColor(255, 255, 255, 38))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 18.0, 18.0)
        super().paintEvent(event)

    # ── Draggable Window Support ──────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_start is not None:
            self.move(event.globalPosition().toPoint() - self._drag_start)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_start = None
        super().mouseReleaseEvent(event)
