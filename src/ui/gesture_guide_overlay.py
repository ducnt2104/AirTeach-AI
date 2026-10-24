"""
AirTeach AI - Interactive In-Air Gesture Cheat-Sheet & HUD Guide
================================================================
A draggable acrylic card providing a real-time visual guide for touchless
hand gestures (Air Pen, Laser, Eraser, Pinch Click, Fist Pause, Page Swipe,
and Two-Hand Bimanual Interaction) with active state highlights.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.hand_gesture_engine import GestureState

logger = logging.getLogger(__name__)


class GestureCard(QFrame):
    """A visual card representing a single recognized hand gesture."""

    def __init__(
        self,
        icon: str,
        name: str,
        desc: str,
        gesture_enum: Optional[GestureState] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.gesture_enum = gesture_enum
        self._is_active = False

        self.setFixedSize(175, 78)
        self.setStyleSheet(self._card_style(active=False))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        header_h = QHBoxLayout()
        header_h.setSpacing(6)

        icon_lbl = QLabel(icon)
        icon_lbl.setFont(QFont("Segoe UI Emoji", 14))
        header_h.addWidget(icon_lbl)

        name_lbl = QLabel(name)
        name_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        name_lbl.setStyleSheet("color: #ffffff;")
        header_h.addWidget(name_lbl, stretch=1)
        layout.addLayout(header_h)

        desc_lbl = QLabel(desc)
        desc_lbl.setFont(QFont("Segoe UI", 8))
        desc_lbl.setStyleSheet("color: #bdc1c6;")
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)

    def set_active(self, active: bool) -> None:
        if self._is_active != active:
            self._is_active = active
            self.setStyleSheet(self._card_style(active=active))

    def _card_style(self, active: bool) -> str:
        if active:
            return """
                QFrame {
                    background: rgba(26, 115, 232, 0.40);
                    border: 1.5px solid #00ff88;
                    border-radius: 10px;
                }
            """
        else:
            return """
                QFrame {
                    background: rgba(255, 255, 255, 0.06);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 10px;
                }
            """


class GestureGuideOverlay(QFrame):
    """
    Floating acrylic HUD card showing quick reference for touchless air gestures.
    Lights up corresponding gesture cards in real time as hands are recognized.
    """

    closed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._drag_start: Optional[QPoint] = None

        self._cards: dict[str, GestureCard] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root_v = QVBoxLayout(self)
        root_v.setContentsMargins(14, 10, 14, 12)
        root_v.setSpacing(10)

        # Header with drag handle and close button
        header_h = QHBoxLayout()
        title = QLabel("✋ Cử chỉ Không chạm (Air Gestures)")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff;")
        header_h.addWidget(title)

        header_h.addStretch(1)

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(22, 22)
        btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.12);
                color: #e8eaed;
                border: none;
                border-radius: 11px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: rgba(234, 67, 53, 0.7);
                color: #ffffff;
            }
        """)
        btn_close.clicked.connect(self.close_guide)
        header_h.addWidget(btn_close)
        root_v.addLayout(header_h)

        # 2x3 Grid of gesture cards
        grid = QGridLayout()
        grid.setSpacing(8)

        # Card definitions
        card_data = [
            ("DRAW", "✌️", "Air Pen", "Giơ 2 ngón (Trỏ + Giữa) để vẽ nét hoặc nắn hình", GestureState.DRAW, 0, 0),
            ("POINT", "☝️", "Laser / Trỏ", "Giơ 1 ngón trỏ để trỏ bảng, dẫn Đèn rọi / Kính lúp", GestureState.POINT, 0, 1),
            ("SELECT", "✋", "Tẩy / Menu", "Xòe cả bàn tay 5 ngón để tẩy xóa hoặc mở Menu", GestureState.SELECT, 1, 0),
            ("PINCH", "👌", "Pinch / Tap", "Chụm ngón cái & trỏ để chọn công cụ / Air Tap", GestureState.NONE, 1, 1),
            ("PAUSE", "✊", "Tạm dừng", "Nắm chặt bàn tay để ngắt nét vẽ hoặc kéo thả", GestureState.PAUSE, 2, 0),
            ("SWIPE", "👈👉", "Lật trang", "Vung tay nhanh sang trái / phải để chuyển trang", GestureState.SWIPE_RIGHT, 2, 1),
        ]

        for key, icon, name, desc, state, row, col in card_data:
            c = GestureCard(icon, name, desc, gesture_enum=state)
            self._cards[key] = c
            grid.addWidget(c, row, col)

        root_v.addLayout(grid)

        # Bimanual footer tip
        tip_lbl = QLabel("💡 <b>Tương tác 2 tay:</b> Tay thuận vẽ/trỏ, tay phụ mở menu hoặc chỉnh thấu kính.")
        tip_lbl.setFont(QFont("Segoe UI", 8))
        tip_lbl.setStyleSheet("color: #9aa0a6; padding-top: 2px;")
        root_v.addWidget(tip_lbl)

    def update_active_gesture(self, gesture: GestureState, is_pinched: bool = False) -> None:
        """Light up the matching gesture card based on current recognition state."""
        for key, card in self._cards.items():
            active = False
            if is_pinched and key == "PINCH":
                active = True
            elif key == "DRAW" and gesture == GestureState.DRAW and not is_pinched:
                active = True
            elif key == "POINT" and gesture == GestureState.POINT and not is_pinched:
                active = True
            elif key == "SELECT" and gesture == GestureState.SELECT:
                active = True
            elif key == "PAUSE" and gesture == GestureState.PAUSE:
                active = True
            elif key == "SWIPE" and gesture in (GestureState.SWIPE_LEFT, GestureState.SWIPE_RIGHT):
                active = True
            card.set_active(active)

    def close_guide(self) -> None:
        """Hide guide and emit signal."""
        self.hide()
        self.closed.emit()

    def toggle_guide(self) -> bool:
        """Toggle guide visibility; returns new state."""
        if self.isVisible():
            self.close_guide()
            return False
        else:
            self.show()
            self.raise_()
            return True

    # ── Paint Acrylic Background ──────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Dark acrylic card container
        painter.setBrush(QColor(24, 26, 32, 240))
        painter.setPen(QColor(255, 255, 255, 45))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16.0, 16.0)
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
