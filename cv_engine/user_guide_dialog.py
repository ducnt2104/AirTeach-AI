#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Built-in Interactive User Guide, 4-Step First-Run Onboarding Tour & Live Gesture Sandbox
(`cv_engine/user_guide_dialog.py`) for AirTeach AI (DNT - Victer).

Includes:
1. `OnboardingTourDialog`: Interactive 4-step first-launch walkthrough with "Không hiện lại" & "Bỏ qua".
2. `GestureSandboxWidget`: Live webcam testbed where teachers practice 1-hand & 2-hand gestures
   and receive immediate Green (✓ Đã nhận diện) / Red (✗ Chưa khớp) feedback.
3. `UserGuideDialog`: 5-Tab visual Cheat-Sheet (`?` button / `F1`):
   - Tab 1: Cử Chỉ 1 Tay
   - Tab 2: Cử Chỉ 2 Tay
   - Tab 3: Chuột, Phím Tắt & Bảng Tròn 8 Hướng
   - Tab 4: Tài Liệu Đa Định Dạng & AI Local Offline
   - Tab 5: 🧪 Thử Cử Chỉ Trực Tiếp (Gesture Sandbox)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cv_engine.radial_pie_menu import PIE_MENU_SLICES


BG_DARK = "#0F172A"
SURFACE_DARK = "#1E293B"
BORDER_SLATE = "#334155"
PRIMARY_INDIGO = "#4F46E5"
SECONDARY_TEAL = "#06B6D4"
NEON_CYAN = "#00F5D4"
BRIGHT_YELLOW = "#FEE440"
ALERT_ROSE = "#F43F5E"
SUCCESS_GREEN = "#10B981"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"


class GestureDiagramWidget(QWidget):
    """Crisp custom QPainter vector hand & UI schematic for each guide card."""

    def __init__(self, diagram_kind: str, accent_hex: str = NEON_CYAN, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.diagram_kind = diagram_kind
        self.accent = QColor(accent_hex)
        self.setFixedSize(92, 80)

    @staticmethod
    def _draw_hand(
        p: QPainter,
        cx: float,
        cy: float,
        scale: float,
        pose: str,
        accent: QColor,
        mirror: bool = False,
    ) -> None:
        """Render a clean articulated vector hand silhouette for 1-hand & 2-hand gestures."""
        p.save()
        p.translate(cx, cy)
        if mirror:
            p.scale(-scale, scale)
        else:
            p.scale(scale, scale)

        palm_fill = QColor(accent.red(), accent.green(), accent.blue(), 46)
        palm_pen = QPen(accent, 1.9, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        curled_pen = QPen(QColor(TEXT_MUTED), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        # Palm base (rounded trapezoid/box)
        p.setBrush(palm_fill)
        p.setPen(palm_pen)
        p.drawRoundedRect(QRectF(-12.0, -2.0, 24.0, 22.0), 6.0, 6.0)

        # Finger anchor X positions (index, middle, ring, pinky)
        fx = (-8.5, -3.0, 2.8, 8.2)
        flen = (20.0, 23.0, 20.5, 16.0)
        fspread = (-2.2, -0.5, 1.2, 3.0)

        if pose == "PINCH":
            # Thumb + Index form an unmistakable pinch ring on top-left
            p.setPen(palm_pen)
            p.setBrush(Qt.NoBrush)
            p.drawLine(QPointF(-12.0, 9.0), QPointF(-16.0, -3.0))
            p.drawLine(QPointF(-16.0, -3.0), QPointF(-9.5, -13.5))
            p.drawLine(QPointF(fx[0], -2.0), QPointF(-7.5, -13.5))
            # Glowing pinch contact dot
            p.setBrush(QColor(BRIGHT_YELLOW))
            p.setPen(QPen(QColor("#FFFFFF"), 1.0))
            p.drawEllipse(QPointF(-8.5, -13.8), 3.6, 3.6)
            # Curled remaining 3 fingers
            p.setBrush(QColor(30, 41, 59, 190))
            p.setPen(curled_pen)
            for i in (1, 2, 3):
                p.drawRoundedRect(QRectF(fx[i] - 2.3, -6.0, 4.6, 7.0), 2.2, 2.2)
        elif pose == "FIST":
            # All 4 fingers tightly folded across top of palm + thumb tucked in front
            p.setBrush(palm_fill)
            p.setPen(palm_pen)
            for i in range(4):
                p.drawRoundedRect(QRectF(fx[i] - 2.6, -7.5, 5.2, 8.0), 2.5, 2.5)
            p.drawRoundedRect(QRectF(-13.5, 1.5, 13.0, 6.0), 3.0, 3.0)
        else:
            # Decide extended fingers per pose
            if pose == "INDEX":
                ext = (True, False, False, False)
                thumb_out = False
            elif pose == "TWO_FINGERS":
                ext = (True, True, False, False)
                thumb_out = False
            elif pose == "THREE_FINGERS":
                ext = (True, True, True, False)
                thumb_out = False
            else:  # OPEN_PALM / SWIPE
                ext = (True, True, True, True)
                thumb_out = True

            # Thumb
            if thumb_out:
                p.setPen(QPen(accent, 3.6, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(-11.5, 10.0), QPointF(-20.0, 1.0))
            else:
                p.setBrush(QColor(30, 41, 59, 190))
                p.setPen(curled_pen)
                p.drawRoundedRect(QRectF(-12.5, 4.0, 9.5, 5.2), 2.5, 2.5)

            for i in range(4):
                if ext[i]:
                    spread = fspread[i] * (1.8 if pose == "TWO_FINGERS" and i in (0, 1) else 1.0)
                    tip = QPointF(fx[i] + spread, -2.0 - flen[i])
                    p.setPen(QPen(accent, 3.8, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(QPointF(fx[i], -2.0), tip)
                    p.setBrush(QColor("#FFFFFF"))
                    p.setPen(Qt.NoPen)
                    p.drawEllipse(tip, 1.6, 1.6)
                else:
                    p.setBrush(QColor(30, 41, 59, 190))
                    p.setPen(curled_pen)
                    p.drawRoundedRect(QRectF(fx[i] - 2.4, -6.5, 4.8, 7.2), 2.2, 2.2)

        p.restore()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(2.0, 2.0, self.width() - 4.0, self.height() - 4.0)
        p.setBrush(QColor(BG_DARK))
        p.setPen(QPen(QColor(BORDER_SLATE), 1.2))
        p.drawRoundedRect(r, 10.0, 10.0)

        cx = self.width() * 0.5
        cy = self.height() * 0.5
        k = self.diagram_kind

        if k == "POINTER":
            # 2-finger V-sign hand + subtle cursor halo
            self._draw_hand(p, cx - 4.0, cy + 12.0, 1.05, "TWO_FINGERS", self.accent)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 1.6))
            p.drawEllipse(QPointF(cx + 18.0, 18.0), 7.0, 7.0)
            p.setBrush(QColor(ALERT_ROSE))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx + 18.0, 18.0), 2.6, 2.6)
        elif k == "PINCH_DRAW":
            # 1-finger Index hand + smooth ink curve from fingertip
            self._draw_hand(p, 34.0, 52.0, 1.02, "INDEX", self.accent)
            p.setPen(QPen(self.accent, 2.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(24.0, 24.0), QPointF(46.0, 16.0))
            p.drawLine(QPointF(46.0, 16.0), QPointF(62.0, 32.0))
            p.drawLine(QPointF(62.0, 32.0), QPointF(78.0, 18.0))
        elif k == "PINCH_GRAB":
            # Pinch hand grabbing a floating dashed object card
            p.setBrush(QColor(250, 204, 21, 35))
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 1.6, Qt.DashLine))
            p.drawRoundedRect(QRectF(48.0, 14.0, 32.0, 26.0), 5.0, 5.0)
            self._draw_hand(p, 38.0, 52.0, 1.05, "PINCH", self.accent, mirror=True)
        elif k == "OPEN_ERASE":
            # 5-finger open palm + rose dashed eraser ring
            p.setBrush(QColor(244, 63, 94, 34))
            p.setPen(QPen(QColor(ALERT_ROSE), 1.6, Qt.DashLine))
            p.drawEllipse(QPointF(cx, cy), 28.0, 28.0)
            self._draw_hand(p, cx + 2.0, cy + 12.0, 0.98, "OPEN_PALM", self.accent)
        elif k == "SWIPE_H":
            # Open hand + horizontal swipe left/right arrows
            self._draw_hand(p, cx, cy + 14.0, 0.90, "OPEN_PALM", self.accent)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 2.2, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(12.0, 18.0), QPointF(80.0, 18.0))
            p.drawLine(QPointF(73.0, 13.0), QPointF(80.0, 18.0))
            p.drawLine(QPointF(73.0, 23.0), QPointF(80.0, 18.0))
            p.drawLine(QPointF(19.0, 13.0), QPointF(12.0, 18.0))
            p.drawLine(QPointF(19.0, 23.0), QPointF(12.0, 18.0))
        elif k == "DUAL_ZOOM":
            # Two pinching hands spreading apart + outward zoom arrows
            self._draw_hand(p, 24.0, 52.0, 0.80, "PINCH", self.accent, mirror=True)
            self._draw_hand(p, 68.0, 52.0, 0.80, "PINCH", self.accent, mirror=False)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 2.0, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(30.0, 20.0), QPointF(62.0, 20.0))
            p.drawLine(QPointF(36.0, 15.0), QPointF(30.0, 20.0))
            p.drawLine(QPointF(36.0, 25.0), QPointF(30.0, 20.0))
            p.drawLine(QPointF(56.0, 15.0), QPointF(62.0, 20.0))
            p.drawLine(QPointF(56.0, 25.0), QPointF(62.0, 20.0))
        elif k == "DUAL_PAN":
            # Two open hands gliding in parallel + 4-way pan arrows
            self._draw_hand(p, 26.0, 52.0, 0.74, "OPEN_PALM", self.accent, mirror=True)
            self._draw_hand(p, 66.0, 52.0, 0.74, "OPEN_PALM", self.accent, mirror=False)
            p.setPen(QPen(QColor(TEXT_LIGHT), 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx - 12.0, 18.0), QPointF(cx + 12.0, 18.0))
            p.drawLine(QPointF(cx, 10.0), QPointF(cx, 26.0))
        elif k == "MOD_LOCK":
            # Off-hand Fist + Primary hand drawing straight line
            self._draw_hand(p, 25.0, 52.0, 0.78, "FIST", QColor(BRIGHT_YELLOW), mirror=True)
            self._draw_hand(p, 67.0, 54.0, 0.78, "PINCH", self.accent, mirror=False)
            p.setPen(QPen(self.accent, 2.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(20.0, 24.0), QPointF(76.0, 14.0))
        elif k == "PIE_MENU":
            p.setPen(QPen(self.accent, 1.8))
            p.drawEllipse(QPointF(cx, cy), 27.0, 27.0)
            p.drawEllipse(QPointF(cx, cy), 9.0, 9.0)
            for i in range(8):
                ang = math.radians(i * 45.0 + 22.5)
                p.drawLine(
                    QPointF(cx + 9.0 * math.cos(ang), cy + 9.0 * math.sin(ang)),
                    QPointF(cx + 27.0 * math.cos(ang), cy + 27.0 * math.sin(ang)),
                )
        elif k == "MOUSE_NAV":
            p.setPen(QPen(self.accent, 2.0))
            p.drawRoundedRect(QRectF(28.0, 12.0, 36.0, 54.0), 14.0, 14.0)
            p.drawLine(QPointF(28.0, 34.0), QPointF(64.0, 34.0))
            p.setBrush(QColor(BRIGHT_YELLOW))
            p.drawRoundedRect(QRectF(43.0, 17.0, 6.0, 12.0), 3.0, 3.0)
        elif k == "SHAPES":
            # Off-hand 2-Finger V + crisp geometric circle/square snap badge
            self._draw_hand(p, 26.0, 54.0, 0.78, "TWO_FINGERS", self.accent, mirror=True)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(self.accent, 2.0))
            p.drawEllipse(QPointF(58.0, 28.0), 11.0, 11.0)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 1.8))
            p.drawRect(QRectF(56.0, 44.0, 20.0, 18.0))
        elif k == "UNDO":
            # 3-Finger hand + curved Undo arrow
            self._draw_hand(p, 34.0, 54.0, 0.86, "THREE_FINGERS", self.accent)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 2.2, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(QRectF(50.0, 14.0, 28.0, 24.0), -30 * 16, 200 * 16)
            p.drawLine(QPointF(50.0, 24.0), QPointF(50.0, 14.0))
            p.drawLine(QPointF(50.0, 24.0), QPointF(60.0, 24.0))
        elif k == "DOCS":
            p.setPen(QPen(self.accent, 2.0))
            p.drawRoundedRect(QRectF(20.0, 12.0, 36.0, 46.0), 5.0, 5.0)
            p.setBrush(QColor(30, 41, 59))
            p.drawRoundedRect(QRectF(34.0, 22.0, 38.0, 44.0), 5.0, 5.0)
            p.setPen(QPen(QColor(TEXT_LIGHT), 1.6))
            p.drawLine(QPointF(42.0, 34.0), QPointF(64.0, 34.0))
            p.drawLine(QPointF(42.0, 43.0), QPointF(64.0, 43.0))
            p.drawLine(QPointF(42.0, 52.0), QPointF(58.0, 52.0))
        else:  # AI_LOCAL
            p.setPen(QPen(self.accent, 2.0))
            p.drawRoundedRect(QRectF(20.0, 16.0, 52.0, 46.0), 10.0, 10.0)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 2.2))
            p.drawEllipse(QPointF(36.0, 36.0), 4.0, 4.0)
            p.drawEllipse(QPointF(56.0, 36.0), 4.0, 4.0)
            p.drawLine(QPointF(36.0, 50.0), QPointF(56.0, 50.0))


def _make_guide_card(
    diagram_kind: str,
    title: str,
    desc: str,
    accent_hex: str = NEON_CYAN,
) -> QFrame:
    card = QFrame()
    card.setStyleSheet(
        f"background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px;"
    )
    row = QHBoxLayout(card)
    row.setContentsMargins(12, 10, 12, 10)
    row.setSpacing(12)

    row.addWidget(GestureDiagramWidget(diagram_kind, accent_hex=accent_hex))

    col = QVBoxLayout()
    col.setSpacing(4)
    t_lbl = QLabel(title)
    t_lbl.setWordWrap(True)
    t_lbl.setStyleSheet(
        f"color: {accent_hex}; font-size: 13px; font-weight: 800; border: none; background: transparent;"
    )
    d_lbl = QLabel(desc)
    d_lbl.setWordWrap(True)
    d_lbl.setStyleSheet(
        f"color: {TEXT_LIGHT}; font-size: 12px; line-height: 1.45; border: none; background: transparent;"
    )
    col.addWidget(t_lbl)
    col.addWidget(d_lbl)
    col.addStretch(1)
    row.addLayout(col, stretch=1)
    return card


def _wrap_scroll(cards: List[QWidget]) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setStyleSheet("background: transparent; border: none;")
    container = QWidget()
    container.setStyleSheet("background: transparent;")
    vbox = QVBoxLayout(container)
    vbox.setContentsMargins(8, 8, 8, 8)
    vbox.setSpacing(10)
    for w in cards:
        vbox.addWidget(w)
    vbox.addStretch(1)
    area.setWidget(container)
    return area


class GestureSandboxWidget(QWidget):
    """
    Live Gesture Sandbox testbed (Tab 5 of UserGuideDialog):
    Receives real-time camera frames & recognized gestures from `VisionInferenceWorker`,
    turning each 1-hand / 2-hand gesture card GREEN (✓ Đạt) on live detection or showing
    RED feedback when palm orientation / gesture is not recognized.
    """

    SANDBOX_TARGETS: Tuple[Tuple[str, str, str], ...] = (
        ("INDEX", "☝️ 1 Ngón Trỏ", "Vẽ nét (Draw)"),
        ("TWO_FINGERS", "✌️ 2 Ngón (Chữ V)", "Con trỏ / Hover"),
        ("PINCH", "🤏 Chụm Cái + Trỏ", "Cầm & kéo Object"),
        ("OPEN_PALM", "🖐️ Xòe 5 Ngón", "Tẩy nét (Eraser)"),
        ("SWIPE", "↔ Vẫy Tay Ngang", "Chuyển Slide"),
        ("DUAL_PINCH_ZOOM", "🔍 2 Tay Chụm", "Phóng to / Thu nhỏ"),
        ("DUAL_PALM_PAN", "✋ 2 Tay Nắm / Xòe", "Di chuyển bảng / Cuộn"),
        ("THREE_FINGERS", "🤟 3 Ngón Tay", "Hoàn tác (Undo)"),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.passed_gestures: set[str] = set()
        self._cards: Dict[str, QLabel] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(12)

        self.cam_preview = QLabel("Đang chờ khung hình từ Webcam...\nHãy giơ 1 tay hoặc 2 tay trước camera để thử!")
        self.cam_preview.setFixedSize(310, 210)
        self.cam_preview.setAlignment(Qt.AlignCenter)
        self.cam_preview.setWordWrap(True)
        self.cam_preview.setStyleSheet(
            f"background: #020617; color: {TEXT_MUTED}; border: 2px solid {BORDER_SLATE}; border-radius: 10px; font-size: 11.5px; font-weight: 600;"
        )
        top.addWidget(self.cam_preview)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        intro = QLabel("🧪 PHÒNG THỬ CỬ CHỈ TRỰC TIẾP (GESTURE SANDBOX)")
        intro.setStyleSheet(f"color: {NEON_CYAN}; font-size: 13.5px; font-weight: 800;")
        right_col.addWidget(intro)

        desc = QLabel(
            "Thực hiện lần lượt các cử chỉ 1 tay và 2 tay trước Camera. "
            "Khi máy nhận diện chính xác, ô tương ứng sẽ sáng XANH LÁ (✓ Đạt). "
            "Nếu tay quay ngang hoặc che khuất, hệ thống sẽ báo ĐỎ để bạn điều chỉnh góc tay."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11.5px;")
        right_col.addWidget(desc)

        self.live_feedback_lbl = QLabel("Trạng thái trực tiếp: Chưa phát hiện bàn tay")
        self.live_feedback_lbl.setWordWrap(True)
        self.live_feedback_lbl.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1.5px solid {BORDER_SLATE}; border-radius: 8px; padding: 8px 10px; font-size: 12px; font-weight: 700;"
        )
        right_col.addWidget(self.live_feedback_lbl)

        self.progress_lbl = QLabel("Tiến độ kiểm tra: 0 / 8 cử chỉ đã xác nhận")
        self.progress_lbl.setStyleSheet(f"color: {BRIGHT_YELLOW}; font-size: 12px; font-weight: 800;")
        right_col.addWidget(self.progress_lbl)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("↺ Thử Lại Từ Đầu")
        reset_btn.clicked.connect(self.reset_progress)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        right_col.addLayout(btn_row)
        right_col.addStretch(1)
        top.addLayout(right_col, stretch=1)
        root.addLayout(top)

        grid = QGridLayout()
        grid.setSpacing(8)
        for idx, (gid, title, sub) in enumerate(self.SANDBOX_TARGETS):
            lbl = QLabel(f"{title}\n{sub}\nChờ thực hiện")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(68)
            lbl.setStyleSheet(
                f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1.5px solid {BORDER_SLATE}; border-radius: 9px; font-size: 11px; font-weight: 700; padding: 4px;"
            )
            grid.addWidget(lbl, idx // 4, idx % 4)
            self._cards[gid] = lbl
        root.addLayout(grid)

    def reset_progress(self) -> None:
        self.passed_gestures.clear()
        for gid, title, sub in self.SANDBOX_TARGETS:
            lbl = self._cards[gid]
            lbl.setText(f"{title}\n{sub}\nChờ thực hiện")
            lbl.setStyleSheet(
                f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1.5px solid {BORDER_SLATE}; border-radius: 9px; font-size: 11px; font-weight: 700; padding: 4px;"
            )
        self.progress_lbl.setText(f"Tiến độ kiểm tra: 0 / {len(self.SANDBOX_TARGETS)} cử chỉ đã xác nhận")
        self.live_feedback_lbl.setText("Trạng thái trực tiếp: Đã đặt lại — Hãy giơ tay trước Camera")
        self.live_feedback_lbl.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1.5px solid {BORDER_SLATE}; border-radius: 8px; padding: 8px 10px; font-size: 12px; font-weight: 700;"
        )

    def _mark_passed(self, gid: str) -> None:
        if gid not in self._cards:
            return
        self.passed_gestures.add(gid)
        for k, title, sub in self.SANDBOX_TARGETS:
            if k == gid:
                lbl = self._cards[gid]
                lbl.setText(f"{title}\n{sub}\n✓ ĐÃ NHẬN DIỆN")
                lbl.setStyleSheet(
                    f"background: #064E3B; color: #ECFDF5; border: 2px solid {SUCCESS_GREEN}; border-radius: 9px; font-size: 11px; font-weight: 800; padding: 4px;"
                )
                break
        self.progress_lbl.setText(
            f"Tiến độ kiểm tra: {len(self.passed_gestures)} / {len(self.SANDBOX_TARGETS)} cử chỉ đã xác nhận"
        )

    def ingest_camera_packet(self, pkt: Dict[str, Any]) -> str:
        """Process a live packet from `VisionInferenceWorker` and return recognized gesture key."""
        qimg = pkt.get("qimage")
        if isinstance(qimg, QImage) and not qimg.isNull():
            self.cam_preview.setPixmap(
                QPixmap.fromImage(qimg.scaled(306, 206, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            )

        pose_tag = str(pkt.get("pose_tag") or pkt.get("gesture") or "IDLE").upper()
        dual_info = pkt.get("dual_hand") or {}
        two_mode = str(dual_info.get("two_hand_mode") or "SINGLE").upper()
        nav_ev = str(pkt.get("nav_event") or "").upper()
        vt = pkt.get("vision_telemetry") or {}
        palm_ok = bool(vt.get("palm_facing_ok", True))

        matched = ""
        if two_mode in ("DUAL_PINCH_ZOOM", "DUAL_PALM_PAN"):
            matched = two_mode
        elif nav_ev in ("SLIDE_NEXT", "SLIDE_PREV"):
            matched = "SWIPE"
        elif pose_tag in ("INDEX", "TWO_FINGERS", "PINCH", "OPEN_PALM", "THREE_FINGERS"):
            matched = pose_tag

        if matched:
            self._mark_passed(matched)
            self.live_feedback_lbl.setText(f"✓ KHỚP CHÍNH XÁC: {matched} (Xanh lá • Sẵn sàng giảng dạy)")
            self.live_feedback_lbl.setStyleSheet(
                f"background: #064E3B; color: #ECFDF5; border: 2px solid {SUCCESS_GREEN}; border-radius: 8px; padding: 8px 10px; font-size: 12px; font-weight: 800;"
            )
        elif pose_tag == "IDLE":
            self.live_feedback_lbl.setText("Chưa thấy bàn tay trong khung hình Camera (giữ khoảng cách 0.5m–2.5m)")
            self.live_feedback_lbl.setStyleSheet(
                f"background: {SURFACE_DARK}; color: {TEXT_MUTED}; border: 1.5px solid {BORDER_SLATE}; border-radius: 8px; padding: 8px 10px; font-size: 12px; font-weight: 700;"
            )
        elif not palm_ok:
            self.live_feedback_lbl.setText("✗ CHƯA KHỚP: Lòng bàn tay đang quay nghiêng — Hãy hướng lòng bàn tay về Camera")
            self.live_feedback_lbl.setStyleSheet(
                f"background: #7F1D1D; color: #FEE2E2; border: 2px solid {ALERT_ROSE}; border-radius: 8px; padding: 8px 10px; font-size: 12px; font-weight: 800;"
            )
        else:
            self.live_feedback_lbl.setText(f"✗ Tư thế trung gian ({pose_tag}) — Hãy tách rõ ngón tay theo hướng dẫn")
            self.live_feedback_lbl.setStyleSheet(
                f"background: #7F1D1D; color: #FEE2E2; border: 1.5px solid {ALERT_ROSE}; border-radius: 8px; padding: 8px 10px; font-size: 12px; font-weight: 700;"
            )
        return matched


class OnboardingTourDialog(QDialog):
    """Interactive 4-step First-Run Onboarding Guide shown on first launch (skippable)."""

    open_sandbox_requested = Signal()

    STEPS: Tuple[Tuple[str, str, str, str, str], ...] = (
        (
            "MOUSE_NAV",
            "Bước 1/4 — Di Chuyển Bảng Mượt & Object Độc Lập",
            NEON_CYAN,
            "• Giữ Chuột Giữa, giữ phím Space + Chuột Trái, hoặc chọn công cụ ✥ Kéo Bảng để trượt vô tận ở 60fps.\n"
            "• Cuộn chuột để Zoom đúng tâm con trỏ (25%–400%), nhấn Ctrl+0 để về 100% hoặc Vừa Nội Dung.\n"
            "• Mọi Nét vẽ, Hình khối, Văn bản, Ảnh, Sticky Note đều là Object độc lập: chọn (V), kéo 8 góc để co giãn, kéo nút xoay phía trên, nhân bản (Ctrl+D), Ctrl+Z/Ctrl+Y.",
            "Mẹo: Dùng nút khóa 🔓 Pan góc phải dưới để chống trượt bảng nhầm khi đang viết.",
        ),
        (
            "PIE_MENU",
            "Bước 2/4 — Bảng Chọn Tròn Chuột Phải Siêu Tốc",
            "#A855F7",
            "• Nhấn giữ Chuột Phải tại bất kỳ đâu trên bảng → Vòng tròn 8 công cụ lập tức bung ra ngay tại con trỏ.\n"
            "• Di chuột về hướng công cụ cần dùng (Bút vẽ, Tẩy, Chọn, Kéo bảng, Thêm chữ, Hình khối, Hoàn tác, Xóa/Trang mới) → Thả chuột phải để kích hoạt!\n"
            "• Kéo chuột ra vòng ngoài ở mục Bút vẽ / Hình khối / Xóa để chọn nhanh Màu mực, Loại hình hoặc Thêm trang mới.",
            "Mẹo: Thả chuột trong vùng tâm vòng tròn (< 22px) sẽ hủy thao tác an toàn.",
        ),
        (
            "DUAL_ZOOM",
            "Bước 3/4 — Cử Chỉ 1 Tay & 2 Tay Không Chạm (Rust Core)",
            BRIGHT_YELLOW,
            "• 1 Tay: ☝️ 1 ngón trỏ = Vẽ nét | ✌️ 2 ngón chữ V = Con trỏ Hover | 🤏 Chụm Cái+Trỏ = Cầm & di chuyển Object | 🖐️ Xòe 5 ngón = Tẩy | ↔ Vẫy ngang = Đổi Slide.\n"
            "• 2 Tay: 🔍 2 tay cùng chụm kéo ra/vào = Zoom mượt | ✋ 2 tay cùng nắm hoặc cùng xòe di chuyển song song = Di chuyển bảng hoặc cuộn tài liệu PDF/DOCX.",
            "Mẹo: Thanh HUD phía trên bảng luôn hiển thị rõ cử chỉ máy đang nhận diện theo thời gian thực.",
        ),
        (
            "AI_LOCAL",
            "Bước 4/4 — Trình Chiếu PDF/PPTX/DOCX & AI Cloud / Local Offline",
            SUCCESS_GREEN,
            "• Mở trực tiếp PDF, PPTX, DOCX, Ảnh hoặc Thư mục Ảnh; mỗi trang lưu lớp chú thích Object độc lập và xuất ra PDF/PNG.\n"
            "• Trợ lý AI đọc hiểu ngữ cảnh Slide hiện tại, trả lời chi tiết có Markdown/LaTeX (streaming), nút Sao chép và 'Đưa nội dung lên bảng'.\n"
            "• Chuyển 1-click giữa Cloud Gemini ↔ Local AI Offline (tải model GGUF ngay trong 🤖 Model Hub từ HuggingFace / GitHub).",
            "Bạn luôn có thể mở lại hướng dẫn và Phòng Thử Cử Chỉ bất cứ lúc nào bằng phím F1 hoặc nút '?'.",
        ),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chào mừng đến với AirTeach AI — Hướng Dẫn Nhanh (4 Bước)")
        self.resize(680, 440)
        self.setStyleSheet(f"background: {BG_DARK}; color: {TEXT_LIGHT};")
        self._step_idx = 0
        self.dont_show_again = True
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        hdr = QHBoxLayout()
        t = QLabel("KHÁM PHÁ AIRTEACH AI TRONG 30 GIÂY")
        t.setStyleSheet(f"color: {NEON_CYAN}; font-size: 16px; font-weight: 800;")
        self.step_pill = QLabel("Bước 1 / 4")
        self.step_pill.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 10px; padding: 4px 10px; font-size: 11px; font-weight: 800;"
        )
        hdr.addWidget(t)
        hdr.addStretch(1)
        hdr.addWidget(self.step_pill)
        root.addLayout(hdr)

        self.stack = QStackedWidget()
        for diag, title, accent, body, tip in self.STEPS:
            page = QFrame()
            page.setStyleSheet(
                f"background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 12px;"
            )
            pl = QVBoxLayout(page)
            pl.setContentsMargins(16, 14, 16, 14)
            pl.setSpacing(10)

            top_r = QHBoxLayout()
            top_r.addWidget(GestureDiagramWidget(diag, accent_hex=accent))
            tl = QLabel(title)
            tl.setWordWrap(True)
            tl.setStyleSheet(f"color: {accent}; font-size: 15px; font-weight: 800; border: none; background: transparent;")
            top_r.addWidget(tl, stretch=1)
            pl.addLayout(top_r)

            bl = QLabel(body)
            bl.setWordWrap(True)
            bl.setStyleSheet(f"color: {TEXT_LIGHT}; font-size: 12.5px; line-height: 1.5; border: none; background: transparent;")
            pl.addWidget(bl)

            tip_lbl = QLabel(tip)
            tip_lbl.setWordWrap(True)
            tip_lbl.setStyleSheet(
                f"background: {BG_DARK}; color: {SECONDARY_TEAL}; border: 1px solid {BORDER_SLATE}; border-radius: 8px; padding: 8px 10px; font-size: 11.5px; font-weight: 600;"
            )
            pl.addWidget(tip_lbl)
            pl.addStretch(1)
            self.stack.addWidget(page)
        root.addWidget(self.stack, stretch=1)

        foot = QHBoxLayout()
        self.dont_show_chk = QCheckBox("Không hiện lại ở lần khởi động sau")
        self.dont_show_chk.setChecked(True)
        self.dont_show_chk.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11.5px;")
        foot.addWidget(self.dont_show_chk)
        foot.addStretch(1)

        sandbox_btn = QPushButton("🧪 Thử Cử Chỉ Ngay")
        sandbox_btn.setStyleSheet(
            f"background: #0D9488; color: {TEXT_LIGHT}; border: none; border-radius: 7px; padding: 6px 12px; font-weight: 700;"
        )
        sandbox_btn.clicked.connect(self._on_open_sandbox)
        foot.addWidget(sandbox_btn)

        skip_btn = QPushButton("Bỏ qua")
        skip_btn.clicked.connect(self._on_finish)
        foot.addWidget(skip_btn)

        self.prev_btn = QPushButton("◀ Trước")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._on_prev)
        foot.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Tiếp theo ▶")
        self.next_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border: none; border-radius: 7px; padding: 6px 14px; font-weight: 800;"
        )
        self.next_btn.clicked.connect(self._on_next)
        foot.addWidget(self.next_btn)
        root.addLayout(foot)

    def _sync_step_ui(self) -> None:
        self.stack.setCurrentIndex(self._step_idx)
        n = len(self.STEPS)
        self.step_pill.setText(f"Bước {self._step_idx + 1} / {n}")
        self.prev_btn.setEnabled(self._step_idx > 0)
        self.next_btn.setText("Bắt Đầu Ngay ✓" if self._step_idx == n - 1 else "Tiếp theo ▶")

    def _on_prev(self) -> None:
        if self._step_idx > 0:
            self._step_idx -= 1
            self._sync_step_ui()

    def _on_next(self) -> None:
        if self._step_idx < len(self.STEPS) - 1:
            self._step_idx += 1
            self._sync_step_ui()
        else:
            self._on_finish()

    def _on_open_sandbox(self) -> None:
        self.dont_show_again = self.dont_show_chk.isChecked()
        self.open_sandbox_requested.emit()
        self.accept()

    def _on_finish(self) -> None:
        self.dont_show_again = self.dont_show_chk.isChecked()
        self.accept()


class UserGuideDialog(QDialog):
    """Built-in interactive 5-tab User Guide, Cheat-Sheet & Live Gesture Sandbox (`?` / `F1`)."""

    start_spotlight_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None, initial_tab: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hướng Dẫn Sử Dụng, Phím Tắt & Phòng Thử Cử Chỉ — AirTeach AI")
        self.resize(880, 650)
        self.setStyleSheet(f"background: {BG_DARK}; color: {TEXT_LIGHT};")

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("HƯỚNG DẪN SỬ DỤNG, BẢNG TRA CỨU CỬ CHỈ & GESTURE SANDBOX")
        title.setStyleSheet(f"color: {TEXT_LIGHT}; font-size: 16px; font-weight: 800;")
        sub = QLabel("Làm chủ cử chỉ 1 & 2 tay, Bảng chọn tròn chuột phải, Object Canvas, Slide đa định dạng & AI Offline")
        sub.setStyleSheet(f"color: {SECONDARY_TEAL}; font-size: 12px; font-weight: 600;")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        hdr.addLayout(title_col, stretch=1)

        self.spotlight_btn = QPushButton("🔦 Tour Spotlight")
        self.spotlight_btn.setFixedHeight(32)
        self.spotlight_btn.setCursor(Qt.PointingHandCursor)
        self.spotlight_btn.setToolTip("Khởi chạy hướng dẫn tương tác Spotlight trực tiếp trên giao diện chính")
        self.spotlight_btn.setStyleSheet(
            f"QPushButton {{ background: #0F766E; color: {TEXT_LIGHT}; border: 1.5px solid {NEON_CYAN}; border-radius: 7px; padding: 0 14px; font-weight: 800; }}"
            f"QPushButton:hover {{ background: #0D9488; }}"
        )
        self.spotlight_btn.clicked.connect(self._on_start_spotlight)
        hdr.addWidget(self.spotlight_btn)

        close_btn = QPushButton("Đóng (Esc)")
        close_btn.setFixedHeight(32)
        close_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 7px; padding: 0 14px; font-weight: 700;"
        )
        close_btn.clicked.connect(self.accept)
        hdr.addWidget(close_btn)
        root.addLayout(hdr)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{ background: {BG_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px; }}
            QTabBar::tab {{ background: {SURFACE_DARK}; color: {TEXT_MUTED}; padding: 9px 13px; font-weight: 700; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 4px; }}
            QTabBar::tab:selected {{ background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; }}
            """
        )

        # Tab 1: Cử Chỉ 1 Tay
        tab1_cards: List[QWidget] = [
            _make_guide_card(
                "PINCH_DRAW",
                "1. Giơ 1 Ngón Trỏ (Index) để Vẽ Nét",
                "Giơ 1 ngón trỏ thẳng để vẽ nét mượt 60Hz (làm mượt Catmull-Rom & nắn hình học bằng lõi Rust `airteach_core_rs`).",
                NEON_CYAN,
            ),
            _make_guide_card(
                "POINTER",
                "2. Giơ 2 Ngón Chữ V (Trỏ + Giữa) để Di Chuyển Con Trỏ / Hover",
                "Dùng 2 ngón chữ V để trỏ chỉ bài hoặc di chuyển con trỏ mà KHÔNG tạo nét vẽ thừa trên bảng.",
                SECONDARY_TEAL,
            ),
            _make_guide_card(
                "PINCH_GRAB",
                "3. Chụm Ngón Cái + Trỏ (Pinch) để Cầm & Di Chuyển Object",
                "Đưa tay vào một Hình khối, Hộp chữ, Ghi chú hoặc Ảnh trên bảng rồi chụm ngón cái + ngón trỏ để kéo di chuyển đối tượng đó mượt mà.",
                BRIGHT_YELLOW,
            ),
            _make_guide_card(
                "OPEN_ERASE",
                "4. Xòe Cả 5 Ngón (Open Palm) để Tẩy Nét / Dừng Vẽ",
                "Xòe rộng lòng bàn tay hướng vào camera để bật vòng tẩy động hoặc hủy nét đang vẽ.",
                ALERT_ROSE,
            ),
            _make_guide_card(
                "SWIPE_H",
                "5. Vẫy Tay Ngang (Trái / Phải) Đổi Trang / Slide",
                "Vẫy dứt khoát sang Trái để qua trang kế tiếp hoặc sang Phải để lùi trang trước (có cooldown chống lật nhầm).",
                BRIGHT_YELLOW,
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab1_cards), "1. Cử Chỉ 1 Tay")

        # Tab 2: Cử Chỉ 2 Tay
        tab2_cards: List[QWidget] = [
            _make_guide_card(
                "DUAL_ZOOM",
                "1. Thu / Phóng Bảng (2 Tay Cùng Chụm — Dual Pinch)",
                "Chụm cả hai tay rồi kéo rộng ra để Phóng to (tới 400%) hoặc thu hẹp lại để Thu nhỏ (tới 25%), lấy tâm đúng điểm giữa 2 bàn tay (lọc mượt Rust EMA).",
                BRIGHT_YELLOW,
            ),
            _make_guide_card(
                "DUAL_PAN",
                "2. Di Chuyển Bảng & Cuộn Tài Liệu (2 Tay Cùng Nắm hoặc Cùng Xòe)",
                "Cùng nắm hoặc cùng xòe cả hai bàn tay rồi di chuyển song song để kéo bảng vô tận mọi hướng, hoặc cuộn mượt tài liệu PDF / DOCX lên xuống khi đang bật Cuộn Dọc.",
                SECONDARY_TEAL,
            ),
            _make_guide_card(
                "MOD_LOCK",
                "3. Khóa Nét Thẳng, Nắn Hình & Hoàn Tác (Phối Hợp 2 Tay)",
                "• Tay phụ NẮM + Tay chính VẼ: Khóa đoạn thẳng tuyệt đối.\n"
                "• Tay phụ GIƠ 2 NGÓN + Tay chính VẼ: Tăng cường chuẩn hóa hình tròn, vuông, chữ nhật, tam giác, mũi tên.\n"
                "• Tay phụ GIƠ 3 NGÓN: Hoàn tác (Undo) tức thì.",
                SUCCESS_GREEN,
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab2_cards), "2. Cử Chỉ 2 Tay")

        # Tab 3: Chuột, Phím Tắt & Bảng Tròn
        pie_grid_frame = QFrame()
        pie_grid_frame.setStyleSheet(
            f"background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px;"
        )
        pg_vbox = QVBoxLayout(pie_grid_frame)
        pg_vbox.setContentsMargins(12, 10, 12, 10)
        pg_title = QLabel("Bố cục 8 Hướng Bảng Tròn Chuột Phải (+ Vòng Con ngoài ở Bút / Hình / Xóa)")
        pg_title.setStyleSheet(f"color: {NEON_CYAN}; font-size: 13px; font-weight: 800; border: none;")
        pg_vbox.addWidget(pg_title)
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, sl in enumerate(PIE_MENU_SLICES):
            chip = QLabel(f"{sl.direction} · {sl.full_label_vi}")
            chip.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {sl.accent_hex}; border-radius: 6px; padding: 6px 8px; font-size: 11.5px; font-weight: 700;"
            )
            grid.addWidget(chip, i // 2, i % 2)
        pg_vbox.addLayout(grid)

        tab3_cards: List[QWidget] = [
            _make_guide_card(
                "PIE_MENU",
                "Bảng Chọn Tròn Chuột Phải (Nhấn giữ → Kéo hướng → Thả)",
                "Nhấn giữ Chuột Phải bất kỳ đâu trên bảng để mở vòng tròn 8 công cụ. Thả chuột để kích hoạt; giữ ở tâm (< 22px) để Hủy; kéo xa ra vòng ngoài ở Bút vẽ / Hình khối / Cử Chỉ & AI để chọn nhanh Màu mực, Hình cụ thể, Sơ Đồ Cử Chỉ hoặc Mở AI Chat.",
                NEON_CYAN,
            ),
            pie_grid_frame,
            _make_guide_card(
                "MOUSE_NAV",
                "Điều Hướng Bảng, Object Canvas & Phím Tắt Chuẩn",
                "• Kéo Chuột Giữa hoặc giữ Space + Chuột Trái: Di chuyển bảng vô tận 60fps.\n"
                "• Cuộn Chuột: Zoom-to-cursor (25%–400%) · Ctrl+0: Reset 100% · Nút Vừa Nội Dung: Fit to Content.\n"
                "• Phím tắt công cụ: V (Chọn/Di chuyển/Resize/Xoay Object) · S (Hình khối) · T (Thêm chữ) · N (Sticky Note).\n"
                "• Thao tác Object: Ctrl+D (Nhân bản) · Delete (Xóa) · Ctrl+Z (Hoàn tác) · Ctrl+Y (Làm lại) · Nhấn đúp để sửa chữ.",
                SECONDARY_TEAL,
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab3_cards), "3. Chuột & Bảng Tròn")

        # Tab 4: AI Local & Tài Liệu
        tab4_cards: List[QWidget] = [
            _make_guide_card(
                "AI_LOCAL",
                "1. Trợ Lý AI Chuyên Sâu & Kho Model Local Offline (GGUF)",
                "• Trả lời đầy đủ (tới 8192 token) chuẩn Markdown, bảng biểu, công thức LaTeX, streaming thời gian thực.\n"
                "• Nút 'Sao chép' và 'Đưa nội dung lên bảng' tạo ngay Text Object độc lập.\n"
                "• Mở Kho Model Hub để tải Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, Gemma-2-2B hoặc dán link GitHub/HuggingFace bất kỳ; chuyển Cloud ↔ Local chỉ 1 cú click.",
                "#A855F7",
            ),
            _make_guide_card(
                "DOCS",
                "2. Mở & Trình Chiếu Đa Định Dạng (PDF / PPTX / DOCX / Ảnh)",
                "• Render PDF sắc nét tới 400% bằng PyMuPDF, cache trang kề lật tức thì.\n"
                "• Mở Ảnh làm nền Slide hoặc Chèn Ảnh làm Object độc lập (di chuyển, co giãn 8 góc, xoay).\n"
                "• Mỗi trang có lớp chú thích Object riêng biệt; xuất trọn vẹn bài giảng kèm chú thích ra PDF hoặc bộ ảnh PNG.",
                NEON_CYAN,
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab4_cards), "4. AI Local & Tài Liệu")

        # Tab 5: Live Gesture Sandbox
        self.sandbox = GestureSandboxWidget(self)
        self.tabs.addTab(self.sandbox, "5. Thử Cử Chỉ (Sandbox)")

        self.tabs.setCurrentIndex(max(0, min(self.tabs.count() - 1, int(initial_tab))))
        root.addWidget(self.tabs, stretch=1)

    def _on_start_spotlight(self) -> None:
        self.start_spotlight_requested.emit()
        self.accept()


class GestureCheatSheetDialog(QDialog):
    """
    2-Column Visual Gesture Cheat-Sheet Modal (Cử Chỉ 1 Tay | Cử Chỉ 2 Tay)
    with crisp vector hand diagrams, activation thresholds, and a live camera
    gesture test bar wired directly to VisionInferenceWorker.
    """

    start_spotlight_requested = Signal()
    open_full_guide_requested = Signal()

    ONE_HAND_ITEMS: Tuple[Tuple[str, str, str, str, str], ...] = (
        (
            "PINCH_DRAW",
            "1 Ngón Trỏ (INDEX) — Vẽ nét",
            "Ngưỡng: Ngón trỏ duỗi thẳng (< 42°), các ngón khác khép",
            "Vẽ nét mượt 60Hz qua bộ lọc One-Euro + Kalman Rust và tự nắn hình học.",
            NEON_CYAN,
        ),
        (
            "POINTER",
            "2 Ngón Chữ V (TWO_FINGERS) — Di chuyển con trỏ",
            "Ngưỡng: Trỏ + Giữa duỗi thẳng, mở > 0.40",
            "Chỉ bài / Hover con trỏ không để lại nét mực thừa trên bảng.",
            SECONDARY_TEAL,
        ),
        (
            "PINCH_GRAB",
            "Chụm Cái + Trỏ (PINCH) — Chọn & Kéo Object",
            "Ngưỡng vào: tỷ lệ < 0.28 (giữ >= 3 frame) · thoát > 0.40",
            "Chụm trực tiếp lên Nét vẽ / Hình khối / Chữ / Ảnh để kéo thả mượt mà.",
            BRIGHT_YELLOW,
        ),
        (
            "OPEN_ERASE",
            "Xòe 5 Ngón (OPEN_PALM) — Tẩy nét & Dừng vẽ",
            "Ngưỡng: 5 ngón duỗi thẳng, lòng bàn tay hướng camera (z >= 0.20)",
            "Bật vòng tẩy động theo kích thước bàn tay hoặc hủy nét đang vẽ.",
            ALERT_ROSE,
        ),
        (
            "SWIPE_H",
            "Vẫy Tay Ngang (SWIPE) — Chuyển Slide",
            "Ngưỡng: dịch chuyển >= 16% khung hình, vận tốc >= 0.85, cooldown 0.6s",
            "Vẫy Trái → Slide tiếp theo · Vẫy Phải → Slide trước (không tẩy nhầm mực).",
            SUCCESS_GREEN,
        ),
    )

    TWO_HAND_ITEMS: Tuple[Tuple[str, str, str, str, str], ...] = (
        (
            "DUAL_ZOOM",
            "2 Tay Cùng Chụm (DUAL_PINCH_ZOOM) — Phóng to / Thu nhỏ",
            "Ngưỡng: cả 2 tay pinch < 0.33, zoom mượt 25% – 400%",
            "Kéo 2 tay ra xa để Phóng to, khép lại để Thu nhỏ quanh đúng tâm 2 tay.",
            BRIGHT_YELLOW,
        ),
        (
            "DUAL_PAN",
            "2 Tay Cùng Mở / Nắm (DUAL_PALM_PAN) — Kéo bảng & Cuộn",
            "Ngưỡng: cả 2 tay OPEN_PALM hoặc FIST di chuyển song song",
            "Trượt bảng vô tận 360° hoặc cuộn mượt tài liệu PDF / DOCX nhiều trang.",
            SECONDARY_TEAL,
        ),
        (
            "MOD_LOCK",
            "Tay Phụ Nắm + Tay Chính Vẽ (LINE_LOCK) — Kẻ đường thẳng",
            "Ngưỡng: tay phụ FIST + tay chính đang vẽ",
            "Khóa đường thẳng tuyệt đối và tự bắt góc chuẩn 0° / 45° / 90°.",
            NEON_CYAN,
        ),
        (
            "SHAPES",
            "Tay Phụ 2 Ngón + Tay Chính Vẽ (SHAPE_LOCK) — Nắn hình học",
            "Ngưỡng: tay phụ TWO_FINGERS + tay chính đang vẽ",
            "Ưu tiên chuẩn hóa hình tròn, vuông, chữ nhật, tam giác và mũi tên vector.",
            "#A855F7",
        ),
        (
            "UNDO",
            "Tay Phụ Giơ 3 Ngón (THREE_FINGERS) — Hoàn tác tức thì",
            "Ngưỡng: 3 ngón duỗi ổn định ~220ms, cooldown 400ms",
            "Hoàn tác nhanh thao tác vừa thực hiện bằng tay phụ không cần chạm phím.",
            SUCCESS_GREEN,
        ),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sơ Đồ Cử Chỉ Trực Quan (1 Tay & 2 Tay) — AirTeach AI")
        self.resize(980, 680)
        self.setStyleSheet(f"background: {BG_DARK}; color: {TEXT_LIGHT};")
        self.tested_gestures: set[str] = set()
        self._test_pills: Dict[str, QLabel] = {}
        self._build_ui()

    def _make_threshold_card(
        self,
        diagram_kind: str,
        title: str,
        threshold: str,
        desc: str,
        accent_hex: str,
    ) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px;"
        )
        row = QHBoxLayout(card)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        row.addWidget(GestureDiagramWidget(diagram_kind, accent_hex=accent_hex))

        col = QVBoxLayout()
        col.setSpacing(3)
        t_lbl = QLabel(title)
        t_lbl.setWordWrap(True)
        t_lbl.setStyleSheet(
            f"color: {accent_hex}; font-size: 12.5px; font-weight: 800; border: none; background: transparent;"
        )
        th_lbl = QLabel(threshold)
        th_lbl.setWordWrap(True)
        th_lbl.setStyleSheet(
            f"color: {BRIGHT_YELLOW}; font-size: 10.5px; font-weight: 700; border: none; background: transparent;"
        )
        d_lbl = QLabel(desc)
        d_lbl.setWordWrap(True)
        d_lbl.setStyleSheet(
            f"color: {TEXT_LIGHT}; font-size: 11.5px; border: none; background: transparent;"
        )
        col.addWidget(t_lbl)
        col.addWidget(th_lbl)
        col.addWidget(d_lbl)
        row.addLayout(col, stretch=1)
        return card

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        t_col = QVBoxLayout()
        t_col.setSpacing(2)
        title = QLabel("SƠ ĐỒ CỬ CHỈ KHÔNG CHẠM — 1 TAY & 2 TAY (RUST 60HZ)")
        title.setStyleSheet(f"color: {NEON_CYAN}; font-size: 15.5px; font-weight: 800;")
        sub = QLabel("Bảng tra cứu 2 cột kèm ngưỡng kích hoạt và thanh kiểm thử trực tiếp qua Webcam")
        sub.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11.5px; font-weight: 600;")
        t_col.addWidget(title)
        t_col.addWidget(sub)
        hdr.addLayout(t_col, stretch=1)

        self.spotlight_btn = QPushButton("Tour Spotlight")
        self.spotlight_btn.setFixedHeight(30)
        self.spotlight_btn.setStyleSheet(
            f"background: #0F766E; color: {TEXT_LIGHT}; border: 1px solid {NEON_CYAN}; border-radius: 7px; padding: 0 12px; font-weight: 700;"
        )
        self.spotlight_btn.clicked.connect(self._on_spotlight)
        hdr.addWidget(self.spotlight_btn)

        self.full_guide_btn = QPushButton("Hướng dẫn đầy đủ")
        self.full_guide_btn.setFixedHeight(30)
        self.full_guide_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 7px; padding: 0 12px; font-weight: 700;"
        )
        self.full_guide_btn.clicked.connect(self._on_full_guide)
        hdr.addWidget(self.full_guide_btn)

        close_btn = QPushButton("Đóng")
        close_btn.setFixedHeight(30)
        close_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border: none; border-radius: 7px; padding: 0 14px; font-weight: 700;"
        )
        close_btn.clicked.connect(self.accept)
        hdr.addWidget(close_btn)
        root.addLayout(hdr)

        # Two-column Gesture Reference Area
        cols = QHBoxLayout()
        cols.setSpacing(12)

        self.one_hand_col_frame = QFrame()
        self.one_hand_col_frame.setStyleSheet(
            f"QFrame {{ background: #0B1222; border: 1px solid {BORDER_SLATE}; border-radius: 10px; }}"
        )
        c1_vbox = QVBoxLayout(self.one_hand_col_frame)
        c1_vbox.setContentsMargins(10, 8, 10, 8)
        c1_vbox.setSpacing(6)
        c1_hdr = QLabel("CỬ CHỈ 1 TAY (Vẽ · Trỏ · Kéo Object · Tẩy · Đổi Slide)")
        c1_hdr.setStyleSheet(f"color: {NEON_CYAN}; font-size: 12.5px; font-weight: 800; border: none; background: transparent;")
        c1_vbox.addWidget(c1_hdr)
        c1_cards = [self._make_threshold_card(*item) for item in self.ONE_HAND_ITEMS]
        c1_vbox.addWidget(_wrap_scroll(c1_cards), stretch=1)
        cols.addWidget(self.one_hand_col_frame, stretch=1)

        self.two_hand_col_frame = QFrame()
        self.two_hand_col_frame.setStyleSheet(
            f"QFrame {{ background: #0B1222; border: 1px solid {BORDER_SLATE}; border-radius: 10px; }}"
        )
        c2_vbox = QVBoxLayout(self.two_hand_col_frame)
        c2_vbox.setContentsMargins(10, 8, 10, 8)
        c2_vbox.setSpacing(6)
        c2_hdr = QLabel("CỬ CHỈ 2 TAY (Zoom · Kéo Bảng · Khóa Thẳng/Hình · Undo)")
        c2_hdr.setStyleSheet(f"color: {BRIGHT_YELLOW}; font-size: 12.5px; font-weight: 800; border: none; background: transparent;")
        c2_vbox.addWidget(c2_hdr)
        c2_cards = [self._make_threshold_card(*item) for item in self.TWO_HAND_ITEMS]
        c2_vbox.addWidget(_wrap_scroll(c2_cards), stretch=1)
        cols.addWidget(self.two_hand_col_frame, stretch=1)

        root.addLayout(cols, stretch=1)

        # Live Gesture Test Bar at Bottom
        self.live_test_bar = QFrame()
        self.live_test_bar.setStyleSheet(
            f"QFrame {{ background: {SURFACE_DARK}; border: 1.5px solid {SECONDARY_TEAL}; border-radius: 10px; }}"
        )
        lt_row = QHBoxLayout(self.live_test_bar)
        lt_row.setContentsMargins(10, 8, 10, 8)
        lt_row.setSpacing(10)

        self.live_cam_lbl = QLabel("Camera Test\nChờ khung hình")
        self.live_cam_lbl.setFixedSize(120, 76)
        self.live_cam_lbl.setAlignment(Qt.AlignCenter)
        self.live_cam_lbl.setWordWrap(True)
        self.live_cam_lbl.setStyleSheet(
            f"background: {BG_DARK}; color: {TEXT_MUTED}; border: 1px solid {BORDER_SLATE}; border-radius: 7px; font-size: 10px; font-weight: 700;"
        )
        lt_row.addWidget(self.live_cam_lbl)

        lt_mid = QVBoxLayout()
        lt_mid.setSpacing(6)
        top_status = QHBoxLayout()
        self.live_status_lbl = QLabel("Thanh Test Trực Tiếp: Hãy giơ 1 tay hoặc 2 tay trước Camera để kiểm thử ngay")
        self.live_status_lbl.setWordWrap(True)
        self.live_status_lbl.setStyleSheet(
            f"color: {TEXT_LIGHT}; font-size: 12px; font-weight: 700; border: none; background: transparent;"
        )
        self.live_progress_lbl = QLabel("0 / 8 đã thử")
        self.live_progress_lbl.setStyleSheet(
            f"color: {NEON_CYAN}; font-size: 11.5px; font-weight: 800; border: none; background: transparent;"
        )
        reset_btn = QPushButton("Đặt lại")
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self.reset_live_test)
        top_status.addWidget(self.live_status_lbl, stretch=1)
        top_status.addWidget(self.live_progress_lbl)
        top_status.addWidget(reset_btn)
        lt_mid.addLayout(top_status)

        pills_row = QHBoxLayout()
        pills_row.setSpacing(5)
        for gid, short_name in (
            ("INDEX", "1 Ngón"),
            ("TWO_FINGERS", "2 Ngón"),
            ("PINCH", "Chụm"),
            ("OPEN_PALM", "5 Ngón"),
            ("SWIPE", "Vẫy"),
            ("DUAL_PINCH_ZOOM", "2T Zoom"),
            ("DUAL_PALM_PAN", "2T Pan"),
            ("THREE_FINGERS", "3 Ngón"),
        ):
            pill = QLabel(short_name)
            pill.setAlignment(Qt.AlignCenter)
            pill.setFixedHeight(24)
            pill.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_MUTED}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 0 7px; font-size: 10.5px; font-weight: 700;"
            )
            pills_row.addWidget(pill)
            self._test_pills[gid] = pill
        lt_mid.addLayout(pills_row)
        lt_row.addLayout(lt_mid, stretch=1)
        root.addWidget(self.live_test_bar)

    def reset_live_test(self) -> None:
        self.tested_gestures.clear()
        for pill in self._test_pills.values():
            pill.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_MUTED}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 0 7px; font-size: 10.5px; font-weight: 700;"
            )
        self.live_progress_lbl.setText(f"0 / {len(self._test_pills)} đã thử")
        self.live_status_lbl.setText("Thanh Test Trực Tiếp: Đã đặt lại — Hãy giơ tay trước Camera")
        self.live_status_lbl.setStyleSheet(
            f"color: {TEXT_LIGHT}; font-size: 12px; font-weight: 700; border: none; background: transparent;"
        )

    def ingest_camera_packet(self, pkt: Dict[str, Any]) -> str:
        qimg = pkt.get("qimage")
        if isinstance(qimg, QImage) and not qimg.isNull():
            self.live_cam_lbl.setPixmap(
                QPixmap.fromImage(qimg.scaled(116, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            )

        pose_tag = str(pkt.get("pose_tag") or pkt.get("gesture") or "IDLE").upper()
        dual_info = pkt.get("dual_hand") or {}
        two_mode = str(dual_info.get("two_hand_mode") or "SINGLE").upper()
        nav_ev = str(pkt.get("nav_event") or "").upper()
        vt = pkt.get("vision_telemetry") or {}
        palm_ok = bool(vt.get("palm_facing_ok", True))

        matched = ""
        if two_mode in ("DUAL_PINCH_ZOOM", "DUAL_PALM_PAN"):
            matched = two_mode
        elif nav_ev in ("SLIDE_NEXT", "SLIDE_PREV"):
            matched = "SWIPE"
        elif pose_tag in ("INDEX", "TWO_FINGERS", "PINCH", "OPEN_PALM", "THREE_FINGERS"):
            matched = pose_tag

        if matched and matched in self._test_pills:
            self.tested_gestures.add(matched)
            self._test_pills[matched].setStyleSheet(
                f"background: #064E3B; color: #ECFDF5; border: 1.5px solid {SUCCESS_GREEN}; border-radius: 6px; padding: 0 7px; font-size: 10.5px; font-weight: 800;"
            )
            self.live_progress_lbl.setText(f"{len(self.tested_gestures)} / {len(self._test_pills)} đã thử")
            self.live_status_lbl.setText(f"✓ ĐÃ NHẬN DIỆN: {matched} — Hoạt động ổn định")
            self.live_status_lbl.setStyleSheet(
                f"color: {SUCCESS_GREEN}; font-size: 12px; font-weight: 800; border: none; background: transparent;"
            )
        elif pose_tag == "IDLE":
            self.live_status_lbl.setText("Chờ tay trước Camera (0.5m – 2.5m)")
            self.live_status_lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 12px; font-weight: 700; border: none; background: transparent;"
            )
        elif not palm_ok:
            self.live_status_lbl.setText("✗ Hãy xoay lòng bàn tay hướng thẳng về Camera")
            self.live_status_lbl.setStyleSheet(
                f"color: {ALERT_ROSE}; font-size: 12px; font-weight: 800; border: none; background: transparent;"
            )
        else:
            self.live_status_lbl.setText(f"Tư thế hiện tại: {pose_tag}")
            self.live_status_lbl.setStyleSheet(
                f"color: {BRIGHT_YELLOW}; font-size: 12px; font-weight: 700; border: none; background: transparent;"
            )
        return matched

    def _on_spotlight(self) -> None:
        self.start_spotlight_requested.emit()
        self.accept()

    def _on_full_guide(self) -> None:
        self.open_full_guide_requested.emit()
        self.accept()
