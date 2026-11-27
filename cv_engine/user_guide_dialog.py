#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Built-in Interactive User Guide & Gesture Cheat-Sheet (`cv_engine/user_guide_dialog.py`)
for AirTeach AI (DNT - Victer).

4 Visual Tabs with custom vector QPainter diagrams:
  1. Cử Chỉ 1 Tay (Pointer, Pinch Draw, Open-Palm Erase, Horizontal/Vertical Air-Swipe)
  2. Cử Chỉ 2 Tay (Dual-Pinch Zoom 25%-400%, Dual-Palm Pan, Fist+Move Pan, Off-Hand Modifiers)
  3. Chuột & Bảng Tròn (8-Slice Right-Click Radial Pie Menu + Mouse/Keyboard Navigation)
  4. AI Local & Tài Liệu (GGUF Model Manager + Word/PDF/PPTX/Image Folder/Board Images)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
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
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"


class GestureDiagramWidget(QWidget):
    """Crisp custom QPainter vector schematic for each hand/mouse/AI guide card."""

    def __init__(self, diagram_kind: str, accent_hex: str = NEON_CYAN, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.diagram_kind = diagram_kind
        self.accent = QColor(accent_hex)
        self.setFixedSize(96, 84)

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
            p.setPen(QPen(self.accent, 2.4))
            p.drawEllipse(QPointF(cx, cy), 14.0, 14.0)
            p.setBrush(QColor(ALERT_ROSE))
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 2.0))
            p.drawEllipse(QPointF(cx, cy), 5.0, 5.0)
            p.setPen(QPen(self.accent, 2.0, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx - 12.0, cy + 24.0), QPointF(cx - 2.0, cy + 8.0))
        elif k == "PINCH_DRAW":
            p.setPen(QPen(self.accent, 3.0, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(16.0, 56.0), QPointF(38.0, 30.0))
            p.drawLine(QPointF(38.0, 30.0), QPointF(62.0, 48.0))
            p.drawLine(QPointF(62.0, 48.0), QPointF(80.0, 24.0))
            p.setBrush(QColor(self.accent))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(80.0, 24.0), 5.0, 5.0)
        elif k == "OPEN_ERASE":
            p.setBrush(QColor(244, 63, 94, 55))
            p.setPen(QPen(QColor(ALERT_ROSE), 2.0, Qt.DashLine))
            p.drawEllipse(QPointF(cx, cy), 22.0, 22.0)
            p.setPen(QPen(QColor(TEXT_LIGHT), 2.0))
            p.drawLine(QPointF(cx - 8.0, cy - 8.0), QPointF(cx + 8.0, cy + 8.0))
            p.drawLine(QPointF(cx + 8.0, cy - 8.0), QPointF(cx - 8.0, cy + 8.0))
        elif k == "SWIPE_H":
            p.setPen(QPen(self.accent, 2.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(18.0, cy), QPointF(78.0, cy))
            p.drawLine(QPointF(68.0, cy - 9.0), QPointF(78.0, cy))
            p.drawLine(QPointF(68.0, cy + 9.0), QPointF(78.0, cy))
            p.drawLine(QPointF(28.0, cy - 9.0), QPointF(18.0, cy))
            p.drawLine(QPointF(28.0, cy + 9.0), QPointF(18.0, cy))
        elif k == "SWIPE_V":
            p.setPen(QPen(self.accent, 2.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx, 16.0), QPointF(cx, 68.0))
            p.drawLine(QPointF(cx - 9.0, 26.0), QPointF(cx, 16.0))
            p.drawLine(QPointF(cx + 9.0, 26.0), QPointF(cx, 16.0))
            p.drawLine(QPointF(cx - 9.0, 58.0), QPointF(cx, 68.0))
            p.drawLine(QPointF(cx + 9.0, 58.0), QPointF(cx, 68.0))
        elif k == "DUAL_ZOOM":
            p.setPen(QPen(self.accent, 2.2))
            p.drawEllipse(QPointF(24.0, cy), 8.0, 8.0)
            p.drawEllipse(QPointF(72.0, cy), 8.0, 8.0)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 2.2))
            p.drawLine(QPointF(34.0, cy), QPointF(62.0, cy))
            p.drawLine(QPointF(39.0, cy - 6.0), QPointF(33.0, cy))
            p.drawLine(QPointF(39.0, cy + 6.0), QPointF(33.0, cy))
            p.drawLine(QPointF(57.0, cy - 6.0), QPointF(63.0, cy))
            p.drawLine(QPointF(57.0, cy + 6.0), QPointF(63.0, cy))
        elif k == "DUAL_PAN":
            p.setBrush(QColor(56, 189, 248, 45))
            p.setPen(QPen(self.accent, 1.8))
            p.drawRoundedRect(QRectF(20.0, 18.0, 56.0, 48.0), 6.0, 6.0)
            p.setPen(QPen(QColor(TEXT_LIGHT), 2.0))
            p.drawLine(QPointF(cx - 14.0, cy), QPointF(cx + 14.0, cy))
            p.drawLine(QPointF(cx, cy - 14.0), QPointF(cx, cy + 14.0))
        elif k == "MOD_LOCK":
            p.setPen(QPen(self.accent, 2.4))
            p.drawLine(QPointF(20.0, 58.0), QPointF(76.0, 26.0))
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 1.8))
            p.drawRect(QRectF(22.0, 20.0, 20.0, 16.0))
        elif k == "PIE_MENU":
            p.setPen(QPen(self.accent, 1.8))
            p.drawEllipse(QPointF(cx, cy), 28.0, 28.0)
            p.drawEllipse(QPointF(cx, cy), 9.0, 9.0)
            for i in range(8):
                ang = math.radians(i * 45.0 + 22.5)
                p.drawLine(
                    QPointF(cx + 9.0 * math.cos(ang), cy + 9.0 * math.sin(ang)),
                    QPointF(cx + 28.0 * math.cos(ang), cy + 28.0 * math.sin(ang)),
                )
        elif k == "MOUSE_NAV":
            p.setPen(QPen(self.accent, 2.0))
            p.drawRoundedRect(QRectF(30.0, 14.0, 36.0, 56.0), 14.0, 14.0)
            p.drawLine(QPointF(30.0, 36.0), QPointF(66.0, 36.0))
            p.setBrush(QColor(BRIGHT_YELLOW))
            p.drawRoundedRect(QRectF(45.0, 19.0, 6.0, 12.0), 3.0, 3.0)
        elif k == "DOCS":
            p.setPen(QPen(self.accent, 2.0))
            p.drawRoundedRect(QRectF(22.0, 14.0, 36.0, 48.0), 5.0, 5.0)
            p.setBrush(QColor(30, 41, 59))
            p.drawRoundedRect(QRectF(36.0, 24.0, 38.0, 46.0), 5.0, 5.0)
            p.setPen(QPen(QColor(TEXT_LIGHT), 1.6))
            p.drawLine(QPointF(44.0, 36.0), QPointF(66.0, 36.0))
            p.drawLine(QPointF(44.0, 45.0), QPointF(66.0, 45.0))
            p.drawLine(QPointF(44.0, 54.0), QPointF(60.0, 54.0))
        else:  # AI_LOCAL
            p.setPen(QPen(self.accent, 2.0))
            p.drawRoundedRect(QRectF(22.0, 18.0, 52.0, 48.0), 10.0, 10.0)
            p.setPen(QPen(QColor(BRIGHT_YELLOW), 2.2))
            p.drawEllipse(QPointF(38.0, 38.0), 4.0, 4.0)
            p.drawEllipse(QPointF(58.0, 38.0), 4.0, 4.0)
            p.drawLine(QPointF(38.0, 52.0), QPointF(58.0, 52.0))


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


class UserGuideDialog(QDialog):
    """Built-in interactive 4-tab User Guide & Gesture Cheat-Sheet (`[? Hướng Dẫn]`)."""

    def __init__(self, parent: Optional[QWidget] = None, initial_tab: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hướng Dẫn Sử Dụng Trực Quan — AirTeach AI")
        self.resize(840, 620)
        self.setStyleSheet(f"background: {BG_DARK}; color: {TEXT_LIGHT};")

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        hdr = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("HƯỚNG DẪN SỬ DỤNG & BẢNG TRA CỨU CỬ CHỈ AIRTEACH AI")
        title.setStyleSheet(f"color: {TEXT_LIGHT}; font-size: 16px; font-weight: 800;")
        sub = QLabel("Làm chủ bảng trắng không chạm, điều khiển chuột tròn 8 hướng, tài liệu đa định dạng & AI Offline")
        sub.setStyleSheet(f"color: {SECONDARY_TEAL}; font-size: 12px; font-weight: 600;")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        hdr.addLayout(title_col, stretch=1)

        close_btn = QPushButton("Đóng [Esc]")
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
            QTabBar::tab {{ background: {SURFACE_DARK}; color: {TEXT_MUTED}; padding: 9px 14px; font-weight: 700; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 4px; }}
            QTabBar::tab:selected {{ background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; }}
            """
        )

        # Tab 1: Cử Chỉ 1 Tay
        tab1_cards: List[QWidget] = [
            _make_guide_card(
                "POINTER",
                "1. Chỉ Trỏ (1 Ngón Trỏ Duỗi Thẳng)",
                "Giơ ngón trỏ để điều khiển con trỏ Laser 60Hz siêu mượt trên toàn bộ màn hình bảng trắng (vùng camera 12%–88% tự động ánh xạ đủ 4 góc).",
                NEON_CYAN,
            ),
            _make_guide_card(
                "PINCH_DRAW",
                "2. Kẹp Ngón Cái + Trỏ (Pinch < 0.32) để Vẽ",
                "Chạm nhẹ đầu ngón cái và ngón trỏ để bắt đầu nét vẽ. Nhả tay ra (> 0.42) để kết thúc — hệ thống tự động cắt đuôi 105ms chống nét thừa và làm mượt bằng Rust Catmull-Rom.",
                NEON_CYAN,
            ),
            _make_guide_card(
                "OPEN_ERASE",
                "3. Xòe Cả 5 Ngón để Tẩy Động",
                "Xòe rộng lòng bàn tay hướng vào camera để bật vòng tẩy động (bán kính tỷ lệ theo kích thước bàn tay) và quét qua nét mực cần xóa.",
                ALERT_ROSE,
            ),
            _make_guide_card(
                "SWIPE_H",
                "4. Vẫy Ngang Nhanh (Trái / Phải) Đổi Slide",
                "Vẫy dứt khoát sang Trái để qua Slide kế tiếp (Next) hoặc sang Phải để quay lại Slide trước (Prev). Hỗ trợ PPTX, PDF và Thư mục Ảnh.",
                BRIGHT_YELLOW,
            ),
            _make_guide_card(
                "SWIPE_V",
                "5. Vẫy Dọc / 2 Ngón Trượt để Cuộn Tài Liệu",
                "Vẫy lên/xuống hoặc dùng 2 ngón trỏ+giữa di chuyển lên xuống để cuộn mượt tài liệu Word (.docx) và PDF nhiều trang.",
                SECONDARY_TEAL,
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab1_cards), "1. Cử Chỉ 1 Tay")

        # Tab 2: Cử Chỉ 2 Tay
        tab2_cards: List[QWidget] = [
            _make_guide_card(
                "DUAL_ZOOM",
                "1. Thu / Phóng Bảng (2 Tay Kẹp Kéo Ra / Vào)",
                "Chụm cả hai tay rồi kéo rộng ra để Phóng to (tới 400%) hoặc thu hẹp lại để Thu nhỏ (tới 25%), lấy tâm chính xác tại điểm giữa hai bàn tay.",
                BRIGHT_YELLOW,
            ),
            _make_guide_card(
                "DUAL_PAN",
                "2. Di Chuyển Bảng Vô Tận (2 Tay Xòe hoặc Tay Trái Nắm + Tay Phải Kéo)",
                "Xòe cả hai lòng bàn tay (hoặc nắm nhẹ tay trái + di chuyển tay phải) rồi kéo theo bất kỳ hướng nào để trượt toàn bộ bảng trắng và tài liệu cực nhẹ nhàng.",
                SECONDARY_TEAL,
            ),
            _make_guide_card(
                "MOD_LOCK",
                "3. Khóa Nét Thẳng & Nắn Hình Chuẩn (2 Tay Phối Hợp)",
                "• Tay phụ NẮM + Tay chính KẸP VẼ: Khóa thành đoạn thẳng tuyệt đối.\n"
                "• Tay phụ GIƠ 2 NGÓN + Tay chính KẸP VẼ: Tự động chuẩn hóa hình tròn, elip, tam giác, chữ nhật, mũi tên.\n"
                "• Tay phụ GIƠ 3 NGÓN: Hoàn tác (Undo) tức thì.",
                "#10B981",
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab2_cards), "2. Cử Chỉ 2 Tay")

        # Tab 3: Chuột & Bảng Tròn
        pie_grid_frame = QFrame()
        pie_grid_frame.setStyleSheet(
            f"background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px;"
        )
        pg_vbox = QVBoxLayout(pie_grid_frame)
        pg_vbox.setContentsMargins(12, 10, 12, 10)
        pg_title = QLabel("Bố cục 8 Hướng Bảng Tròn Chuột Phải (Nhấn giữ chuột phải → Kéo về hướng → Thả)")
        pg_title.setStyleSheet(f"color: {NEON_CYAN}; font-size: 13px; font-weight: 800; border: none;")
        pg_vbox.addWidget(pg_title)
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, sl in enumerate(PIE_MENU_SLICES):
            chip = QLabel(f"[{sl.direction}] {sl.glyph} {sl.full_label_vi}")
            chip.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {sl.accent_hex}; border-radius: 6px; padding: 6px 8px; font-size: 11.5px; font-weight: 700;"
            )
            grid.addWidget(chip, i // 2, i % 2)
        pg_vbox.addLayout(grid)

        tab3_cards: List[QWidget] = [
            _make_guide_card(
                "PIE_MENU",
                "Bảng Tròn Chuột Phải (Blender/Maya Radial Pie Menu)",
                "Nhấn giữ Chuột Phải ở bất kỳ vị trí nào trên bảng để mở vòng tròn 8 công cụ quanh con trỏ. Kéo sang hướng tương ứng rồi thả chuột phải để kích hoạt ngay; giữ trong tâm (< 20px) để Hủy.",
                NEON_CYAN,
            ),
            pie_grid_frame,
            _make_guide_card(
                "MOUSE_NAV",
                "Điều Hướng Bảng Siêu Mượt Bằng Chuột & Phím Tắt",
                "• Kéo Chuột Giữa (hoặc giữ Space + Chuột Trái): Di chuyển bảng mọi hướng.\n"
                "• Cuộn Chuột: Phóng to / Thu nhỏ tại con trỏ (Ctrl + Cuộn khi ở chế độ Cuộn Dọc Word).\n"
                "• Shift + Cuộn Chuột: Cuộn ngang bảng.\n"
                "• Ctrl + 0 (hoặc nút [Về Giữa]): Đưa bảng về 100% chính giữa ngay lập tức.",
                SECONDARY_TEAL,
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab3_cards), "3. Chuột & Bảng Tròn")

        # Tab 4: AI Local & Tài Liệu
        tab4_cards: List[QWidget] = [
            _make_guide_card(
                "AI_LOCAL",
                "1. Trợ Lý Sư Phạm & Quản Lý AI Local (100% Offline)",
                "• Mở [Quản Lý AI Local] để tải trực tiếp các mô hình GGUF gọn nhẹ (Qwen2.5-1.5B/3B, Phi-3.5-mini) từ HuggingFace/GitHub với tiến độ MB/s & kiểm tra SHA-256.\n"
                "• Dán bất kỳ link .gguf từ GitHub/HuggingFace để tải về %APPDATA%/AirTeachAI/models.\n"
                "• Chuyển linh hoạt 3 chế độ: Local AI (Offline) | Gemini Cloud AI | Auto Hybrid.",
                "#A855F7",
            ),
            _make_guide_card(
                "DOCS",
                "2. Mở Word / PDF / PowerPoint / Ảnh / Thư Mục Ảnh",
                "• Hỗ trợ .PDF, .PPTX/.PPT, .DOCX/.DOC (cuộn dọc liên tục), và Ảnh (.PNG/.JPG/.WEBP/.BMP/.SVG).\n"
                "• Mở cả Thư mục Ảnh làm bộ Slide trình chiếu, hoặc Chèn Ảnh trực tiếp lên bảng trắng (kéo để di chuyển, kéo góc dưới phải để phóng to/thu nhỏ).\n"
                "• Nét vẽ lưu riêng từng trang và xuất trọn vẹn ra [Xuất Bài Giảng (.PDF)].",
                NEON_CYAN,
            ),
        ]
        self.tabs.addTab(_wrap_scroll(tab4_cards), "4. AI Local & Tài Liệu")

        self.tabs.setCurrentIndex(max(0, min(3, int(initial_tab))))
        root.addWidget(self.tabs, stretch=1)
