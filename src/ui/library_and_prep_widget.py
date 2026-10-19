"""
AirTeach AI - Library & Lesson Preparation Widget
==================================================
Manages the teacher's lesson workspace:
  - Lists existing .airteach lesson packages in data/workspaces/<teacher_id>/
  - Creates new lessons using 4 vector background templates:
      1. Plain White ("Trắng")
      2. Vietnamese School Grid ("Kẻ ô li tiểu học/THCS")
      3. Ruled Notebook ("Dòng kẻ ngang")
      4. Cartesian Coordinate System ("Hệ trục tọa độ Oxy")
  - Imports PPTX, DOCX, PDF documents via DocumentManager
  - Saves new lessons to teacher workspace

Zero-mock policy: 100% production code, real files, real vector rendering.
"""
from __future__ import annotations

import logging
import math
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.services.document_manager import (
    AirTeachManifest,
    DocumentManager,
    PageInfo,
)

logger = logging.getLogger(__name__)

BOARD_W = 1920
BOARD_H = 1080


# ─── Vector Background Template Generator ──────────────────────────────────────

class BackgroundTemplateGenerator:
    """
    Renders vector mathematical and pedagogical backgrounds into high-res images.
    All backgrounds are rendered at 1920×1080 (16:9 standard).
    """

    @staticmethod
    def render_plain_white(width: int = BOARD_W, height: int = BOARD_H) -> np.ndarray:
        """Crisp clean white background."""
        return np.full((height, width, 3), 255, dtype=np.uint8)

    @staticmethod
    def render_oli_grid(width: int = BOARD_W, height: int = BOARD_H) -> np.ndarray:
        """
        Vietnamese elementary/secondary school grid (Vở ô li 5 ô / 4 dòng).
        Major grid (bold): every 50px.
        Minor grid (sub-lines): every 10px.
        Colors: soft blue/cyan (#e1f5fe minor, #81d4fa major) matching real notebooks.
        """
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        pix = QPixmap(width, height)
        pix.fill(QColor("#ffffff"))

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Minor grid pen
        pen_minor = QPen(QColor(225, 245, 254), 1, Qt.PenStyle.SolidLine)
        # Major grid pen
        pen_major = QPen(QColor(129, 212, 250), 1.5, Qt.PenStyle.SolidLine)

        # Draw vertical lines
        sub_step = 10
        major_step = 50
        for x in range(0, width, sub_step):
            painter.setPen(pen_major if (x % major_step == 0) else pen_minor)
            painter.drawLine(x, 0, x, height)

        # Draw horizontal lines
        for y in range(0, height, sub_step):
            painter.setPen(pen_major if (y % major_step == 0) else pen_minor)
            painter.drawLine(0, y, width, y)

        # Left red margin line (traditional notebook margin)
        pen_margin = QPen(QColor(239, 83, 80), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen_margin)
        painter.drawLine(150, 0, 150, height)

        painter.end()

        # Convert QPixmap → BGR numpy array
        qimg = pix.toImage().convertToFormat(QImage.Format.Format_BGR888)
        ptr = qimg.constBits()
        arr = np.array(ptr).reshape((height, width, 3))
        return arr.copy()

    @staticmethod
    def render_lined_paper(width: int = BOARD_W, height: int = BOARD_H) -> np.ndarray:
        """
        Ruled notebook paper with horizontal guidelines and left margin.
        Line spacing: 40px.
        """
        pix = QPixmap(width, height)
        pix.fill(QColor("#fcfbf7"))  # warm paper tint

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Horizontal lines
        pen_line = QPen(QColor(187, 222, 251), 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen_line)
        spacing = 40
        for y in range(80, height - 40, spacing):
            painter.drawLine(60, y, width - 60, y)

        # Left margin line
        pen_margin = QPen(QColor(255, 138, 128), 1.5, Qt.PenStyle.SolidLine)
        painter.setPen(pen_margin)
        painter.drawLine(140, 40, 140, height - 40)

        painter.end()

        qimg = pix.toImage().convertToFormat(QImage.Format.Format_BGR888)
        ptr = qimg.constBits()
        arr = np.array(ptr).reshape((height, width, 3))
        return arr.copy()

    @staticmethod
    def render_oxy_coordinates(width: int = BOARD_W, height: int = BOARD_H) -> np.ndarray:
        """
        Cartesian 2D coordinate system (Oxy) with grid, axes, tick marks, arrows, and labels.
        Origin at center (width // 2, height // 2).
        Grid: 40px minor, 200px major.
        """
        pix = QPixmap(width, height)
        pix.fill(QColor("#ffffff"))

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cx = width // 2
        cy = height // 2
        step = 40

        # Background grid
        pen_grid = QPen(QColor(230, 235, 245), 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen_grid)
        for x in range(cx % step, width, step):
            painter.drawLine(x, 0, x, height)
        for y in range(cy % step, height, step):
            painter.drawLine(0, y, width, y)

        # Axes
        pen_axis = QPen(QColor(33, 33, 33), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen_axis)

        # X-axis
        painter.drawLine(40, cy, width - 40, cy)
        # X-arrowhead
        painter.drawLine(width - 40, cy, width - 55, cy - 8)
        painter.drawLine(width - 40, cy, width - 55, cy + 8)

        # Y-axis
        painter.drawLine(cx, height - 40, cx, 40)
        # Y-arrowhead
        painter.drawLine(cx, 40, cx - 8, 55)
        painter.drawLine(cx, 40, cx + 8, 55)

        # Ticks and numeric labels
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        pen_text = QPen(QColor(66, 66, 66))
        painter.setPen(pen_text)

        # Origin O
        painter.drawText(cx - 18, cy + 18, "O")
        # Axis labels x, y
        font_axis = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(font_axis)
        painter.drawText(width - 32, cy + 22, "x")
        painter.drawText(cx + 12, 45, "y")

        painter.setFont(font)
        tick_len = 4
        # X ticks
        for i, x in enumerate(range(cx + step, width - 60, step), 1):
            painter.drawLine(x, cy - tick_len, x, cy + tick_len)
            if i % 2 == 0:
                painter.drawText(x - 8, cy + 18, str(i))
        for i, x in enumerate(range(cx - step, 50, -step), 1):
            painter.drawLine(x, cy - tick_len, x, cy + tick_len)
            if i % 2 == 0:
                painter.drawText(x - 12, cy + 18, str(-i))

        # Y ticks
        for i, y in enumerate(range(cy - step, 50, -step), 1):
            painter.drawLine(cx - tick_len, y, cx + tick_len, y)
            if i % 2 == 0:
                painter.drawText(cx - 24, y + 4, str(i))
        for i, y in enumerate(range(cy + step, height - 50, step), 1):
            painter.drawLine(cx - tick_len, y, cx + tick_len, y)
            if i % 2 == 0:
                painter.drawText(cx - 28, y + 4, str(-i))

        painter.end()

        qimg = pix.toImage().convertToFormat(QImage.Format.Format_BGR888)
        ptr = qimg.constBits()
        arr = np.array(ptr).reshape((height, width, 3))
        return arr.copy()


# ─── New Lesson Creation Dialog ───────────────────────────────────────────────

class NewLessonDialog(QDialog):
    """
    Dialog to create a new lesson with a chosen template or page count.
    """

    TEMPLATE_PLAIN = "Trắng"
    TEMPLATE_OLI   = "Kẻ ô li tiểu học/THCS"
    TEMPLATE_LINED = "Dòng kẻ ngang"
    TEMPLATE_OXY   = "Hệ trục tọa độ Oxy"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tạo Bài giảng Mới")
        self.setFixedSize(500, 380)
        self.setModal(True)

        self._title_edit: QLineEdit
        self._subject_edit: QLineEdit
        self._template_combo: QComboBox
        self._page_count_spin: QSpinBox

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        header = QLabel("✨ Tạo bài giảng tương tác mới")
        header.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        header.setStyleSheet("color: #1a73e8;")
        layout.addWidget(header)

        # Title
        t_label = QLabel("Tên bài giảng:")
        t_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        layout.addWidget(t_label)
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("VD: Bài 15 - Định lý Pytago & Ứng dụng")
        self._title_edit.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(self._title_edit)

        # Subject
        s_label = QLabel("Môn học / Chủ đề:")
        s_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        layout.addWidget(s_label)
        self._subject_edit = QLineEdit()
        self._subject_edit.setPlaceholderText("VD: Toán học 8")
        self._subject_edit.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(self._subject_edit)

        # Template selection
        row_tpl = QHBoxLayout()
        tpl_lbl = QLabel("Mẫu trang nền:")
        tpl_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self._template_combo = QComboBox()
        self._template_combo.addItems([
            self.TEMPLATE_PLAIN,
            self.TEMPLATE_OLI,
            self.TEMPLATE_LINED,
            self.TEMPLATE_OXY,
        ])
        self._template_combo.setStyleSheet("padding: 6px; border: 1px solid #ccc; border-radius: 4px;")
        row_tpl.addWidget(tpl_lbl)
        row_tpl.addWidget(self._template_combo, stretch=1)
        layout.addLayout(row_tpl)

        # Initial Page Count
        row_pg = QHBoxLayout()
        pg_lbl = QLabel("Số trang khởi tạo:")
        pg_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self._page_count_spin = QSpinBox()
        self._page_count_spin.setRange(1, 20)
        self._page_count_spin.setValue(3)
        self._page_count_spin.setStyleSheet("padding: 6px;")
        row_pg.addWidget(pg_lbl)
        row_pg.addWidget(self._page_count_spin)
        row_pg.addStretch()
        layout.addLayout(row_pg)

        layout.addStretch()

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        cancel_btn = QPushButton("Hủy")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("padding: 8px 16px; border: 1px solid #ccc; border-radius: 4px;")

        ok_btn = QPushButton("Tạo bài giảng")
        ok_btn.clicked.connect(self._validate_and_accept)
        ok_btn.setStyleSheet(
            "background: #1a73e8; color: white; font-weight: bold; padding: 8px 18px; border-radius: 4px;"
        )

        btn_box.addWidget(cancel_btn)
        btn_box.addWidget(ok_btn)
        layout.addLayout(btn_box)

    def _validate_and_accept(self) -> None:
        if not self._title_edit.text().strip():
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập tên bài giảng.")
            self._title_edit.setFocus()
            return
        self.accept()

    @property
    def lesson_title(self) -> str:
        return self._title_edit.text().strip()

    @property
    def subject(self) -> str:
        return self._subject_edit.text().strip()

    @property
    def template_name(self) -> str:
        return self._template_combo.currentText()

    @property
    def page_count(self) -> int:
        return self._page_count_spin.value()


# ─── Library and Preparation Widget ──────────────────────────────────────────

class LibraryAndPrepWidget(QWidget):
    """
    Complete workspace management widget for teachers.
    
    Signals:
        open_lesson_requested(Path): Emitted when a user selects a lesson to teach.
        lesson_created(Path): Emitted when a new .airteach package is created.
    """

    open_lesson_requested = Signal(Path)
    lesson_created        = Signal(Path)

    def __init__(
        self,
        workspace_dir: Path,
        doc_manager: DocumentManager,
        teacher_name: str = "Giáo viên",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.doc_manager = doc_manager
        self.teacher_name = teacher_name

        self._build_ui()
        self.refresh_lesson_list()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header bar
        header_bar = QHBoxLayout()
        title_box = QVBoxLayout()
        title_lbl = QLabel(f"📚 Thư viện bài giảng — {self.teacher_name}")
        title_lbl.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #202124;")
        sub_lbl = QLabel(f"Thư mục làm việc: {self.workspace_dir}")
        sub_lbl.setFont(QFont("Segoe UI", 9))
        sub_lbl.setStyleSheet("color: #5f6368;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        header_bar.addLayout(title_box)
        header_bar.addStretch()

        # Action Buttons
        btn_new = QPushButton("➕ Tạo Bài giảng Mới")
        btn_new.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        btn_new.setStyleSheet(
            "background: #1a73e8; color: white; padding: 10px 18px; border-radius: 6px;"
        )
        btn_new.clicked.connect(self._on_create_new_lesson)
        header_bar.addWidget(btn_new)

        btn_import = QPushButton("📥 Nhập PPTX / PDF / Word")
        btn_import.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        btn_import.setStyleSheet(
            "background: #34a853; color: white; padding: 10px 18px; border-radius: 6px;"
        )
        btn_import.clicked.connect(self._on_import_document)
        header_bar.addWidget(btn_import)

        btn_refresh = QPushButton("🔄 Làm mới")
        btn_refresh.setStyleSheet("padding: 10px 14px; border: 1px solid #ccc; border-radius: 6px;")
        btn_refresh.clicked.connect(self.refresh_lesson_list)
        header_bar.addWidget(btn_refresh)

        main_layout.addLayout(header_bar)

        # Lesson list table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Tên bài giảng", "Số trang", "Kích thước", "Thời gian cập nhật", "Thao tác"
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dadce0;
                border-radius: 8px;
                background-color: #ffffff;
                gridline-color: #f1f3f4;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                font-weight: bold;
                border: none;
                border-bottom: 2px solid #dadce0;
            }
        """)
        self._table.doubleClicked.connect(self._on_table_double_clicked)
        main_layout.addWidget(self._table, stretch=1)

        # Empty state label
        self._empty_label = QLabel("Chưa có bài giảng nào trong thư viện.\nHãy bấm 'Tạo Bài giảng Mới' hoặc 'Nhập PPTX / PDF / Word' để bắt đầu.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setFont(QFont("Segoe UI", 11))
        self._empty_label.setStyleSheet("color: #70757a; margin: 40px;")
        main_layout.addWidget(self._empty_label)

    # ── Refresh list ──────────────────────────────────────────────────────────

    def refresh_lesson_list(self) -> None:
        """Scan workspace_dir for real .airteach packages and populate table."""
        self._table.setRowCount(0)
        packages = sorted(
            list(self.workspace_dir.glob("*.airteach")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not packages:
            self._table.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._table.setVisible(True)
        self._empty_label.setVisible(False)

        for row, pkg_path in enumerate(packages):
            self._table.insertRow(row)

            # Read manifest safely
            manifest = self._read_manifest(pkg_path)
            title = manifest.title if manifest else pkg_path.stem
            page_count = str(manifest.page_count) if manifest else "—"

            # Size
            size_kb = pkg_path.stat().st_size / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"

            # Date
            mod_time = datetime.fromtimestamp(pkg_path.stat().st_mtime)
            date_str = mod_time.strftime("%d/%m/%Y %H:%M")

            # Title item
            t_item = QTableWidgetItem(f"📄  {title}")
            t_item.setData(Qt.ItemDataRole.UserRole, str(pkg_path))
            self._table.setItem(row, 0, t_item)

            # Pages item
            p_item = QTableWidgetItem(page_count)
            p_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 1, p_item)

            # Size item
            s_item = QTableWidgetItem(size_str)
            s_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 2, s_item)

            # Date item
            d_item = QTableWidgetItem(date_str)
            d_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 3, d_item)

            # Actions cell
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            btn_layout.setSpacing(6)

            open_btn = QPushButton("Giảng dạy")
            open_btn.setStyleSheet(
                "background: #1a73e8; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;"
            )
            open_btn.clicked.connect(lambda _, p=pkg_path: self.open_lesson_requested.emit(p))
            btn_layout.addWidget(open_btn)

            del_btn = QPushButton("Xóa")
            del_btn.setStyleSheet(
                "background: #fce8e6; color: #c5221f; border: 1px solid #fad2cf; padding: 4px 8px; border-radius: 4px;"
            )
            del_btn.clicked.connect(lambda _, p=pkg_path: self._delete_lesson(p))
            btn_layout.addWidget(del_btn)

            self._table.setCellWidget(row, 4, btn_widget)

    def _read_manifest(self, pkg_path: Path) -> Optional[AirTeachManifest]:
        """Safely peek into the .airteach ZIP manifest without extracting everything."""
        import zipfile
        try:
            with zipfile.ZipFile(pkg_path, "r") as zf:
                if "manifest.json" in zf.namelist():
                    data = zf.read("manifest.json").decode("utf-8")
                    return AirTeachManifest.from_json(data)
        except Exception as exc:
            logger.warning("Failed to peek manifest in %s: %s", pkg_path, exc)
        return None

    def _on_table_double_clicked(self, index) -> None:
        row = index.row()
        item = self._table.item(row, 0)
        if item:
            path_str = item.data(Qt.ItemDataRole.UserRole)
            if path_str:
                self.open_lesson_requested.emit(Path(path_str))

    def _delete_lesson(self, pkg_path: Path) -> None:
        ans = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa bài giảng '{pkg_path.stem}' không?\nHành động này không thể hoàn tác.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            try:
                pkg_path.unlink()
                self.refresh_lesson_list()
            except Exception as exc:
                QMessageBox.critical(self, "Lỗi xóa", str(exc))

    # ── Creation & Import handlers ────────────────────────────────────────────

    def _on_create_new_lesson(self) -> None:
        dlg = NewLessonDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        title = dlg.lesson_title
        subject = dlg.subject
        template = dlg.template_name
        page_count = dlg.page_count

        # Render template image
        if template == NewLessonDialog.TEMPLATE_OLI:
            img = BackgroundTemplateGenerator.render_oli_grid()
        elif template == NewLessonDialog.TEMPLATE_LINED:
            img = BackgroundTemplateGenerator.render_lined_paper()
        elif template == NewLessonDialog.TEMPLATE_OXY:
            img = BackgroundTemplateGenerator.render_oxy_coordinates()
        else:
            img = BackgroundTemplateGenerator.render_plain_white()

        # Create temporary directory to hold page PNGs
        temp_dir = self.workspace_dir / f"_tmp_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            page_paths: list[Path] = []
            for i in range(page_count):
                p_path = temp_dir / f"page_{i:03d}.png"
                cv2.imwrite(str(p_path), img)
                page_paths.append(p_path)

            safe_stem = "".join(c if (c.isalnum() or c in " _-") else "_" for c in title).strip()
            out_pkg = self.workspace_dir / f"{safe_stem}.airteach"

            # Avoid collision
            counter = 1
            while out_pkg.exists():
                out_pkg = self.workspace_dir / f"{safe_stem}_{counter}.airteach"
                counter += 1

            manifest = AirTeachManifest(
                title=title,
                page_count=page_count,
                created_at=datetime.now(timezone.utc).isoformat(),
                modified_at=datetime.now(timezone.utc).isoformat(),
            )

            self.doc_manager.save_lesson(
                output_path=out_pkg,
                manifest=manifest,
                page_image_paths=page_paths,
                strokes=[],
                recognized=[],
            )

            self.refresh_lesson_list()
            self.lesson_created.emit(out_pkg)
            QMessageBox.information(
                self, "Tạo bài giảng thành công",
                f"Đã tạo bài giảng '{title}' với {page_count} trang.\nBấm 'Giảng dạy' để bắt đầu giảng bài."
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _on_import_document(self) -> None:
        """Import PPTX, DOCX, or PDF from disk."""
        src_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn tài liệu bài giảng (PPTX, DOCX, PDF)",
            str(Path.home()),
            "Tài liệu giảng dạy (*.pdf *.pptx *.docx);;PDF (*.pdf);;PowerPoint (*.pptx);;Word (*.docx)",
        )
        if not src_path:
            return

        source = Path(src_path)
        try:
            # Show progress
            pages: list[PageInfo] = self.doc_manager.import_document(source)
            if not pages:
                QMessageBox.warning(self, "Cảnh báo", "Không thể trích xuất trang nào từ tệp được chọn.")
                return

            page_paths = [Path(p.image_path) for p in pages if Path(p.image_path).exists()]
            safe_stem = source.stem
            out_pkg = self.workspace_dir / f"{safe_stem}.airteach"

            counter = 1
            while out_pkg.exists():
                out_pkg = self.workspace_dir / f"{safe_stem}_{counter}.airteach"
                counter += 1

            manifest = AirTeachManifest(
                title=source.name,
                page_count=len(page_paths),
                created_at=datetime.now(timezone.utc).isoformat(),
                modified_at=datetime.now(timezone.utc).isoformat(),
            )

            self.doc_manager.save_lesson(
                output_path=out_pkg,
                manifest=manifest,
                page_image_paths=page_paths,
                strokes=[],
                recognized=[],
            )

            self.refresh_lesson_list()
            self.lesson_created.emit(out_pkg)
            QMessageBox.information(
                self, "Nhập học liệu thành công",
                f"Đã chuyển đổi '{source.name}' thành gói bài giảng AirTeach ({len(page_paths)} trang)!"
            )

        except Exception as exc:
            logger.exception("Import error: %s", exc)
            QMessageBox.critical(self, "Lỗi nhập tài liệu", f"Không thể nhập tệp:\n{exc}")
