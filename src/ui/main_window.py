"""
AirTeach AI - Main Window
==========================
QMainWindow integrating all subsystems:

Layout (horizontal split):
  ┌─────────────────────────────────────────┬──────────────┐
  │  Toolbar (gesture mode, page nav, OCR)  │              │
  ├─────────────────────────┬───────────────┤  Chat Panel  │
  │   LessonBoard           │  Camera       │  (collapsible│
  │   (QGraphicsView)       │  Widget       │  right dock) │
  │                         │  + Gesture    │              │
  │                         │  HUD          │              │
  └─────────────────────────┴───────────────┴──────────────┘
  │          Status Bar (FPS, mode, page number)           │

Signal flow:
  HandTrackingWorker → (gesture signals) → MainWindow slots
    → LessonBoard.begin_stroke_preview / update / commit
    → SmartInkRecognizer (background) → add_text/latex_overlay
    → ChatPanel stream
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QRunnable,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

def _get_icon(name: str) -> QIcon:
    path = _ASSETS_DIR / "icons" / f"{name}.svg"
    return QIcon(str(path)) if path.exists() else QIcon()

from src.core.hand_gesture_engine import (
    GestureState,
    HandTrackingWorker,
    PerspectiveMapper,
)
from src.core.stroke_beautifier import (
    BezierSegment,
    GeometricShapeSnapper,
    OneEuroFilter2D,
    SnappedShape as SnapResult,
    ShapeType,
    fit_cubic_bezier_spline,
    ramer_douglas_peucker,
)
from src.services.document_manager import DocumentManager
from src.services.gemini_assistant import GeminiAssistant
from src.services.smart_ink_ocr import SmartInkRecognizer, Stroke, StrokePoint
from src.ui.camera_widget import CameraWidget
from src.ui.chat_panel import ChatPanel
from src.ui.lesson_board import InkStroke, LessonBoard, BOARD_WIDTH, BOARD_HEIGHT
from src.ui.replay_controller import ReplayController
from src.ui.settings_dialog import SettingsDialog
from src.utils.telemetry_logger import TelemetryLogger

logger = logging.getLogger(__name__)


# ── Qt bridge for callback-based HandTrackingWorker ───────────────────────────

class _QtCameraWorker(QObject):
    """
    QObject wrapper around HandTrackingWorker.
    HandTrackingWorker uses plain Python callbacks; this class converts
    those callbacks into Qt Signals so UI slots can safely receive them.
    """

    frame_ready    = Signal(np.ndarray)          # BGR frame
    gesture_event  = Signal(GestureState, float, float)  # (state, cx_norm, cy_norm)
    error_occurred = Signal(str)

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._worker = HandTrackingWorker(
            camera_index=camera_index,
            on_frame=self._on_frame_cb,
            on_error=self._on_error_cb,
        )

    def _on_frame_cb(self, frame_bgr, landmarks, state: GestureState, cx: float, cy: float) -> None:
        self.frame_ready.emit(frame_bgr)
        if state != GestureState.NONE:
            self.gesture_event.emit(state, cx, cy)

    def _on_error_cb(self, msg: str) -> None:
        self.error_occurred.emit(msg)

    def start_capture(self) -> None:
        """Called when QThread starts — runs the blocking camera loop."""
        self._worker.start()

    def stop(self) -> None:
        self._worker.stop()

    def start_perspective_calibration(self) -> None:
        """Reset perspective mapping to identity (full-frame → full-canvas)."""
        identity_corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
        )
        self._worker.fsm.perspective.calibrate(identity_corners)


# ── OCR background worker ──────────────────────────────────────────────────────

class _OcrRunnable(QRunnable):
    """Runs SmartInkRecognizer on a background thread pool."""

    class _Signals(QObject):
        done = Signal(object)   # OcrResult

    def __init__(
        self,
        recognizer: SmartInkRecognizer,
        strokes: list[Stroke],
        board_x: float,
        board_y: float,
        force_formula: bool = False,
    ) -> None:
        super().__init__()
        self.signals      = _OcrRunnable._Signals()
        self._recognizer  = recognizer
        self._strokes     = strokes
        self._x           = board_x
        self._y           = board_y
        self._formula     = force_formula

    def run(self) -> None:
        result = self._recognizer.recognize_strokes(
            self._strokes, force_formula=self._formula
        )
        result._board_x = self._x   # type: ignore[attr-defined]
        result._board_y = self._y   # type: ignore[attr-defined]
        self.signals.done.emit(result)


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """AirTeach AI main application window."""

    logout_requested = Signal()

    def __init__(
        self,
        teacher_row=None,           # sqlite3.Row from face auth (optional)
        camera_index: int = 0,
    ) -> None:
        super().__init__()
        if teacher_row:
            from src.services.face_profile_service import TeacherProfileService  # noqa: PLC0415
            self._teacher = TeacherProfileService.teacher_row_to_dict(teacher_row)
        else:
            self._teacher = None
        self._cam_index = camera_index

        # State
        self._current_stroke_id: Optional[str] = None
        self._live_raw_points: list[tuple[float, float, float]] = []  # (x, y, t_ms)
        self._filter = OneEuroFilter2D()
        self._snapper = GeometricShapeSnapper()
        self._current_page_index = 0
        self._page_strokes: dict[int, list[InkStroke]] = {}
        self._page_texts: list[str] = []
        self._ink_color = QColor("#202124")
        self._ink_min_width = 1.5
        self._ink_max_width = 6.0
        self._rdp_epsilon   = 2.5
        self._snapping_on   = True
        self._select_start: Optional[QPointF] = None
        self._teaching_mode: str = "PEN"  # "PEN", "LASER", "ERASER", "SELECT"
        self._redo_stack: list[InkStroke] = []

        # Services (lazy-init)
        _work_root = Path(__file__).resolve().parents[2] / "data" / "work"
        _work_root.mkdir(parents=True, exist_ok=True)
        self._doc_manager  = DocumentManager(_work_root)
        if self._teacher and self._teacher.get("workspace_dir"):
            self._workspace_dir = Path(self._teacher["workspace_dir"])
        else:
            self._workspace_dir = Path(__file__).resolve().parents[2] / "data" / "workspaces" / "demo"
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        _api_key = os.environ.get("GEMINI_API_KEY", "")
        _model   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._assistant    = GeminiAssistant(api_key=_api_key, model_name=_model)
        self._recognizer   = SmartInkRecognizer()
        _tel_path = Path(__file__).resolve().parents[2] / "data" / "telemetry.csv"
        _tel_path.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry    = TelemetryLogger(_tel_path)
        self._telemetry.start()
        self._thread_pool  = QThreadPool.globalInstance()

        # Page cache (list[PageInfo])
        self._pages: list = []

        # Camera worker
        self._cam_thread: Optional[QThread] = None
        self._cam_worker: Optional[_QtCameraWorker] = None

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._connect_camera(camera_index)

        # Initialize default crisp canvas background
        from src.ui.library_and_prep_widget import BackgroundTemplateGenerator  # noqa: PLC0415
        self._scene.set_background_page(BackgroundTemplateGenerator.render_plain_white())

        # Welcome state
        code_str = f"[{self._teacher['teacher_code']}] " if self._teacher and self._teacher.get("teacher_code") else ""
        t_name = self._teacher.get("full_name") if self._teacher else "Dạy thử / Demo"
        self.setWindowTitle(f"AirTeach AI — {code_str}{t_name}")
        self._update_status_bar()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_h = QHBoxLayout(central)
        root_h.setContentsMargins(0, 0, 0, 0)
        root_h.setSpacing(0)

        # ── Left: board + camera stack ─────────────────────────────────────
        left_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Board view container
        board_container = QWidget()
        board_layout = QVBoxLayout(board_container)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(0)

        self._scene = LessonBoard(self)
        self._scene.stroke_committed.connect(self._on_stroke_committed)

        self._board_view = QGraphicsView(self._scene)
        from PySide6.QtGui import QPainter
        self._board_view.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self._board_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._board_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._board_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._board_view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._board_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        board_layout.addWidget(self._board_view, stretch=1)

        # Stroke Replay Controller (collapsible)
        self._replay_ctrl = ReplayController(parent=self)
        self._replay_ctrl.set_canvas(self._scene)
        self._replay_ctrl.setVisible(False)
        board_layout.addWidget(self._replay_ctrl)

        left_splitter.addWidget(board_container)

        # Camera + HUD panel
        cam_panel = QWidget()
        cam_layout = QVBoxLayout(cam_panel)
        cam_layout.setContentsMargins(4, 4, 4, 4)
        cam_layout.setSpacing(6)

        self._cam_widget = CameraWidget()
        cam_layout.addWidget(self._cam_widget)

        self._gesture_hud = QLabel("Mode: —")
        self._gesture_hud.setFont(QFont("Consolas", 10))
        self._gesture_hud.setStyleSheet(
            "background:#1a1a2e; color:#00ff88; padding:4px 8px; border-radius:4px;"
        )
        cam_layout.addWidget(self._gesture_hud)

        self._fps_label = QLabel("FPS: —")
        self._fps_label.setStyleSheet("color:#8d99ae; font-size:10px;")
        cam_layout.addWidget(self._fps_label)
        cam_panel.setFixedWidth(340)
        left_splitter.addWidget(cam_panel)

        root_h.addWidget(left_splitter, stretch=1)

        # ── Right: chat panel (dock) ───────────────────────────────────────
        self._chat_panel = ChatPanel(self._assistant, self)
        self._chat_panel.request_canvas_snapshot.connect(self._on_snapshot_request)
        chat_dock = QDockWidget("AI Assistant", self)
        chat_dock.setWidget(self._chat_panel)
        chat_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, chat_dock)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_mode  = QLabel("Mode: IDLE")
        self._status_page  = QLabel("Trang: —")
        self._status_fps   = QLabel("FPS: —")
        for lbl in (self._status_mode, self._status_page, self._status_fps):
            lbl.setStyleSheet("padding: 0 8px;")
            self._status.addPermanentWidget(lbl)

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("Tệp")
        lib_act = QAction(_get_icon("library"), "Thư viện bài giảng…", self)
        lib_act.triggered.connect(self._on_open_library)
        file_menu.addAction(lib_act)

        open_act = QAction(_get_icon("import_doc"), "Mở tài liệu (PDF, PPTX, DOCX)…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_document)
        file_menu.addAction(open_act)

        open_at_act = QAction(_get_icon("new_lesson"), "Mở gói bài giảng .airteach…", self)
        open_at_act.triggered.connect(self._on_open_airteach)
        file_menu.addAction(open_at_act)

        save_at_act = QAction(_get_icon("new_lesson"), "Lưu bài giảng (.airteach)…", self)
        save_at_act.setShortcut(QKeySequence.StandardKey.Save)
        save_at_act.triggered.connect(self._on_save_airteach)
        file_menu.addAction(save_at_act)

        file_menu.addSeparator()
        export_act = QAction(_get_icon("export_pdf"), "Xuất PDF có chú thích…", self)
        export_act.triggered.connect(self._on_export_pdf)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        quit_act = QAction("Thoát", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(quit_act)

        # Edit
        edit_menu = mb.addMenu("Chỉnh sửa")
        undo_act = QAction(_get_icon("undo"), "Hoàn tác nét cuối", self)
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(self._on_undo)
        edit_menu.addAction(undo_act)

        redo_act = QAction(_get_icon("redo"), "Làm lại", self)
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        redo_act.triggered.connect(self._on_redo)
        edit_menu.addAction(redo_act)

        clear_act = QAction(_get_icon("clear"), "Xóa tất cả nét", self)
        clear_act.triggered.connect(self._on_clear_all)
        edit_menu.addAction(clear_act)

        replay_act = QAction(_get_icon("replay"), "Phát lại nét vẽ (Replay)", self)
        replay_act.triggered.connect(self._on_toggle_replay)
        edit_menu.addAction(replay_act)

        # Tools
        tools_menu = mb.addMenu("Công cụ")
        trainer_act = QAction(_get_icon("camera"), "Luyện tập & Hiệu chuẩn Cử chỉ…", self)
        trainer_act.setShortcut(Qt.Key.Key_T)
        trainer_act.triggered.connect(self._on_open_gesture_trainer)
        tools_menu.addAction(trainer_act)

        settings_act = QAction(_get_icon("settings"), "Cài đặt…", self)
        settings_act.triggered.connect(self._on_settings)
        tools_menu.addAction(settings_act)

        download_act = QAction(_get_icon("camera"), "Tải mô hình AI…", self)
        download_act.triggered.connect(self._on_download_models)
        tools_menu.addAction(download_act)

    # ── Toolbars ──────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        # 1. Navigation & Document Toolbar
        nav_tb = QToolBar("Điều hướng Bài giảng")
        nav_tb.setMovable(False)
        from PySide6.QtCore import QSize
        nav_tb.setIconSize(QSize(20, 20))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, nav_tb)

        lib_tb_act = QAction(_get_icon("library"), "📂 Kho Bài giảng", self)
        lib_tb_act.setToolTip("Mở kho bài giảng cá nhân của giáo viên (.airteach)")
        lib_tb_act.triggered.connect(self._on_open_library)
        nav_tb.addAction(lib_tb_act)

        open_tb_act = QAction(_get_icon("import_doc"), "Mở tài liệu", self)
        open_tb_act.setToolTip("Nhập tệp PDF, PPTX hoặc Word DOCX")
        open_tb_act.triggered.connect(self._on_open_document)
        nav_tb.addAction(open_tb_act)

        save_tb_act = QAction(_get_icon("new_lesson"), "Lưu", self)
        save_tb_act.setToolTip("Lưu gói bài giảng .airteach")
        save_tb_act.triggered.connect(self._on_save_airteach)
        nav_tb.addAction(save_tb_act)

        export_tb_act = QAction(_get_icon("export_pdf"), "Xuất PDF", self)
        export_tb_act.setToolTip("Xuất bài giảng kèm nét vẽ sang tệp PDF")
        export_tb_act.triggered.connect(self._on_export_pdf)
        nav_tb.addAction(export_tb_act)

        nav_tb.addSeparator()

        prev_act = QAction(_get_icon("page_prev"), "Trang trước", self)
        prev_act.setShortcut(Qt.Key.Key_Left)
        prev_act.setToolTip("Lùi lại trang trước (Mũi tên Trái hoặc Gạt trái)")
        prev_act.triggered.connect(self._on_prev_page)
        nav_tb.addAction(prev_act)

        next_act = QAction(_get_icon("page_next"), "Trang sau", self)
        next_act.setShortcut(Qt.Key.Key_Right)
        next_act.setToolTip("Chuyển sang trang sau (Mũi tên Phải hoặc Gạt phải)")
        next_act.triggered.connect(self._on_next_page)
        nav_tb.addAction(next_act)

        nav_tb.addSeparator()

        ocr_act = QAction(_get_icon("ocr_text"), "Nhận dạng chữ", self)
        ocr_act.setToolTip("Nhận dạng chữ viết tay sang văn bản")
        ocr_act.triggered.connect(self._on_manual_ocr)
        nav_tb.addAction(ocr_act)

        formula_act = QAction(_get_icon("ocr_math"), "Nhận dạng công thức", self)
        formula_act.setToolTip("Nhận dạng công thức toán học viết tay sang LaTeX")
        formula_act.triggered.connect(self._on_manual_formula_ocr)
        nav_tb.addAction(formula_act)

        nav_tb.addSeparator()

        replay_tb_act = QAction(_get_icon("replay"), "Phát lại", self)
        replay_tb_act.setToolTip("Bật thanh tua lại quá trình viết bài giảng (Replay)")
        replay_tb_act.triggered.connect(self._on_toggle_replay)
        nav_tb.addAction(replay_tb_act)

        clear_tb_act = QAction(_get_icon("clear"), "Xóa nét", self)
        clear_tb_act.setToolTip("Xóa toàn bộ nét vẽ trên trang hiện tại")
        clear_tb_act.triggered.connect(self._on_clear_all)
        nav_tb.addAction(clear_tb_act)

        # Spacer to push profile badge and logout to right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        nav_tb.addWidget(spacer)

        code_tag = f"[{self._teacher['teacher_code']}] " if self._teacher and self._teacher.get("teacher_code") else ""
        teacher_name = self._teacher["full_name"] if self._teacher and "full_name" in self._teacher.keys() else "Dạy thử / Demo"
        subject_name = f" • {self._teacher['subject']}" if self._teacher and "subject" in self._teacher.keys() and self._teacher["subject"] else ""
        profile_pill = QLabel(f"👤 {code_tag}{teacher_name}{subject_name}")
        profile_pill.setStyleSheet("""
            background: #e8f0fe;
            color: #1a73e8;
            font-weight: bold;
            font-size: 11px;
            padding: 4px 12px;
            border-radius: 12px;
            border: 1px solid #d2e3fc;
            margin-right: 6px;
        """)
        self.teacher_badge = profile_pill
        nav_tb.addWidget(profile_pill)

        logout_act = QAction(_get_icon("settings"), "Đổi GV / Đăng xuất", self)
        logout_act.setToolTip("Đăng xuất tài khoản hiện tại và trở về màn hình đăng nhập")
        logout_act.triggered.connect(self._on_logout_requested)
        self.logout_action = logout_act
        nav_tb.addAction(logout_act)

        # 2. Classroom Canvas Teaching Toolbar
        canvas_tb = QToolBar("Công cụ Giảng dạy Bảng viết")
        canvas_tb.setMovable(False)
        canvas_tb.setIconSize(QSize(20, 20))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, canvas_tb)

        tool_grp = QActionGroup(self)
        tool_grp.setExclusive(True)

        pen_act = QAction(_get_icon("pen"), "Bút vẽ (P)", self)
        pen_act.setCheckable(True)
        pen_act.setChecked(True)
        pen_act.setShortcut(Qt.Key.Key_P)
        pen_act.triggered.connect(lambda: self._set_teaching_mode("PEN"))
        tool_grp.addAction(pen_act)
        canvas_tb.addAction(pen_act)

        laser_act = QAction(_get_icon("laser"), "Tia Laser (L)", self)
        laser_act.setCheckable(True)
        laser_act.setShortcut(Qt.Key.Key_L)
        laser_act.triggered.connect(lambda: self._set_teaching_mode("LASER"))
        tool_grp.addAction(laser_act)
        canvas_tb.addAction(laser_act)

        eraser_act = QAction(_get_icon("eraser"), "Tẩy nét (E)", self)
        eraser_act.setCheckable(True)
        eraser_act.setShortcut(Qt.Key.Key_E)
        eraser_act.triggered.connect(lambda: self._set_teaching_mode("ERASER"))
        tool_grp.addAction(eraser_act)
        canvas_tb.addAction(eraser_act)

        select_act = QAction(_get_icon("select"), "Chọn vùng (S)", self)
        select_act.setCheckable(True)
        select_act.setShortcut(Qt.Key.Key_S)
        select_act.triggered.connect(lambda: self._set_teaching_mode("SELECT"))
        tool_grp.addAction(select_act)
        canvas_tb.addAction(select_act)

        canvas_tb.addSeparator()

        snap_act = QAction(_get_icon("shape"), "Nắn hình học (G)", self)
        snap_act.setCheckable(True)
        snap_act.setChecked(self._snapping_on)
        snap_act.setShortcut(Qt.Key.Key_G)
        snap_act.triggered.connect(self._toggle_shape_snapping)
        canvas_tb.addAction(snap_act)

        canvas_tb.addSeparator()

        undo_tb_act = QAction(_get_icon("undo"), "Hoàn tác", self)
        undo_tb_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_tb_act.triggered.connect(self._on_undo)
        canvas_tb.addAction(undo_tb_act)

        redo_tb_act = QAction(_get_icon("redo"), "Làm lại", self)
        redo_tb_act.setShortcut(QKeySequence.StandardKey.Redo)
        redo_tb_act.triggered.connect(self._on_redo)
        canvas_tb.addAction(redo_tb_act)

        canvas_tb.addSeparator()

        # Preset Color buttons
        colors = [
            ("#202124", "Đen"),
            ("#1a73e8", "Xanh"),
            ("#d93025", "Đỏ"),
            ("#137333", "Xanh lá"),
            ("#8e24aa", "Tím"),
        ]
        color_container = QWidget()
        c_layout = QHBoxLayout(color_container)
        c_layout.setContentsMargins(4, 0, 4, 0)
        c_layout.setSpacing(4)
        for hex_code, tip in colors:
            btn = QPushButton()
            btn.setFixedSize(18, 18)
            btn.setToolTip(f"Màu mực: {tip}")
            btn.setStyleSheet(f"background: {hex_code}; border-radius: 9px; border: 1.5px solid #dadce0;")
            btn.clicked.connect(lambda _, c=hex_code: self._set_ink_color(c))
            c_layout.addWidget(btn)
        canvas_tb.addWidget(color_container)

        canvas_tb.addSeparator()

        # Width selector
        self._width_combo = QComboBox()
        self._width_combo.addItem("Mảnh 1.5px", 1.5)
        self._width_combo.addItem("Vừa 2.5px", 2.5)
        self._width_combo.addItem("Đậm 4.5px", 4.5)
        self._width_combo.addItem("Dạ quang 8.0px", 8.0)
        self._width_combo.setCurrentIndex(1)
        self._width_combo.currentIndexChanged.connect(self._on_width_changed)
        canvas_tb.addWidget(self._width_combo)

        canvas_tb.addSeparator()

        # Gesture Trainer trigger
        train_tb_act = QAction(_get_icon("camera"), "Hiệu chuẩn cử chỉ", self)
        train_tb_act.setToolTip("Mở cửa sổ Luyện tập cử chỉ và Hiệu chuẩn góc camera")
        train_tb_act.triggered.connect(self._on_open_gesture_trainer)
        canvas_tb.addAction(train_tb_act)

        canvas_tb.addSeparator()

        # Background Template selector
        self._tpl_combo = QComboBox()
        self._tpl_combo.addItem("Nền: Trắng", "plain")
        self._tpl_combo.addItem("Nền: Ô li học sinh", "oli")
        self._tpl_combo.addItem("Nền: Dòng kẻ ngang", "lined")
        self._tpl_combo.addItem("Nền: Hệ trục Oxy", "oxy")
        self._tpl_combo.setToolTip("Thay đổi mẫu trang nền bảng viết kỹ thuật số")
        self._tpl_combo.currentIndexChanged.connect(self._on_template_bg_changed)
        canvas_tb.addWidget(self._tpl_combo)

    # ── Camera / Hand tracking ─────────────────────────────────────────────────

    def _connect_camera(self, camera_index: int) -> None:
        """Start the _QtCameraWorker on a background thread."""
        if self._cam_thread is not None:
            self._disconnect_camera()

        if camera_index < 0:
            logger.info("Camera index is negative (%d); camera thread not started.", camera_index)
            self._cam_widget.setText("📷 Camera offline")
            return

        worker = _QtCameraWorker(camera_index=camera_index)
        thread = QThread(self)
        worker.moveToThread(thread)

        # Frame display
        worker.frame_ready.connect(self._on_camera_frame, Qt.ConnectionType.QueuedConnection)
        # Gesture events
        worker.gesture_event.connect(self._on_gesture_event, Qt.ConnectionType.QueuedConnection)
        # Error handling
        worker.error_occurred.connect(
            lambda msg: self._status.showMessage(f"Camera: {msg}", 5000),
            Qt.ConnectionType.QueuedConnection,
        )

        thread.started.connect(worker.start_capture)
        thread.start()
        self._cam_thread = thread
        self._cam_worker = worker

    def _disconnect_camera(self) -> None:
        if self._cam_worker is not None:
            self._cam_worker.stop()
        if self._cam_thread is not None and self._cam_thread.isRunning():
            self._cam_thread.quit()
            if not self._cam_thread.wait(1000):
                self._cam_thread.terminate()
                self._cam_thread.wait(500)
        self._cam_worker = None
        self._cam_thread = None

    # ── Camera frame ──────────────────────────────────────────────────────────

    @Slot(np.ndarray)
    def _on_camera_frame(self, frame_bgr: np.ndarray) -> None:
        self._cam_widget.on_frame(frame_bgr)
        fps = self._telemetry.record_frame_tick()
        if fps > 0:
            self._fps_label.setText(f"FPS: {fps:.1f}")
            self._status_fps.setText(f"FPS: {fps:.1f}")

    # ── Gesture FSM events ────────────────────────────────────────────────────

    @Slot(GestureState, float, float)
    def _on_gesture_event(self, state: GestureState, cx: float, cy: float) -> None:
        """
        Handle gesture state changes from the camera worker.
        cx, cy are normalized [0,1] canvas coordinates of the index fingertip.
        """
        bx, by = self._scene_coords(cx, cy)

        mode_name = state.name
        self._gesture_hud.setText(f"Mode: {mode_name}")
        self._status_mode.setText(f"Mode: {mode_name}")

        # Handle explicit manual teaching tool overrides
        if self._teaching_mode == "LASER":
            if state in (GestureState.POINT, GestureState.DRAW):
                self._scene.move_laser(bx, by, visible=True)
                if self._current_stroke_id is not None:
                    self._finalize_stroke()
            return
        elif self._teaching_mode == "ERASER":
            self._scene.move_laser(bx, by, visible=False)
            if state == GestureState.DRAW:
                erased = self._scene.erase_strokes_at(bx, by, radius=32.0)
                if erased:
                    self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
            elif self._current_stroke_id is not None:
                self._finalize_stroke()
            return

        if state == GestureState.POINT:
            self._scene.move_laser(bx, by, visible=True)
            if self._current_stroke_id is not None:
                self._finalize_stroke()

        elif state == GestureState.DRAW:
            self._scene.move_laser(bx, by, visible=False)
            t_ms = time.perf_counter() * 1000.0
            if self._current_stroke_id is None:
                self._start_new_stroke(bx, by, t_ms)
            else:
                self._extend_stroke(bx, by, t_ms)

        elif state == GestureState.PAUSE:
            self._scene.move_laser(bx, by, visible=False)
            self._finalize_stroke()

        elif state == GestureState.SELECT:
            if self._select_start is None:
                self._select_start = QPointF(bx, by)
            else:
                rect = QRectF(self._select_start, QPointF(bx, by)).normalized()
                self._scene.show_selection_rect(rect)

        elif state == GestureState.SWIPE_LEFT:
            self._on_prev_page()

        elif state == GestureState.SWIPE_RIGHT:
            self._on_next_page()

        elif state == GestureState.CONFIRM:
            self._scene.hide_selection_rect()
            self._select_start = None
            self._trigger_ocr_on_selection()

    def _scene_coords(self, norm_x: float, norm_y: float) -> tuple[float, float]:
        """Convert normalized [0,1] gesture coords to board scene coords."""
        return norm_x * BOARD_WIDTH, norm_y * BOARD_HEIGHT

    # ── Stroke building ───────────────────────────────────────────────────────

    def _start_new_stroke(self, x: float, y: float, t_ms: float) -> None:
        self._current_stroke_id = str(uuid.uuid4())
        self._live_raw_points = [(x, y, t_ms)]
        self._filter = OneEuroFilter2D()
        fx, fy = self._filter.filter(x, y, t_ms)
        self._scene.begin_stroke_preview(fx, fy, self._ink_color)

    def _extend_stroke(self, x: float, y: float, t_ms: float) -> None:
        if self._current_stroke_id is None:
            return
        self._live_raw_points.append((x, y, t_ms))
        fx, fy = self._filter.filter(x, y, t_ms)
        self._scene.update_stroke_preview(fx, fy)

    def _finalize_stroke(self) -> None:
        if self._current_stroke_id is None or len(self._live_raw_points) < 2:
            self._scene.end_stroke_preview()
            self._current_stroke_id = None
            self._live_raw_points.clear()
            return

        raw_pts = np.array([(p[0], p[1]) for p in self._live_raw_points])
        t_ms    = [p[2] for p in self._live_raw_points]

        # Tier 2: RDP simplification
        simplified = ramer_douglas_peucker(raw_pts, epsilon=self._rdp_epsilon)

        # Compute velocities for variable width
        dt = np.diff(t_ms) if len(t_ms) > 1 else np.array([1.0])
        dt = np.where(dt < 1, 1, dt)
        d  = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
        vels = d / dt
        avg_vel = float(vels.mean()) if len(vels) > 0 else 1.0

        # Tier 2: Bézier fitting
        segments = fit_cubic_bezier_spline(
            simplified,
            velocities=vels if len(vels) == len(simplified) else None,
            width_min=self._ink_min_width,
            width_max=self._ink_max_width,
        )
        widths = [seg.width_start for seg in segments] if segments else [2.5]

        # Tier 3: Shape snapping
        snap: Optional[SnapResult] = None
        if self._snapping_on and len(simplified) >= 2:
            try:
                snap = self._snapper.snap(simplified)
                if snap.shape_type == ShapeType.FREEFORM:
                    snap = None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Snap failed: %s", exc)

        stroke = InkStroke(
            stroke_id=self._current_stroke_id,
            segments=segments,
            widths=widths,
            snap=snap,
            color=self._ink_color,
            timestamps_ms=t_ms,
        )
        self._scene.commit_stroke(stroke)
        self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
        self._current_stroke_id = None
        self._live_raw_points.clear()

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _trigger_ocr_on_selection(self) -> None:
        """Run OCR on strokes within the selected region."""
        sel_item = self._scene._select_rect_item
        if sel_item is None:
            return
        rect = sel_item.rect()
        # Convert InkStroke → Stroke (smart_ink_ocr format) for strokes in region
        ocr_strokes: list[Stroke] = []
        for ink in self._scene._strokes.values():
            for seg in ink.segments:
                if rect.contains(QPointF(seg.p0[0], seg.p0[1])):
                    stroke = Stroke(
                        points=[StrokePoint(seg.p0[0], seg.p0[1], 0)],
                        color_bgr=(0, 0, 0),
                        width=2.5,
                    )
                    ocr_strokes.append(stroke)
                    break

        if not ocr_strokes:
            return

        # Determine top-left of region for overlay placement
        bx = rect.x()
        by = rect.y()

        runnable = _OcrRunnable(self._recognizer, ocr_strokes, bx, by)
        runnable.signals.done.connect(self._on_ocr_done, Qt.ConnectionType.QueuedConnection)
        self._thread_pool.start(runnable)
        self._scene.hide_selection_rect()

    @Slot(object)
    def _on_ocr_done(self, result) -> None:
        x = getattr(result, "_board_x", 120.0)
        y = getattr(result, "_board_y", 120.0)

        # Open SmartInkReviewDialog so teacher can inspect, edit LaTeX, and confirm
        from src.ui.smart_ink_review_dialog import SmartInkReviewDialog  # noqa: PLC0415
        dlg = SmartInkReviewDialog(ocr_result=result, parent=self)

        def _on_confirmed(latex_or_text: str, replace_ink: bool):
            if replace_ink:
                self._scene.erase_strokes_at(x, y, radius=120.0)
                self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())

            if result.is_formula and result.render_image_rgba is not None:
                self._scene.add_latex_overlay(result.render_image_rgba, x, y)
            elif result.text or latex_or_text:
                self._scene.add_text_overlay(latex_or_text or result.text, x, y)
            self._status.showMessage("Đã chèn nội dung nhận dạng lên bảng.", 3000)

        dlg.result_confirmed.connect(_on_confirmed)
        dlg.exec()

    def _on_manual_ocr(self) -> None:
        sel_item = self._scene._select_rect_item
        if sel_item is None:
            if not self._scene._strokes:
                QMessageBox.information(
                    self, "Nhận dạng chữ",
                    "Chưa có nét vẽ nào trên bảng. Hãy vẽ chữ hoặc dùng cử chỉ SELECT để khoanh vùng."
                )
                return
            # Auto-select all strokes on current page
            all_pts = [seg.p0 for s in self._scene._strokes.values() for seg in s.segments]
            if all_pts:
                xs = [p[0] for p in all_pts]
                ys = [p[1] for p in all_pts]
                rect = QRectF(min(xs) - 15, min(ys) - 15, max(xs) - min(xs) + 30, max(ys) - min(ys) + 30)
                self._scene.show_selection_rect(rect)
        self._trigger_ocr_on_selection()

    def _on_manual_formula_ocr(self) -> None:
        sel_item = self._scene._select_rect_item
        if sel_item is None:
            if not self._scene._strokes:
                QMessageBox.information(
                    self, "Nhận dạng công thức",
                    "Chưa có nét vẽ nào trên bảng. Hãy vẽ công thức hoặc dùng cử chỉ SELECT để khoanh vùng."
                )
                return
            all_pts = [seg.p0 for s in self._scene._strokes.values() for seg in s.segments]
            if all_pts:
                xs = [p[0] for p in all_pts]
                ys = [p[1] for p in all_pts]
                rect = QRectF(min(xs) - 15, min(ys) - 15, max(xs) - min(xs) + 30, max(ys) - min(ys) + 30)
                self._scene.show_selection_rect(rect)

        rect = self._scene._select_rect_item.rect()
        ocr_strokes: list[Stroke] = []
        for ink in self._scene._strokes.values():
            for seg in ink.segments:
                if rect.contains(QPointF(seg.p0[0], seg.p0[1])):
                    stroke = Stroke(
                        points=[StrokePoint(seg.p0[0], seg.p0[1], 0)],
                        color_bgr=(0, 0, 0),
                    )
                    ocr_strokes.append(stroke)
                    break
        if not ocr_strokes:
            return
        runnable = _OcrRunnable(
            self._recognizer, ocr_strokes, rect.x(), rect.y(), force_formula=True
        )
        runnable.signals.done.connect(self._on_ocr_done, Qt.ConnectionType.QueuedConnection)
        self._thread_pool.start(runnable)
        self._scene.hide_selection_rect()

    # ── Document management ───────────────────────────────────────────────────

    def _stash_current_page_strokes(self) -> None:
        """Save current canvas strokes to the active page buffer."""
        self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())

    def _restore_current_page_strokes(self) -> None:
        """Clear canvas and restore strokes for the active page."""
        self._scene.clear_all_strokes()
        saved_strokes = self._page_strokes.get(self._current_page_index, [])
        for stroke in saved_strokes:
            self._scene.commit_stroke(stroke)

    def _load_strokes_from_records(self, records: list) -> None:
        """Reconstruct InkStroke objects from StrokeRecords and populate _page_strokes."""
        for rec in records:
            segments: list[BezierSegment] = []
            if getattr(rec, "bezier_segments", None):
                for seg_d in rec.bezier_segments:
                    try:
                        p0 = np.array(seg_d["p0"], dtype=np.float64)
                        p1 = np.array(seg_d["p1"], dtype=np.float64)
                        p2 = np.array(seg_d["p2"], dtype=np.float64)
                        p3 = np.array(seg_d["p3"], dtype=np.float64)
                        w_start = float(seg_d.get("width_start", rec.width))
                        w_end = float(seg_d.get("width_end", rec.width))
                        segments.append(BezierSegment(p0, p1, p2, p3, w_start, w_end))
                    except Exception as err:
                        logger.warning("Error parsing bezier segment: %s", err)
            elif getattr(rec, "points", None) and len(rec.points) >= 2:
                pts = np.array([[p.x, p.y] for p in rec.points], dtype=np.float64)
                w = float(rec.width) if rec.width else 2.5
                segments = fit_cubic_bezier_spline(pts, width_min=w, width_max=w)

            widths = [float(rec.width) if rec.width else 2.5] * max(len(segments), 1)
            color = QColor(rec.color_hex) if rec.color_hex else QColor("#202124")

            snap = None
            shape_type_str = getattr(rec, "shape_type", "FREEFORM")
            if shape_type_str and shape_type_str != "FREEFORM":
                try:
                    st = ShapeType[shape_type_str]
                    if getattr(rec, "points", None) and len(rec.points) >= 2:
                        pts = np.array([[p.x, p.y] for p in rec.points], dtype=np.float64)
                        snap = self._snapper.snap(pts)
                    else:
                        snap = SnapResult(shape_type=st)
                except Exception as err:
                    logger.debug("Could not restore snapped shape: %s", err)

            stroke = InkStroke(
                stroke_id=rec.stroke_id,
                segments=segments,
                widths=widths,
                snap=snap,
                color=color,
                timestamps_ms=getattr(rec, "timestamps_ms", []),
            )
            self._page_strokes.setdefault(rec.page_index, []).append(stroke)

    def _on_open_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở tài liệu",
            str(Path.home()),
            "Tài liệu (*.pdf *.pptx *.docx);;Tất cả tệp (*.*)"
        )
        if not path:
            return
        try:
            self._status.showMessage("Đang mở tài liệu…", 0)
            QApplication.processEvents()
            self._page_strokes.clear()
            self._scene.clear_all_strokes()
            self._pages = self._doc_manager.import_document(Path(path))
            self._current_page_index = 0
            self._show_current_page()
            self._status.showMessage(
                f"Đã mở: {Path(path).name}  ({len(self._pages)} trang)", 5000
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi mở tài liệu", str(exc))

    def _on_open_airteach(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở gói bài giảng .airteach",
            str(Path.home()),
            "AirTeach packages (*.airteach)"
        )
        if not path:
            return
        try:
            from src.services.document_manager import PageInfo  # noqa: PLC0415
            manifest, page_paths, strokes, recognized = self._doc_manager.open_lesson(Path(path))
            self._pages = [
                PageInfo(page_index=i, image_path=str(p), width_px=0, height_px=0, text="")
                for i, p in enumerate(page_paths)
            ]
            self._page_strokes.clear()
            self._scene.clear_all_strokes()
            self._load_strokes_from_records(strokes)
            self._current_page_index = 0
            self._show_current_page()
            self._status.showMessage(
                f"Đã mở {Path(path).name} — {len(strokes)} nét mực.", 4000
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi mở gói bài giảng", str(exc))

    def _on_save_airteach(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu bài giảng",
            str(Path.home() / "bai_giang.airteach"),
            "AirTeach packages (*.airteach)"
        )
        if not path:
            return
        try:
            from src.services.document_manager import (  # noqa: PLC0415
                AirTeachManifest, StrokeRecord, BezierPointRecord, RecognizedItem
            )
            from datetime import datetime, timezone  # noqa: PLC0415

            self._stash_current_page_strokes()

            manifest = AirTeachManifest(
                title="Bài giảng AirTeach",
                page_count=len(getattr(self, "_pages", [])),
                created_at=datetime.now(timezone.utc).isoformat(),
                modified_at=datetime.now(timezone.utc).isoformat(),
            )
            stroke_records: list[StrokeRecord] = []
            for p_idx, p_strokes in self._page_strokes.items():
                for ink in p_strokes:
                    pts = []
                    t_list = ink.timestamps_ms
                    for i, seg in enumerate(ink.segments):
                        t = t_list[i] if i < len(t_list) else 0.0
                        pts.append(BezierPointRecord(x=float(seg.p0[0]), y=float(seg.p0[1]), t_ms=t))
                    if ink.segments:
                        last_seg = ink.segments[-1]
                        last_t = t_list[-1] if t_list else 0.0
                        pts.append(BezierPointRecord(x=float(last_seg.p3[0]), y=float(last_seg.p3[1]), t_ms=last_t))
                    seg_dicts = [seg.to_dict() for seg in ink.segments]
                    shape_str = ink.snap.shape_type.name if ink.snap else "FREEFORM"
                    avg_w = float(np.mean(ink.widths)) if ink.widths else 2.5
                    stroke_records.append(StrokeRecord(
                        stroke_id=ink.stroke_id,
                        page_index=p_idx,
                        color_hex=ink.color.name(),
                        width=avg_w,
                        points=pts,
                        bezier_segments=seg_dicts,
                        timestamps_ms=ink.timestamps_ms,
                        shape_type=shape_str,
                    ))
            page_image_paths = [
                Path(p.image_path) for p in getattr(self, "_pages", [])
                if Path(p.image_path).exists()
            ]
            self._doc_manager.save_lesson(
                output_path=Path(path),
                manifest=manifest,
                page_image_paths=page_image_paths,
                strokes=stroke_records,
                recognized=[],
            )
            self._status.showMessage(f"Đã lưu: {path}", 4000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi lưu", str(exc))

    def _on_export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất PDF", str(Path.home() / "bai_giang_chunthich.pdf"),
            "PDF (*.pdf)"
        )
        if not path:
            return
        try:
            from src.services.document_manager import StrokeRecord, BezierPointRecord  # noqa: PLC0415
            page_image_paths = [
                Path(p.image_path) for p in getattr(self, "_pages", [])
                if Path(p.image_path).exists()
            ]
            if not page_image_paths:
                QMessageBox.warning(self, "Xuất PDF", "Chưa có tài liệu nào được mở.")
                return

            self._stash_current_page_strokes()

            stroke_records: list[StrokeRecord] = []
            for p_idx, p_strokes in self._page_strokes.items():
                for ink in p_strokes:
                    pts = [BezierPointRecord(x=float(seg.p0[0]), y=float(seg.p0[1]), t_ms=0.0)
                           for seg in ink.segments]
                    if ink.segments:
                        last_seg = ink.segments[-1]
                        pts.append(BezierPointRecord(x=float(last_seg.p3[0]), y=float(last_seg.p3[1]), t_ms=0.0))
                    avg_w = float(np.mean(ink.widths)) if ink.widths else 2.5
                    stroke_records.append(StrokeRecord(
                        stroke_id=ink.stroke_id,
                        page_index=p_idx,
                        color_hex=ink.color.name(),
                        width=avg_w,
                        points=pts,
                    ))
            self._doc_manager.export_annotated_pdf(
                page_image_paths=page_image_paths,
                strokes=stroke_records,
                output_pdf_path=Path(path),
            )
            self._status.showMessage(f"Đã xuất PDF: {path}", 4000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi xuất PDF", str(exc))

    def _show_current_page(self) -> None:
        pages = getattr(self, "_pages", [])
        if not pages:
            return
        idx = max(0, min(self._current_page_index, len(pages) - 1))
        self._current_page_index = idx
        page = pages[idx]
        import cv2  # noqa: PLC0415
        img = cv2.imread(page.image_path)
        if img is not None:
            self._scene.set_background_page(img)
        self._restore_current_page_strokes()
        self._board_view.fitInView(
            self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self._chat_panel.set_current_page_text(page.text)
        self._update_status_bar()

    # ── Page navigation ───────────────────────────────────────────────────────

    def _on_prev_page(self) -> None:
        if self._current_page_index > 0:
            self._stash_current_page_strokes()
            self._current_page_index -= 1
            self._show_current_page()

    def _on_next_page(self) -> None:
        pages = getattr(self, "_pages", [])
        if self._current_page_index < len(pages) - 1:
            self._stash_current_page_strokes()
            self._current_page_index += 1
            self._show_current_page()

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _on_undo(self) -> None:
        if self._scene._strokes:
            last_id = list(self._scene._strokes.keys())[-1]
            stroke = self._scene._strokes.get(last_id)
            if stroke is not None:
                self._redo_stack.append(stroke)
            self._scene.remove_stroke(last_id)
            self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())

    def _on_redo(self) -> None:
        if self._redo_stack:
            stroke = self._redo_stack.pop()
            self._scene.commit_stroke(stroke)
            self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())

    def _set_teaching_mode(self, mode: str) -> None:
        self._teaching_mode = mode
        logger.info("Teaching mode changed to %s", mode)
        self._status_mode.setText(f"Mode: {mode}")

    def _set_ink_color(self, hex_color: str) -> None:
        self._ink_color = QColor(hex_color)
        self._status.showMessage(f"Đã chọn màu mực: {hex_color}", 2000)

    def _on_width_changed(self, index: int) -> None:
        val = float(self._width_combo.currentData())
        self._ink_min_width = val
        self._ink_max_width = val * 1.5
        self._status.showMessage(f"Độ dày nét: {val:.1f} px", 2000)

    def _toggle_shape_snapping(self, enabled: bool) -> None:
        self._snapping_on = enabled
        status_txt = "BẬT" if enabled else "TẮT"
        self._status.showMessage(f"Nắn hình học tự động: {status_txt}", 2000)

    def _on_template_bg_changed(self, index: int) -> None:
        """Switch the board background template (Plain, Oli grid, Lined, Oxy)."""
        from src.ui.library_and_prep_widget import BackgroundTemplateGenerator  # noqa: PLC0415
        key = self._tpl_combo.currentData() if hasattr(self, "_tpl_combo") else "plain"
        if key == "plain":
            img = BackgroundTemplateGenerator.render_plain_white()
        elif key == "oli":
            img = BackgroundTemplateGenerator.render_oli_grid()
        elif key == "lined":
            img = BackgroundTemplateGenerator.render_lined_paper()
        elif key == "oxy":
            img = BackgroundTemplateGenerator.render_oxy_coordinates()
        else:
            img = BackgroundTemplateGenerator.render_plain_white()
        self._scene.set_background_page(img)
        self._status.showMessage(f"Đã chuyển nền bảng: {self._tpl_combo.currentText()}", 2500)

    def _on_open_gesture_trainer(self) -> None:
        from src.ui.gesture_trainer_dialog import GestureTrainerDialog  # noqa: PLC0415
        dlg = GestureTrainerDialog(parent=self)
        if self._cam_worker is not None:
            prev_cb = self._cam_worker._on_frame_cb
            def _trainer_frame_cb(frame, lm, st, cx, cy):
                prev_cb(frame, lm, st, cx, cy)
                dlg.update_frame(frame, lm, st, cx, cy)
            self._cam_worker._worker.on_frame = _trainer_frame_cb
        dlg.exec()
        if self._cam_worker is not None:
            self._cam_worker._worker.on_frame = self._cam_worker._on_frame_cb

    def _on_clear_all(self) -> None:
        reply = QMessageBox.question(
            self, "Xóa tất cả",
            "Bạn có chắc muốn xóa tất cả các nét mực không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._scene.clear_all_strokes()
            self._page_strokes[self._current_page_index] = []

    def _on_toggle_replay(self) -> None:
        """Toggle playback toolbar and load active strokes into the replayer."""
        is_vis = not self._replay_ctrl.isVisible()
        self._replay_ctrl.setVisible(is_vis)
        if is_vis:
            self._replay_ctrl.load_strokes(list(self._scene._strokes.values()))

    def _on_open_library(self) -> None:
        """Open the Teacher Library & Preparation modal."""
        from src.ui.library_and_prep_widget import LibraryAndPrepWidget  # noqa: PLC0415
        teacher_name = self._teacher["full_name"] if self._teacher else "Dạy thử / Demo"

        dlg = QDialog(self)
        dlg.setWindowTitle("Thư viện Bài giảng & Soạn bài")
        dlg.resize(920, 620)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(0, 0, 0, 0)

        lib_widget = LibraryAndPrepWidget(
            workspace_dir=self._workspace_dir,
            doc_manager=self._doc_manager,
            teacher_name=teacher_name,
            parent=dlg,
        )

        def _on_load(path: Path) -> None:
            self._load_airteach_package(path)
            dlg.accept()

        lib_widget.open_lesson_requested.connect(_on_load)
        dlg_layout.addWidget(lib_widget)
        dlg.exec()

    def _on_logout_requested(self) -> None:
        """Confirm and cleanly transition back to the LoginWindow."""
        reply = QMessageBox.question(
            self,
            "Đăng xuất Giáo viên",
            "Bạn có chắc chắn muốn đăng xuất tài khoản hiện tại và trở về màn hình đăng nhập không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._disconnect_camera()
            self.logout_requested.emit()
            self.close()

    def _load_airteach_package(self, pkg_path: Path) -> None:
        """Open and display an .airteach lesson package."""
        try:
            from src.services.document_manager import PageInfo  # noqa: PLC0415
            manifest, page_paths, strokes, recognized = self._doc_manager.open_lesson(pkg_path)
            self._pages = [
                PageInfo(page_index=i, image_path=str(p), width_px=0, height_px=0, text="")
                for i, p in enumerate(page_paths)
            ]
            self._current_page_index = 0
            self._scene.clear_all_strokes()
            self._show_current_page()
            self._status.showMessage(f"Đã mở: {pkg_path.name} ({len(page_paths)} trang)", 4000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi mở bài giảng", str(exc))

    # ── Settings ──────────────────────────────────────────────────────────────

    def _on_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.api_key_changed.connect(self._on_api_key_changed)
        dlg.camera_index_changed.connect(self._on_camera_index_changed)
        dlg.calibration_requested.connect(self._on_calibration_requested)
        if dlg.exec():
            self._ink_color       = QColor(dlg.ink_color)
            self._ink_min_width   = dlg.min_stroke_width
            self._ink_max_width   = dlg.max_stroke_width
            self._rdp_epsilon     = dlg.rdp_epsilon
            self._snapping_on     = dlg.shape_snapping_enabled

    @Slot(str)
    def _on_api_key_changed(self, new_key: str) -> None:
        self._assistant.update_api_key(new_key)
        self._chat_panel.update_online_status(self._assistant.is_online)

    @Slot(int)
    def _on_camera_index_changed(self, index: int) -> None:
        if index != self._cam_index:
            self._cam_index = index
            self._disconnect_camera()
            self._connect_camera(index)

    @Slot()
    def _on_calibration_requested(self) -> None:
        if self._cam_worker is not None:
            self._cam_worker.start_perspective_calibration()
        else:
            QMessageBox.warning(self, "Hiệu chỉnh", "Camera chưa kết nối.")

    def _on_download_models(self) -> None:
        from src.utils.model_downloader import ensure_all_models  # noqa: PLC0415
        try:
            models_dir = Path(__file__).resolve().parents[2] / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            self._status.showMessage("Đang tải mô hình AI…", 0)
            QApplication.processEvents()
            ensure_all_models(models_dir=models_dir)
            self._status.showMessage("Tải mô hình hoàn tất!", 4000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi tải mô hình", str(exc))

    # ── Canvas snapshot for Gemini ─────────────────────────────────────────────

    @Slot()
    def _on_snapshot_request(self) -> None:
        """Capture the selected region of the board and send to chat panel."""
        sel_item = self._scene._select_rect_item
        if sel_item is not None:
            rect = sel_item.rect()
        else:
            rect = self._scene.sceneRect()

        pixmap = self._board_view.grab(
            self._board_view.mapFromScene(rect).boundingRect()
        )
        buf = pixmap.toImage()
        import io  # noqa: PLC0415
        from PySide6.QtCore import QBuffer, QIODevice  # noqa: PLC0415
        qbuf = QBuffer()
        qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
        buf.save(qbuf, "PNG")
        png_bytes = bytes(qbuf.data())
        self._chat_panel.receive_canvas_snapshot(png_bytes)

    # ── Stroke committed callback ─────────────────────────────────────────────

    @Slot(str)
    def _on_stroke_committed(self, stroke_id: str) -> None:
        logger.debug("Stroke committed: %s", stroke_id)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_status_bar(self) -> None:
        pages = getattr(self, "_pages", [])
        total = len(pages)
        if total > 0:
            self._status_page.setText(
                f"Trang: {self._current_page_index + 1} / {total}"
            )
        else:
            self._status_page.setText("Trang: —")

    # ── Window events ─────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._disconnect_camera()
        self._telemetry.stop()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._board_view.fitInView(
            self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
