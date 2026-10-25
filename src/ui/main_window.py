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
import math
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
from collections import OrderedDict
from PySide6.QtGui import (
    QAction,
    QKeyEvent,
    QActionGroup,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsLineItem,
    QGraphicsTextItem,
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
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

def _get_icon(name: str) -> QIcon:
    path = _ASSETS_DIR / "icons" / f"{name}.svg"
    return QIcon(str(path)) if path.exists() else QIcon()

import cv2
from src.core.hand_gesture_engine import (
    DrawMode,
    GestureState,
    HandTrackingWorker,
    PerspectiveMapper,
    LM,
)
from src.core.multi_hand_coordinator import (
    HandLandmarkData,
    HandRole,
    MultiHandMode,
    AirClickType,
)
from src.ui.floating_action_menu import FloatingActionMenu, MenuLayoutMode
from src.core.adaptive_vision_pipeline import CameraDiagnosticsData
from src.services.credential_service import TeacherAISettings, TeacherCredentialService
from src.ui.ai_workbench import AIWorkbenchWidget
from src.ui.camera_diagnostics_dialog import CameraDiagnosticsDialog
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
from src.ui.stem_tools import (
    BiologyTools,
    ChemistryTools,
    InformaticsTools,
    MathTools,
    PhysicsTools,
    StemSubject,
    StemToolDialog,
    format_chemical_formula,
)
from src.utils.telemetry_logger import TelemetryLogger

logger = logging.getLogger(__name__)


# ── Hand Live Stroke Data Structure ──────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class HandLiveStroke:
    stroke_id: str
    hand_id: str
    filter_2d: OneEuroFilter2D
    raw_points: list[tuple[float, float, float]]
    color: QColor
    width: float


# ── Qt bridge for callback-based HandTrackingWorker ───────────────────────────

class _QtCameraWorker(QObject):
    """
    QObject wrapper around HandTrackingWorker.
    HandTrackingWorker uses plain Python callbacks; this class converts
    those callbacks into Qt Signals so UI slots can safely receive them.
    """

    frame_ready          = Signal(np.ndarray)          # BGR frame
    gesture_event        = Signal(GestureState, float, float)  # (state, cx_norm, cy_norm)
    multi_gesture_event  = Signal(list)                # list[HandLandmarkData]
    trainer_frame_ready  = Signal(np.ndarray, object, object, float, float) # (frame, lm, state, cx, cy)
    diagnostics_ready    = Signal(object)              # CameraDiagnosticsData
    error_occurred       = Signal(str)
    camera_status_changed = Signal(str, str)          # (status: str, message: str)

    def __init__(
        self,
        camera_index: int = 0,
        lighting_mode: str = "adaptive",
        mirror: bool = True,
        multi_hand_mode: str = "draw_and_tool",
    ) -> None:
        super().__init__()
        self._last_multi_hands_data: list[HandLandmarkData] = []
        self.is_trainer_active: bool = False
        self._worker = HandTrackingWorker(
            camera_index=camera_index,
            on_frame=self._on_frame_cb,
            on_error=self._on_error_cb,
            on_diagnostics=self._on_diagnostics_cb,
            on_multi_hands=self._on_multi_hands_cb,
            on_camera_status=self._on_camera_status_cb,
            lighting_mode=lighting_mode,
            mirror=mirror,
            multi_hand_mode=multi_hand_mode,
        )

    def _on_diagnostics_cb(self, diag) -> None:
        self.diagnostics_ready.emit(diag)

    def _on_multi_hands_cb(self, hands_data: list[HandLandmarkData]) -> None:
        self._last_multi_hands_data = list(hands_data)
        self.multi_gesture_event.emit(hands_data)

    def set_multi_hand_mode(self, mode: str) -> None:
        self._worker.set_multi_hand_mode(mode)

    def set_dominant_hand(self, dominant_hand: str) -> None:
        self._worker.set_dominant_hand(dominant_hand)

    def set_air_click_type(self, click_type: str) -> None:
        self._worker.coordinator.set_click_type(click_type)

    def set_lighting_mode(self, mode: str) -> None:
        self._worker.set_lighting_mode(mode)

    def set_draw_mode(self, mode) -> None:
        self._worker.set_draw_mode(mode)

    def set_pinch_threshold(self, threshold: float) -> None:
        self._worker.set_pinch_threshold(threshold)

    def set_mirror(self, mirror: bool) -> None:
        self._worker.set_mirror(mirror)

    def _on_frame_cb(self, frame_bgr, landmarks, state: GestureState, cx: float, cy: float) -> None:
        annotated_frame = frame_bgr.copy()
        h, w = annotated_frame.shape[:2]
        connections = [
            (LM.WRIST, LM.THUMB_CMC), (LM.THUMB_CMC, LM.THUMB_MCP), (LM.THUMB_MCP, LM.THUMB_IP), (LM.THUMB_IP, LM.THUMB_TIP),
            (LM.WRIST, LM.INDEX_MCP), (LM.INDEX_MCP, LM.INDEX_PIP), (LM.INDEX_PIP, LM.INDEX_DIP), (LM.INDEX_DIP, LM.INDEX_TIP),
            (LM.WRIST, LM.MIDDLE_MCP), (LM.MIDDLE_MCP, LM.MIDDLE_PIP), (LM.MIDDLE_PIP, LM.MIDDLE_DIP), (LM.MIDDLE_DIP, LM.MIDDLE_TIP),
            (LM.WRIST, LM.RING_MCP), (LM.RING_MCP, LM.RING_PIP), (LM.RING_PIP, LM.RING_DIP), (LM.RING_DIP, LM.RING_TIP),
            (LM.WRIST, LM.PINKY_MCP), (LM.PINKY_MCP, LM.PINKY_PIP), (LM.PINKY_PIP, LM.PINKY_DIP), (LM.PINKY_DIP, LM.PINKY_TIP),
        ]

        hands_to_draw = getattr(self, "_last_multi_hands_data", [])
        if hands_to_draw:
            badge_parts = []
            for h_idx, h_data in enumerate(hands_to_draw):
                lms = h_data.landmarks
                if not lms or len(lms) < 21:
                    continue
                is_primary = (h_data.role == HandRole.PRIMARY) or (h_idx == 0)
                bone_color = (0, 220, 255) if is_primary else (255, 120, 220)
                joint_color = (0, 140, 255) if is_primary else (200, 50, 180)

                for p1_idx, p2_idx in connections:
                    pt1 = (int(lms[p1_idx].x * w), int(lms[p1_idx].y * h))
                    pt2 = (int(lms[p2_idx].x * w), int(lms[p2_idx].y * h))
                    cv2.line(annotated_frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

                for lm_pt in lms:
                    px = int(lm_pt.x * w)
                    py = int(lm_pt.y * h)
                    cv2.circle(annotated_frame, (px, py), 3, joint_color, -1, cv2.LINE_AA)

                tip_x = int(lms[LM.INDEX_TIP].x * w)
                tip_y = int(lms[LM.INDEX_TIP].y * h)
                ring_color = (
                    (0, 0, 255) if h_data.state == GestureState.DRAW
                    else ((0, 255, 0) if h_data.state == GestureState.POINT
                    else ((0, 215, 255) if h_data.state in (GestureState.PINCH, GestureState.ERASE)
                    else (255, 200, 0)))
                )
                cv2.circle(annotated_frame, (tip_x, tip_y), 8, ring_color, 2, cv2.LINE_AA)
                cv2.circle(annotated_frame, (tip_x, tip_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

                hand_label = f"{h_data.track_id}:{h_data.handedness[:1]}[{h_data.state.name}]"
                badge_parts.append(hand_label)

            badge_text = " | ".join(badge_parts) if badge_parts else "Mode: NO HAND"
            font_scale = max(0.32, min(0.48, (w - 24) / max(1.0, len(badge_text) * 16.0)))
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            box_width = min(w - 8, 20 + tw)
            cv2.rectangle(annotated_frame, (8, 8), (box_width, max(32, 14 + th + 6)), (20, 20, 30), -1)
            cv2.putText(annotated_frame, badge_text, (14, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 200), 1, cv2.LINE_AA)
        elif landmarks:
            for p1_idx, p2_idx in connections:
                pt1 = (int(landmarks[p1_idx].x * w), int(landmarks[p1_idx].y * h))
                pt2 = (int(landmarks[p2_idx].x * w), int(landmarks[p2_idx].y * h))
                cv2.line(annotated_frame, pt1, pt2, (0, 220, 255), 2, cv2.LINE_AA)

            for lm_pt in landmarks:
                px = int(lm_pt.x * w)
                py = int(lm_pt.y * h)
                cv2.circle(annotated_frame, (px, py), 3, (0, 140, 255), -1, cv2.LINE_AA)

            tip_x = int(landmarks[LM.INDEX_TIP].x * w)
            tip_y = int(landmarks[LM.INDEX_TIP].y * h)
            ring_color = (0, 0, 255) if state == GestureState.DRAW else ((0, 255, 0) if state == GestureState.POINT else (255, 200, 0))
            cv2.circle(annotated_frame, (tip_x, tip_y), 8, ring_color, 2, cv2.LINE_AA)
            cv2.circle(annotated_frame, (tip_x, tip_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

            badge_text = f"Mode: {state.name}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(annotated_frame, (8, 8), (min(w - 8, 20 + tw), max(32, 14 + th + 6)), (20, 20, 30), -1)
            cv2.putText(annotated_frame, badge_text, (14, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.50, ring_color, 1, cv2.LINE_AA)

        self.frame_ready.emit(annotated_frame)
        self.gesture_event.emit(state, cx, cy)
        if getattr(self, 'is_trainer_active', False):
            self.trainer_frame_ready.emit(frame_bgr, landmarks, state, cx, cy)

    def _on_error_cb(self, msg: str) -> None:
        self.error_occurred.emit(msg)

    def _on_camera_status_cb(self, status: str, msg: str) -> None:
        self.camera_status_changed.emit(status, msg)

    def get_camera_status(self) -> dict:
        return self._worker.get_camera_status()

    def start_capture(self) -> None:
        """Called when QThread starts — runs the blocking camera loop."""
        self._worker.start()

    def stop(self) -> None:
        self._worker.stop()

    def start_perspective_calibration(self) -> None:
        """Reset perspective mapping to identity (full-frame → full-canvas)."""
        self._worker.reset_perspective()

    def calibrate_perspective(self, corners_normalized: np.ndarray) -> bool:
        """Apply new perspective calibration corners across tracking engines."""
        return self._worker.calibrate_perspective(corners_normalized)

    def reset_perspective(self) -> None:
        """Reset perspective mapping to identity across tracking engines."""
        self._worker.reset_perspective()


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
        self._page_stem_items: dict[int, list[QGraphicsItem]] = {}
        self._page_undo_stack: dict[int, list[tuple[str, Any]]] = {}
        self._page_redo_stack: dict[int, list[tuple[str, Any]]] = {}
        self._ink_color = QColor("#202124")
        self._ink_min_width = 1.5
        self._ink_max_width = 6.0
        self._rdp_epsilon   = 2.5
        self._snapping_on   = True
        self._select_start: Optional[QPointF] = None
        self._teaching_mode: str = "PEN"  # "PEN", "LASER", "ERASER", "SELECT"
        self._redo_stack: list[InkStroke] = []
        self._selected_items: list[QGraphicsItem] = []
        self._selected_strokes: list[InkStroke] = []
        self._is_dragging_selection: bool = False
        self._drag_last_pos: Optional[QPointF] = None
        self._drag_total_delta: QPointF = QPointF(0.0, 0.0)
        self._page_pixmap_cache: OrderedDict[int, QPixmap] = OrderedDict()
        self._MAX_PAGE_CACHE: int = 8

        # Multi-Hand & Air Interaction State
        self._multi_hand_mode = getattr(self._ai_settings, "multi_hand_mode", "draw_and_tool") if hasattr(self, "_ai_settings") else "draw_and_tool"
        self._dominant_hand = getattr(self._ai_settings, "dominant_hand", "Right") if hasattr(self, "_ai_settings") else "Right"
        self._air_click_type = getattr(self._ai_settings, "air_click_type", "pinch") if hasattr(self, "_ai_settings") else "pinch"
        self._eraser_radius = float(getattr(self._ai_settings, "eraser_radius", 32.0)) if hasattr(self, "_ai_settings") else 32.0
        self._menu_layout_mode = getattr(self._ai_settings, "menu_layout_mode", "radial_wheel") if hasattr(self, "_ai_settings") else "radial_wheel"
        self._secondary_ink_color = QColor("#ea4335")  # Secondary hand ink color for dual-draw
        self._hand_strokes: dict[str, HandLiveStroke] = {}
        self._multi_hand_active = False

        # Services (lazy-init)
        _work_root = Path(__file__).resolve().parents[2] / "data" / "work"
        _work_root.mkdir(parents=True, exist_ok=True)
        self._doc_manager  = DocumentManager(_work_root)
        if self._teacher and self._teacher.get("workspace_dir"):
            self._workspace_dir = Path(self._teacher["workspace_dir"])
        else:
            self._workspace_dir = Path(__file__).resolve().parents[2] / "data" / "workspaces" / "demo"
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        # Isolated Per-Teacher Credential & AI Settings
        self._cred_service = TeacherCredentialService()
        self._teacher_id = self._teacher.get("id", "demo") if self._teacher else "demo"
        self._ai_settings = self._cred_service.get_settings(self._teacher_id)
        self._draw_mode = self._ai_settings.draw_mode or "index_up"
        self._camera_lighting_mode = self._ai_settings.camera_lighting_mode or "adaptive"
        self._multi_hand_mode = getattr(self._ai_settings, "multi_hand_mode", "draw_and_tool")
        self._dominant_hand = getattr(self._ai_settings, "dominant_hand", "Right")
        self._air_click_type = getattr(self._ai_settings, "air_click_type", "pinch")
        self._eraser_radius = float(getattr(self._ai_settings, "eraser_radius", 32.0))
        self._menu_layout_mode = getattr(self._ai_settings, "menu_layout_mode", "radial_wheel")

        self._assistant = GeminiAssistant(
            teacher_id=self._teacher_id,
            credential_service=self._cred_service,
            model_name=self._ai_settings.model_name or "gemini-2.5-flash",
        )
        self._recognizer   = SmartInkRecognizer()
        _tel_path = Path(__file__).resolve().parents[2] / "data" / "telemetry.csv"
        _tel_path.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry    = TelemetryLogger(_tel_path)
        self._telemetry.start()
        self._thread_pool  = QThreadPool.globalInstance()

        self._diagnostics_dialog: Optional[CameraDiagnosticsDialog] = None
        self._latest_diagnostics: Optional[CameraDiagnosticsData] = None
        self._sidebar_expanded = True
        self._is_desktop_overlay = False
        self._desktop_overlay_bar = None
        self._saved_geometry = None
        self._gesture_guide_overlay = None

        # Page cache (list[PageInfo])
        self._pages: list = []
        self._page_sticky_notes: dict[int, list] = {}

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

        # ── Far Left: Collapsible Workstation Navigation Sidebar ────────────
        self._sidebar = self._build_navigation_sidebar()
        root_h.addWidget(self._sidebar)

        # ── Left: board + camera stack ─────────────────────────────────────
        left_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Board view container
        board_container = QWidget()
        board_layout = QVBoxLayout(board_container)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(0)

        self._scene = LessonBoard(self)
        self._scene.stroke_committed.connect(self._on_stroke_committed)
        self._scene.board_theme_changed.connect(self._on_board_theme_changed)

        # Floating Action Menu (Z=50)
        menu_mode = (
            MenuLayoutMode.FLOATING_PANEL
            if "panel" in self._menu_layout_mode.lower()
            else MenuLayoutMode.RADIAL_WHEEL
        )
        self._floating_menu = FloatingActionMenu(
            canvas_width=BOARD_WIDTH,
            canvas_height=BOARD_HEIGHT,
            layout_mode=menu_mode,
        )
        self._scene.addItem(self._floating_menu)
        self._floating_menu.action_triggered.connect(self._on_menu_action_triggered)
        self._floating_menu.menu_closed.connect(self._on_menu_closed)

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

        # Interactive Canvas Zoom & Pan Controller
        from src.ui.canvas_zoom_controller import CanvasZoomController
        self._zoom_ctrl = CanvasZoomController(self._board_view, self._scene, parent=self)
        self._zoom_ctrl.zoom_changed.connect(self._on_zoom_changed)

        # Bimanual STEM Object Manipulator
        from src.ui.bimanual_stem_manipulator import BimanualSTEMManipulator
        self._bimanual_manipulator = BimanualSTEMManipulator(parent=self)

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

        # ── Right: AI Assistant & AI Workbench Dock ────────────────────────
        self._chat_panel = ChatPanel(self._assistant, self)
        self._chat_panel.request_canvas_snapshot.connect(self._on_snapshot_request)
        self._ai_workbench = AIWorkbenchWidget(assistant=self._assistant, parent=self)
        self._ai_workbench.insert_text_to_board.connect(self._on_insert_ai_text_to_board)

        self._ai_dock_tabs = QTabWidget()
        self._ai_dock_tabs.addTab(self._chat_panel, "💬 Trò chuyện AI")
        self._ai_dock_tabs.addTab(self._ai_workbench, "✨ AI Workbench")

        ai_dock = QDockWidget("Trợ lý & Soạn thảo AI", self)
        ai_dock.setWidget(self._ai_dock_tabs)
        ai_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, ai_dock)

        # ── Left: Slide Thumbnail Drawer Dock ──────────────────────────────
        from src.ui.slide_thumbnail_drawer import SlideThumbnailDrawer
        self._slide_drawer = SlideThumbnailDrawer(self)
        self._slide_drawer.page_selected.connect(self._go_to_page)
        self._slide_drawer.add_blank_page_requested.connect(self._on_add_blank_page)
        self._slide_drawer.delete_page_requested.connect(self._on_delete_page)
        self._slide_drawer.duplicate_page_requested.connect(self._on_duplicate_page)

        self._slide_dock = QDockWidget("📑 Slides Bài Giảng", self)
        self._slide_dock.setWidget(self._slide_drawer)
        self._slide_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._slide_dock)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_mode  = QLabel("Mode: IDLE")
        self._status_page  = QLabel("Trang: —")
        self._status_fps   = QLabel("FPS: —")
        for lbl in (self._status_mode, self._status_page, self._status_fps):
            lbl.setStyleSheet("padding: 0 8px;")
            self._status.addPermanentWidget(lbl)

    # ── Collapsible Navigation Sidebar ─────────────────────────────────────────

    def _build_navigation_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(180)
        sidebar.setStyleSheet("""
            QWidget {
                background: #1e1e2e;
                color: #cdd6f4;
            }
            QPushButton {
                background: transparent;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                text-align: left;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #313244;
                color: #89b4fa;
            }
            QPushButton:pressed {
                background: #45475a;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(6, 10, 6, 10)
        layout.setSpacing(4)

        # Header with expand/collapse toggle
        header_h = QHBoxLayout()
        self._btn_sidebar_toggle = QPushButton("☰")
        self._btn_sidebar_toggle.setFixedSize(32, 32)
        self._btn_sidebar_toggle.setStyleSheet(
            "font-size: 14px; text-align: center; font-weight: bold; background: #313244;"
        )
        self._btn_sidebar_toggle.setToolTip("Thu gọn / Mở rộng thanh điều hướng")
        self._btn_sidebar_toggle.clicked.connect(self._toggle_sidebar_expanded)
        header_h.addWidget(self._btn_sidebar_toggle)

        self._sidebar_title = QLabel(" AirTeach AI")
        self._sidebar_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._sidebar_title.setStyleSheet("color: #89b4fa;")
        header_h.addWidget(self._sidebar_title)
        header_h.addStretch(1)
        layout.addLayout(header_h)

        # Navigation Action Buttons
        self._sidebar_buttons: list[QPushButton] = []

        nav_items = [
            ("🎨 Bảng tương tác", self._on_sidebar_canvas, "Màn hình bảng viết và giảng dạy chính"),
            ("📂 Kho bài giảng", self._on_open_library, "Mở kho lưu trữ bài giảng cá nhân"),
            ("📑 Trình chiếu", self._on_toggle_presentation, "Chế độ trình chiếu toàn màn hình"),
            ("✨ AI Workbench", self._on_open_ai_workbench, "Trạm soạn thảo giáo án và câu hỏi AI"),
            ("🔬 Chẩn đoán Camera", self._on_open_camera_diagnostics, "Theo dõi chất lượng webcam và FPS thực tế"),
            ("🎯 Luyện cử chỉ", self._on_open_gesture_trainer, "Hiệu chuẩn và luyện tập 6 cử chỉ tay"),
            ("⚙️ Cài đặt AI & Máy", self._on_settings, "Cấu hình API Key, Camera và Bút vẽ"),
        ]

        for text, slot, tip in nav_items:
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            layout.addWidget(btn)
            self._sidebar_buttons.append(btn)

        layout.addStretch(1)

        # Logout at bottom
        self._btn_logout = QPushButton("🚪 Đăng xuất")
        self._btn_logout.setStyleSheet("color: #f38ba8; text-align: left;")
        self._btn_logout.setToolTip("Đăng xuất tài khoản giáo viên")
        self._btn_logout.clicked.connect(self._on_logout_requested)
        layout.addWidget(self._btn_logout)
        self._sidebar_buttons.append(self._btn_logout)

        return sidebar

    def _toggle_sidebar_expanded(self) -> None:
        self._sidebar_expanded = not self._sidebar_expanded
        if self._sidebar_expanded:
            self._sidebar.setFixedWidth(180)
            self._sidebar_title.setVisible(True)
            labels = [
                "🎨 Bảng tương tác",
                "📂 Kho bài giảng",
                "📑 Trình chiếu",
                "✨ AI Workbench",
                "🔬 Chẩn đoán Camera",
                "🎯 Luyện cử chỉ",
                "⚙️ Cài đặt AI & Máy",
                "🚪 Đăng xuất",
            ]
            for btn, label in zip(self._sidebar_buttons, labels):
                btn.setText(label)
                btn.setStyleSheet("text-align: left; padding: 8px 12px;")
        else:
            self._sidebar.setFixedWidth(52)
            self._sidebar_title.setVisible(False)
            icons = ["🎨", "📂", "📑", "✨", "🔬", "🎯", "⚙️", "🚪"]
            for btn, icon in zip(self._sidebar_buttons, icons):
                btn.setText(icon)
                btn.setStyleSheet("text-align: center; padding: 8px 0;")

    def _on_sidebar_canvas(self) -> None:
        if self._board_view and self._scene:
            self._board_view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._status.showMessage("Chế độ: Bảng dạy học tương tác", 3000)

    def _on_open_ai_workbench(self) -> None:
        if hasattr(self, "_ai_dock_tabs"):
            self._ai_dock_tabs.setCurrentIndex(1)
        self._status.showMessage("Mở Trạm soạn thảo & Trợ giảng AI", 3000)

    def _on_open_camera_diagnostics(self) -> None:
        if self._diagnostics_dialog is None:
            self._diagnostics_dialog = CameraDiagnosticsDialog(self)
            self._diagnostics_dialog.lighting_mode_changed.connect(self._on_lighting_mode_changed)
            self._diagnostics_dialog.calibration_requested.connect(self._on_open_gesture_trainer)
        if self._latest_diagnostics is not None:
            self._diagnostics_dialog.update_diagnostics(self._latest_diagnostics)
        self._diagnostics_dialog.show()
        self._diagnostics_dialog.raise_()
        self._diagnostics_dialog.activateWindow()

    def _on_lighting_mode_changed(self, mode: str) -> None:
        self._camera_lighting_mode = mode
        if self._cam_worker:
            self._cam_worker.set_lighting_mode(mode)
        if hasattr(self, "_ai_settings"):
            self._ai_settings.camera_lighting_mode = mode
            self._cred_service.save_settings(self._ai_settings)
        self._status.showMessage(f"Chế độ tiền xử lý camera: {mode}", 3000)

    def _on_toggle_presentation(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self._status.showMessage("Thoát chế độ trình chiếu toàn màn hình", 2000)
        else:
            self.showFullScreen()
            self._status.showMessage("Chế độ Trình chiếu toàn màn hình (Nhấn F11 hoặc Esc để thoát)", 4000)

    def _on_insert_ai_text_to_board(self, text: str) -> None:
        """Insert AI generated text cleanly onto current canvas page."""
        cx = BOARD_WIDTH / 2.0 - 250.0
        cy = BOARD_HEIGHT / 2.0 - 150.0
        self._scene.add_text_overlay(text[:1200], cx, cy, font_size=16)
        self._update_status_bar()

    def _toggle_draw_mode(self) -> None:
        if self._draw_mode == "precision_pinch":
            self._draw_mode = "index_up"
        else:
            self._draw_mode = "precision_pinch"
        self._apply_draw_mode()

    def _apply_draw_mode(self) -> None:
        mode_enum = DrawMode.PRECISION_PINCH if self._draw_mode == "precision_pinch" else DrawMode.INDEX_UP
        if self._cam_worker:
            self._cam_worker.set_draw_mode(mode_enum)
        if hasattr(self, "_ai_settings"):
            self._ai_settings.draw_mode = self._draw_mode
            self._cred_service.save_settings(self._ai_settings)
        if hasattr(self, "_draw_mode_btn"):
            if self._draw_mode == "precision_pinch":
                self._draw_mode_btn.setText("👌 Véo ngón (Pinch)")
                self._draw_mode_btn.setToolTip("Chế độ vẽ: Véo ngón trỏ & cái để vẽ. Nhấn để chuyển sang Chỉ tay.")
            else:
                self._draw_mode_btn.setText("☝️ Chỉ tay (Air Pen)")
                self._draw_mode_btn.setToolTip("Chế độ vẽ: Chỉ ngón trỏ để vẽ tự nhiên. Nhấn để chuyển sang Véo ngón.")
        msg = "Chế độ vẽ: Véo ngón tay (Pinch-to-draw)" if self._draw_mode == "precision_pinch" else "Chế độ vẽ: Chỉ tay tự nhiên (Air Pen)"
        self._status.showMessage(msg, 3000)

    @Slot(object)
    def _on_camera_diagnostics(self, diag: CameraDiagnosticsData) -> None:
        self._latest_diagnostics = diag
        if self._diagnostics_dialog is not None and self._diagnostics_dialog.isVisible():
            self._diagnostics_dialog.update_diagnostics(diag)
        if hasattr(self, "_cam_diag_btn"):
            fps = diag.fps
            cond = diag.lighting_condition.replace("_", " ").title()
            self._cam_diag_btn.setText(f"📹 {fps:.0f} FPS • {cond}")

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

        export_img_act = QAction("🖼️ Xuất ảnh bảng vẽ (PNG/JPG)…", self)
        export_img_act.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_img_act.triggered.connect(self._on_export_image)
        file_menu.addAction(export_img_act)

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
        view_menu = mb.addMenu("Hiển thị")
        slide_dock_act = QAction("📑 Danh sách Slide bài giảng (Slide Drawer)", self)
        slide_dock_act.setShortcut(QKeySequence("F2"))
        slide_dock_act.triggered.connect(self._on_toggle_slide_dock)
        view_menu.addAction(slide_dock_act)

        sticky_act = QAction("📝 Thêm Ghi chú dán (Sticky Note)…", self)
        sticky_act.setShortcut(QKeySequence("Ctrl+N"))
        sticky_act.triggered.connect(self._on_add_sticky_note)
        view_menu.addAction(sticky_act)

        board_theme_act = QAction("🎨 Chế độ bảng (Trắng / Phấn xanh / Ô ly)", self)
        board_theme_act.setShortcut(QKeySequence("Ctrl+B"))
        board_theme_act.triggered.connect(self._on_toggle_board_theme)
        view_menu.addAction(board_theme_act)

        grid_act = QAction(_get_icon("grid"), "Lưới tọa độ Oxy / Lưới ô vuông", self)
        grid_act.setShortcut(QKeySequence("Ctrl+G"))
        grid_act.triggered.connect(self._on_toggle_grid)
        view_menu.addAction(grid_act)

        ruler_act = QAction("📐 Thước kẻ centimet (Virtual Ruler)", self)
        ruler_act.setShortcut(QKeySequence("Ctrl+R"))
        ruler_act.triggered.connect(self._on_toggle_ruler)
        view_menu.addAction(ruler_act)

        protractor_act = QAction("🧭 Thước đo độ (Virtual Protractor)", self)
        protractor_act.setShortcut(QKeySequence("Ctrl+Shift+R"))
        protractor_act.triggered.connect(self._on_toggle_protractor)
        view_menu.addAction(protractor_act)

        compass_act = QAction("📐 Compa vẽ đường tròn (Virtual Compass)", self)
        compass_act.setShortcut(QKeySequence("Ctrl+Shift+C"))
        compass_act.triggered.connect(self._on_toggle_compass)
        view_menu.addAction(compass_act)

        spotlight_act = QAction("🔦 Đèn rọi Spotlight (Presentation Spotlight)", self)
        spotlight_act.setShortcut(QKeySequence("S"))
        spotlight_act.triggered.connect(self._on_toggle_spotlight)
        view_menu.addAction(spotlight_act)

        loupe_act = QAction("🔍 Kính lúp phóng đại (Virtual Loupe)", self)
        loupe_act.setShortcut(QKeySequence("M"))
        loupe_act.triggered.connect(self._on_toggle_loupe)
        view_menu.addAction(loupe_act)

        timer_act = QAction("⏱️ Đồng hồ đếm giờ bài giảng (Presentation Timer)", self)
        timer_act.setShortcut(QKeySequence("T"))
        timer_act.triggered.connect(self._on_toggle_timer)
        view_menu.addAction(timer_act)

        metronome_act = QAction("🎵 Máy đếm nhịp & Tiết tấu (Classroom Metronome)", self)
        metronome_act.setShortcut(QKeySequence("Ctrl+M"))
        metronome_act.triggered.connect(self._on_toggle_metronome)
        view_menu.addAction(metronome_act)

        view_menu.addSeparator()
        zoom_in_act = QAction("🔍 Phóng to bảng vẽ (Zoom In)", self)
        zoom_in_act.setShortcut(QKeySequence("Ctrl++"))
        zoom_in_act.triggered.connect(self._on_zoom_in)
        view_menu.addAction(zoom_in_act)

        zoom_out_act = QAction("🔍 Thu nhỏ bảng vẽ (Zoom Out)", self)
        zoom_out_act.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_act.triggered.connect(self._on_zoom_out)
        view_menu.addAction(zoom_out_act)

        zoom_reset_act = QAction("🎯 Vừa khung hình 100% (Reset Zoom)", self)
        zoom_reset_act.setShortcut(QKeySequence("Ctrl+0"))
        zoom_reset_act.triggered.connect(self._on_zoom_reset)
        view_menu.addAction(zoom_reset_act)

        pan_act = QAction("🖐️ Chế độ cuộn/kéo bảng (Pan Mode)", self)
        pan_act.setShortcut(QKeySequence("H"))
        pan_act.triggered.connect(self._on_toggle_pan)
        view_menu.addAction(pan_act)

        minimap_act = QAction("🗺️ Bản đồ thu nhỏ toàn cảnh (Canvas Mini-Map)", self)
        minimap_act.setShortcut(QKeySequence("Ctrl+Shift+V"))
        minimap_act.triggered.connect(self._on_toggle_minimap)
        view_menu.addAction(minimap_act)

        coord_act = QAction("📍 Tọa độ Oxy thời gian thực (Coordinate HUD)", self)
        coord_act.setShortcut(QKeySequence("Ctrl+Shift+X"))
        coord_act.triggered.connect(self._on_toggle_coordinate_hud)
        view_menu.addAction(coord_act)

        snap_coord_act = QAction("🧲 Bắt dính lưới tọa độ Oxy (Snap to Grid)", self)
        snap_coord_act.setShortcut(QKeySequence("Ctrl+Shift+G"))
        snap_coord_act.triggered.connect(self._on_toggle_coordinate_snap)
        view_menu.addAction(snap_coord_act)

        angle_act = QAction("📐 Thước đo góc động (Interactive Angle Measure)", self)
        angle_act.setShortcut(QKeySequence("Ctrl+Shift+A"))
        angle_act.triggered.connect(self._on_toggle_angle_tool)
        view_menu.addAction(angle_act)

        triangle_act = QAction("🔺 Tam giác hình học động (Dynamic Triangle Solver)", self)
        triangle_act.setShortcut(QKeySequence("Ctrl+Shift+T"))
        triangle_act.triggered.connect(self._on_toggle_triangle_tool)
        view_menu.addAction(triangle_act)

        func_act = QAction("📈 Đồ thị hàm số & Tiếp tuyến động (Function Plotter)", self)
        func_act.setShortcut(QKeySequence("Ctrl+Shift+F"))
        func_act.triggered.connect(self._on_toggle_function_plotter)
        view_menu.addAction(func_act)

        incline_act = QAction("⚡ Hệ lực mặt phẳng nghiêng (Inclined Plane Inspector)", self)
        incline_act.setShortcut(QKeySequence("Ctrl+Shift+I"))
        incline_act.triggered.connect(self._on_toggle_inclined_plane)
        view_menu.addAction(incline_act)

        optics_act = QAction("🌈 Khúc xạ & Phản xạ toàn phần (Snell's Law Optics)", self)
        optics_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        optics_act.triggered.connect(self._on_toggle_optics_refraction)
        view_menu.addAction(optics_act)

        transformer_act = QAction("🤖 Mô phỏng Transformer Attention & Multi-Head (Vaswani et al.)", self)
        transformer_act.setShortcut(QKeySequence("Ctrl+Shift+A"))
        transformer_act.triggered.connect(self._on_toggle_transformer_attention)
        view_menu.addAction(transformer_act)

        relativity_act = QAction("🌌 Thuyết Tương đối Hẹp & Giản đồ Minkowski (Einstein 1905)", self)
        relativity_act.setShortcut(QKeySequence("Ctrl+Shift+R"))
        relativity_act.triggered.connect(self._on_toggle_special_relativity)
        view_menu.addAction(relativity_act)

        quantum_act = QAction("⚛️ Cơ học Lượng tử: Giếng thế & Xuyên hầm (Quantum Tunneling)", self)
        quantum_act.setShortcut(QKeySequence("Ctrl+Shift+Q"))
        quantum_act.triggered.connect(self._on_toggle_quantum_tunneling)
        view_menu.addAction(quantum_act)

        coriolis_act = QAction("🌪️ Vật lý Khí quyển: Hiệu ứng Coriolis & Gió địa chuyển", self)
        coriolis_act.setShortcut(QKeySequence("Ctrl+Shift+C"))
        coriolis_act.triggered.connect(self._on_toggle_coriolis_earth)
        view_menu.addAction(coriolis_act)

        acoustic_act = QAction("🔊 Vật lý Âm học: Sóng âm Doppler & Sonar Beamforming", self)
        acoustic_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        acoustic_act.triggered.connect(self._on_toggle_acoustic_sonar)
        view_menu.addAction(acoustic_act)

        cloud_chamber_act = QAction("☢️ Vật lý Hạt nhân: Buồng sương Wilson & Vệt bức xạ", self)
        cloud_chamber_act.setShortcut(QKeySequence("Ctrl+Shift+N"))
        cloud_chamber_act.triggered.connect(self._on_toggle_cloud_chamber)
        view_menu.addAction(cloud_chamber_act)

        orbital_act = QAction("🚀 Cơ học Quỹ đạo: Chuyển tiếp Hohmann & Gravity Slingshot", self)
        orbital_act.setShortcut(QKeySequence("Ctrl+Shift+L"))
        orbital_act.triggered.connect(self._on_toggle_orbital_mechanics)
        view_menu.addAction(orbital_act)

        heat_transfer_act = QAction("🔥 Nhiệt động lực học: Truyền nhiệt 2D Fourier & Đối lưu Newton", self)
        heat_transfer_act.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        heat_transfer_act.triggered.connect(self._on_toggle_heat_transfer)
        view_menu.addAction(heat_transfer_act)

        quantum_diffraction_act = QAction("⚛️ Cơ học Lượng tử: Giao thoa Khe kép & Sóng de Broglie", self)
        quantum_diffraction_act.setShortcut(QKeySequence("Ctrl+Shift+1"))
        quantum_diffraction_act.triggered.connect(self._on_toggle_quantum_diffraction)
        view_menu.addAction(quantum_diffraction_act)

        laval_nozzle_act = QAction("🚀 Khí động học: Ống phun de Laval & Kim cương Mach", self)
        laval_nozzle_act.setShortcut(QKeySequence("Ctrl+Shift+2"))
        laval_nozzle_act.triggered.connect(self._on_toggle_laval_nozzle)
        view_menu.addAction(laval_nozzle_act)

        ramachandran_plot_act = QAction("🧬 Hóa sinh cấu trúc: Giản đồ Ramachandran & Gấp cuộn Protein", self)
        ramachandran_plot_act.setShortcut(QKeySequence("Ctrl+Shift+3"))
        ramachandran_plot_act.triggered.connect(self._on_toggle_ramachandran_plot)
        view_menu.addAction(ramachandran_plot_act)

        double_pendulum_act = QAction("🌀 Động lực học phi tuyến & Hỗn loạn: Con lắc kép (Chaos Theory)", self)
        double_pendulum_act.setShortcut(QKeySequence("Ctrl+Shift+4"))
        double_pendulum_act.triggered.connect(self._on_toggle_double_pendulum)
        view_menu.addAction(double_pendulum_act)

        superconductivity_act = QAction("⚡ Vật lý chất rắn & Siêu dẫn: Hiệu ứng Meissner & Bay lượng tử (Superconductivity)", self)
        superconductivity_act.setShortcut(QKeySequence("Ctrl+Shift+5"))
        superconductivity_act.triggered.connect(self._on_toggle_superconductivity)
        view_menu.addAction(superconductivity_act)

        wave_act = QAction("🌊 Sóng cơ & Sóng dừng (Standing Wave Simulator)", self)
        wave_act.setShortcut(QKeySequence("Ctrl+Shift+W"))
        wave_act.triggered.connect(self._on_toggle_standing_wave)
        view_menu.addAction(wave_act)

        vsepr_act = QAction("⚛️ Hình học phân tử 3D VSEPR (VSEPR Molecular Geometry)", self)
        vsepr_act.setShortcut(QKeySequence("Ctrl+Shift+Y"))
        vsepr_act.triggered.connect(self._on_toggle_vsepr_geometry)
        view_menu.addAction(vsepr_act)

        dna_act = QAction("🧬 Tự nhân đôi DNA (DNA Replication & Base Pairing)", self)
        dna_act.setShortcut(QKeySequence("Ctrl+Shift+D"))
        dna_act.triggered.connect(self._on_toggle_dna_replication)
        view_menu.addAction(dna_act)

        kepler_act = QAction("🪐 Quỹ đạo hành tinh & Định luật Kepler (Kepler Planetary Orbits)", self)
        kepler_act.setShortcut(QKeySequence("Ctrl+Shift+K"))
        kepler_act.triggered.connect(self._on_toggle_kepler_orbit)
        view_menu.addAction(kepler_act)

        graph_act = QAction("🕸️ Lý thuyết đồ thị & Dijkstra / A* (Graph Theory & Pathfinding)", self)
        graph_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
        graph_act.triggered.connect(self._on_toggle_graph_pathfinding)
        view_menu.addAction(graph_act)

        electric_field_act = QAction("⚡ Điện trường & Định luật Coulomb (Coulomb Electric Field)", self)
        electric_field_act.setShortcut(QKeySequence("Ctrl+Shift+E"))
        electric_field_act.triggered.connect(self._on_toggle_electric_field)
        view_menu.addAction(electric_field_act)

        lorentz_act = QAction("🧲 Lực Lorentz & Từ trường (Lorentz Magnetic Deflection)", self)
        lorentz_act.setShortcut(QKeySequence("Ctrl+Shift+B"))
        lorentz_act.triggered.connect(self._on_toggle_lorentz_force)
        view_menu.addAction(lorentz_act)

        trophic_act = QAction("🌿 Lưới thức ăn sinh thái (Trophic Web Lotka-Volterra)", self)
        trophic_act.setShortcut(QKeySequence("Ctrl+Shift+U"))
        trophic_act.triggered.connect(self._on_toggle_trophic_web)
        view_menu.addAction(trophic_act)

        gas_laws_act = QAction("🎈 Định luật khí lý tưởng & Động học phân tử (Ideal Gas Laws PV=nRT)", self)
        gas_laws_act.setShortcut(QKeySequence("Ctrl+Shift+J"))
        gas_laws_act.triggered.connect(self._on_toggle_gas_laws)
        view_menu.addAction(gas_laws_act)

        titration_act = QAction("🧪 Chuẩn độ Axit - Bazơ & Đường cong pH (Acid-Base Titration & Buffer pH)", self)
        titration_act.setShortcut(QKeySequence("Ctrl+Shift+Q"))
        titration_act.triggered.connect(self._on_toggle_titration)
        view_menu.addAction(titration_act)

        oscillator_act = QAction("🧲 Dao động điều hòa & Cộng hưởng (Harmonic Oscillator & Resonance)", self)
        oscillator_act.setShortcut(QKeySequence("Ctrl+Shift+H"))
        oscillator_act.triggered.connect(self._on_toggle_harmonic_oscillator)
        view_menu.addAction(oscillator_act)

        decay_act = QAction("☢️ Định luật phóng xạ & Chuỗi phân rã (Radioactive Decay Law)", self)
        decay_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        decay_act.triggered.connect(self._on_toggle_radioactive_decay)
        view_menu.addAction(decay_act)

        kinetics_act = QAction("🧪 Tốc độ phản ứng & Cân bằng hóa học (Chemical Kinetics & Equilibrium)", self)
        kinetics_act.setShortcut(QKeySequence("Ctrl+Alt+K"))
        kinetics_act.triggered.connect(self._on_toggle_chemical_kinetics)
        view_menu.addAction(kinetics_act)

        crypto_act = QAction("🔐 Mật mã học RSA & Số học đồng dư (RSA Cryptography & Modular Arithmetic)", self)
        crypto_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        crypto_act.triggered.connect(self._on_toggle_cryptography)
        view_menu.addAction(crypto_act)

        orbits_act = QAction("🌌 Quỹ đạo vạn vật hấp dẫn & Điểm Lagrange (Gravitational Orbits & Lagrange Points)", self)
        orbits_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        orbits_act.triggered.connect(self._on_toggle_gravitational_orbits)
        view_menu.addAction(orbits_act)

        relativity_act = QAction("⚡ Không-thời gian Minkowski & Biến đổi Lorentz (Minkowski Spacetime & Lorentz Boost)", self)
        relativity_act.setShortcut(QKeySequence("Ctrl+Alt+M"))
        relativity_act.triggered.connect(self._on_toggle_spacetime_relativity)
        view_menu.addAction(relativity_act)

        young_act = QAction("🌈 Giao thoa sóng ánh sáng & Nhiễu xạ (Young's Interference & Grating)", self)
        young_act.setShortcut(QKeySequence("Ctrl+Alt+Y"))
        young_act.triggered.connect(self._on_toggle_young_interference)
        view_menu.addAction(young_act)

        rlc_act = QAction("⚡ Mạch điện xoay chiều RLC & Giản đồ Fresnel (AC Series RLC Circuit & Phasor)", self)
        rlc_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        rlc_act.triggered.connect(self._on_toggle_rlc_circuit)
        view_menu.addAction(rlc_act)

        thin_lens_act = QAction("🔬 Thấu kính mỏng & Băng quang học (Thin Lens & Optical Bench)", self)
        thin_lens_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        thin_lens_act.triggered.connect(self._on_toggle_thin_lens)
        view_menu.addAction(thin_lens_act)

        photoelectric_act = QAction("⚡ Hiệu ứng Quang điện & Thuyết Lượng tử (Photoelectric Effect)", self)
        photoelectric_act.setShortcut(QKeySequence("Ctrl+Alt+P"))
        photoelectric_act.triggered.connect(self._on_toggle_photoelectric_effect)
        view_menu.addAction(photoelectric_act)

        carnot_act = QAction("🔥 Động cơ nhiệt Carnot & Chu trình PV - TS (Carnot Heat Engine)", self)
        carnot_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_act.triggered.connect(self._on_toggle_carnot_engine)
        view_menu.addAction(carnot_act)
        bohr_act = QAction("⚛️ Mô hình Nguyên tử Bohr & Quang phổ Hydro (Bohr Atom & Hydrogen Spectrum)", self)
        bohr_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_act.triggered.connect(self._on_toggle_bohr_atom)
        view_menu.addAction(bohr_act)
        automata_act = QAction("🤖 Máy trạng thái hữu hạn & Automata DFA (Finite State Machine)", self)
        automata_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        automata_act.triggered.connect(self._on_toggle_automata)
        view_menu.addAction(automata_act)
        sorting_act = QAction("📊 Thuật toán sắp xếp & Độ phức tạp Big-O (Sorting Algorithms)", self)
        sorting_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        sorting_act.triggered.connect(self._on_toggle_sorting)
        view_menu.addAction(sorting_act)
        punnett_menu_act = QAction("🧬 Di truyền Mendel & Khung Punnett Square (Mendel Genetics)", self)
        punnett_menu_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        punnett_menu_act.triggered.connect(self._on_toggle_punnett_genetics)
        view_menu.addAction(punnett_menu_act)

        organic_act = QAction("🧪 Đồng phân Hóa học Hữu cơ & Nhóm chức (Organic Isomerism)", self)
        organic_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_act.triggered.connect(self._on_toggle_organic_chemistry)
        view_menu.addAction(organic_act)

        fluid_act = QAction("🌊 Cơ học chất lưu & Ống Venturi Bernoulli (Fluid Dynamics)", self)
        fluid_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_act.triggered.connect(self._on_toggle_fluid_venturi)
        view_menu.addAction(fluid_act)

        tree_act = QAction("🌳 Cây nhị phân & Mã hóa Huffman (Binary Tree & Huffman)", self)
        tree_act.setShortcut(QKeySequence("Ctrl+Alt+U"))
        tree_act.triggered.connect(self._on_toggle_binary_tree)
        view_menu.addAction(tree_act)

        electro_act = QAction("⚡ Pin điện hóa & Điện phân (Electrochemistry & Electrolysis)", self)
        electro_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        electro_act.triggered.connect(self._on_toggle_electrochemistry)
        view_menu.addAction(electro_act)

        circ_act = QAction("🔄 Chuyển động tròn đều & Lực hướng tâm (Circular Motion)", self)
        circ_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        circ_act.triggered.connect(self._on_toggle_circular_motion)
        view_menu.addAction(circ_act)

        titr_act = QAction("🧪 Chuẩn độ Axit - Bazơ & Đường cong pH (pH Titration)", self)
        titr_act.setShortcut(QKeySequence("Ctrl+Alt+I"))
        titr_act.triggered.connect(self._on_toggle_ph_titration)
        view_menu.addAction(titr_act)

        ind_act = QAction("🧲 Cảm ứng điện từ & Định luật Faraday (Electromagnetic Induction)", self)
        ind_act.setShortcut(QKeySequence("Ctrl+Alt+M"))
        ind_act.triggered.connect(self._on_toggle_electromagnetic_induction)
        view_menu.addAction(ind_act)

        osc_act = QAction("🌀 Dao động cơ học: Điều hòa, Tắt dần & Cộng hưởng (Harmonic Oscillator)", self)
        osc_act.setShortcut(QKeySequence("Ctrl+Alt+W"))
        osc_act.triggered.connect(self._on_toggle_harmonic_oscillator)
        view_menu.addAction(osc_act)

        gcms_act = QAction("🧪 Sắc ký khí & Khối phổ (GC-MS 70 eV)", self)
        gcms_act.setShortcut(QKeySequence("Ctrl+Alt+Y"))
        gcms_act.triggered.connect(self._on_toggle_gcms)
        view_menu.addAction(gcms_act)

        photo_act = QAction("🍃 Quang hợp: Pha sáng Thylakoid & Calvin-Benson (Photosynthesis)", self)
        photo_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photo_act.triggered.connect(self._on_toggle_photosynthesis)
        view_menu.addAction(photo_act)

        proj_act = QAction("🚀 Chuyển động ném & Động học 2D (Projectile Motion)", self)
        proj_act.setShortcut(QKeySequence("Ctrl+Alt+K"))
        proj_act.triggered.connect(self._on_toggle_projectile_motion)
        view_menu.addAction(proj_act)

        dp_act = QAction("📐 Quy hoạch động & Bảng trạng thái (Dynamic Programming)", self)
        dp_act.setShortcut(QKeySequence("Ctrl+Alt+D"))
        dp_act.triggered.connect(self._on_toggle_dynamic_programming)
        view_menu.addAction(dp_act)

        cnn_act = QAction("🧠 Mạng nơ-ron CNN & Tích chập 2D (Convolutional Neural Network)", self)
        cnn_act.setShortcut(QKeySequence("Ctrl+Alt+N"))
        cnn_act.triggered.connect(self._on_toggle_cnn_visualizer)
        view_menu.addAction(cnn_act)

        airfoil_act = QAction("✈️ Khí động học cánh máy bay & Lực nâng/cản (Airfoil Aerodynamics)", self)
        airfoil_act.setShortcut(QKeySequence("Ctrl+Alt+V"))
        airfoil_act.triggered.connect(self._on_toggle_airfoil_aerodynamics)
        view_menu.addAction(airfoil_act)

        periodic_act = QAction("⚛️ Bảng tuần hoàn & Cấu hình electron (Periodic Table)", self)
        periodic_act.setShortcut(QKeySequence("Ctrl+Alt+J"))
        periodic_act.triggered.connect(self._on_toggle_periodic_table)
        view_menu.addAction(periodic_act)

        respiration_act = QAction("⚡ Hô hấp tế bào & Chuỗi truyền electron ETC (Cellular Respiration)", self)
        respiration_act.setShortcut(QKeySequence("Ctrl+Alt+Z"))
        respiration_act.triggered.connect(self._on_toggle_cellular_respiration)
        view_menu.addAction(respiration_act)

        protein_act = QAction("🧬 Cấu trúc bậc hai Protein & Đồ thị Ramachandran (Protein Folding)", self)
        protein_act.setShortcut(QKeySequence("Ctrl+Alt+X"))
        protein_act.triggered.connect(self._on_toggle_protein_folding)
        view_menu.addAction(protein_act)

        mlp_act = QAction("🧠 Mạng nơ-ron MLP & Lan truyền ngược (MLP Backprop)", self)
        mlp_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        mlp_act.triggered.connect(self._on_toggle_mlp_backprop)
        view_menu.addAction(mlp_act)

        max_flow_act = QAction("🌊 Luồng cực đại & Lát cắt hẹp nhất (Max-Flow Min-Cut)", self)
        max_flow_act.setShortcut(QKeySequence("Ctrl+Alt+1"))
        max_flow_act.triggered.connect(self._on_toggle_max_flow)
        view_menu.addAction(max_flow_act)

        oxyz_act = QAction("📐 Hình học không gian Oxyz & Tích có hướng (Oxyz Geometry)", self)
        oxyz_act.setShortcut(QKeySequence("Ctrl+Alt+2"))
        oxyz_act.triggered.connect(self._on_toggle_oxyz_geometry)
        view_menu.addAction(oxyz_act)

        diode_act = QAction("⚡ Bán dẫn tiếp xúc P-N & Điốt Shockley (Semiconductor Diode)", self)
        diode_act.setShortcut(QKeySequence("Ctrl+Alt+3"))
        diode_act.triggered.connect(self._on_toggle_semiconductor_diode)
        view_menu.addAction(diode_act)

        nuclear_act = QAction("⚛️ Vật lý Hạt nhân & Năng lượng liên kết (Nuclear Reactions)", self)
        nuclear_act.setShortcut(QKeySequence("Ctrl+Alt+4"))
        nuclear_act.triggered.connect(self._on_toggle_nuclear_reactions)
        view_menu.addAction(nuclear_act)

        doppler_act = QAction("📢 Hiệu ứng Doppler & Nón sóng Mach (Doppler & Shockwave)", self)
        doppler_act.setShortcut(QKeySequence("Ctrl+Alt+5"))
        doppler_act.triggered.connect(self._on_toggle_doppler_effect)
        view_menu.addAction(doppler_act)

        rb_tree_act = QAction("🌲 Cây tự cân bằng Đỏ-Đen (Red-Black Tree)", self)
        rb_tree_act.setShortcut(QKeySequence("Ctrl+Alt+6"))
        rb_tree_act.triggered.connect(self._on_toggle_red_black_tree)
        view_menu.addAction(rb_tree_act)

        millikan_act = QAction("💧 Giọt dầu Millikan & Điện tích nguyên tố (Millikan Oil Drop)", self)
        millikan_act.setShortcut(QKeySequence("Ctrl+Alt+7"))
        millikan_act.triggered.connect(self._on_toggle_millikan_oil_drop)
        view_menu.addAction(millikan_act)

        quantum_well_act = QAction("⚛️ Giếng lượng tử & Sóng Schrödinger (Particle in a Box)", self)
        quantum_well_act.setShortcut(QKeySequence("Ctrl+Alt+8"))
        quantum_well_act.triggered.connect(self._on_toggle_quantum_well)
        view_menu.addAction(quantum_well_act)

        hohmann_act = QAction("🚀 Cơ học Thiên thể & Chuyển quỹ đạo Hohmann (Orbital Mechanics)", self)
        hohmann_act.setShortcut(QKeySequence("Ctrl+Alt+9"))
        hohmann_act.triggered.connect(self._on_toggle_hohmann_transfer)
        view_menu.addAction(hohmann_act)

        photoelectric_act = QAction("⚡ Hiện tượng Quang điện Ngoài & Photon Einstein (Photoelectric Effect)", self)
        photoelectric_act.setShortcut(QKeySequence("Ctrl+Alt+0"))
        photoelectric_act.triggered.connect(self._on_toggle_photoelectric_effect)
        view_menu.addAction(photoelectric_act)

        punnett_act = QAction("🧬 Di truyền Mendel & Khung Punnett (Mendelian Genetics)", self)
        punnett_act.setShortcut(QKeySequence("Ctrl+Shift+G"))
        punnett_act.triggered.connect(self._on_toggle_punnett_genetics)
        view_menu.addAction(punnett_act)

        kinetics_act = QAction("🧪 Động Hóa học & Định luật Arrhenius (Reaction Kinetics)", self)
        kinetics_act.setShortcut(QKeySequence("Ctrl+Shift+K"))
        kinetics_act.triggered.connect(self._on_toggle_reaction_kinetics)
        view_menu.addAction(kinetics_act)

        calculus_act = QAction("📐 Tích phân Riemann & Chuỗi Taylor (Riemann Calculus)", self)
        calculus_act.setShortcut(QKeySequence("Ctrl+Shift+R"))
        calculus_act.triggered.connect(self._on_toggle_calculus_taylor)
        view_menu.addAction(calculus_act)

        michelson_act = QAction("🔬 Giao thoa kế Michelson & Sóng Hấp dẫn LIGO (Michelson Interferometer)", self)
        michelson_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
        michelson_act.triggered.connect(self._on_toggle_michelson_interferometer)
        view_menu.addAction(michelson_act)

        huffman_act = QAction("🌲 Cây Mã hóa Huffman & Lý thuyết Thông tin Shannon (Huffman Coding)", self)
        huffman_act.setShortcut(QKeySequence("Ctrl+Shift+H"))
        huffman_act.triggered.connect(self._on_toggle_huffman_coding)
        view_menu.addAction(huffman_act)

        action_potential_act = QAction("⚡ Điện thế Hoạt động & Mô hình Hodgkin-Huxley (Action Potential)", self)
        action_potential_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
        action_potential_act.triggered.connect(self._on_toggle_action_potential)
        view_menu.addAction(action_potential_act)

        cyclotron_act = QAction("⚛️ Máy Gia tốc Hạt Cyclotron & Hiệu ứng Khối lượng γ (Cyclotron)", self)
        cyclotron_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        cyclotron_act.triggered.connect(self._on_toggle_cyclotron)
        view_menu.addAction(cyclotron_act)

        aho_corasick_act = QAction("🔤 Máy Tự động Tìm kiếm Chuỗi Aho-Corasick (Aho-Corasick Automaton)", self)
        aho_corasick_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        aho_corasick_act.triggered.connect(self._on_toggle_aho_corasick)
        view_menu.addAction(aho_corasick_act)

        phase_diagram_act = QAction("🌡️ Giản đồ Pha P-T & Clausius-Clapeyron (Phase Diagram)", self)
        phase_diagram_act.setShortcut(QKeySequence("Ctrl+Alt+P"))
        phase_diagram_act.triggered.connect(self._on_toggle_phase_diagram)
        view_menu.addAction(phase_diagram_act)

        em_wave_act = QAction("📡 Sóng Điện từ Maxwell & Vectơ Poynting (EM Waves)", self)
        em_wave_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        em_wave_act.triggered.connect(self._on_toggle_em_wave)
        view_menu.addAction(em_wave_act)

        hardy_weinberg_act = QAction("🧬 Cân bằng Di truyền Quần thể Hardy-Weinberg (Hardy-Weinberg)", self)
        hardy_weinberg_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_act.triggered.connect(self._on_toggle_hardy_weinberg)
        view_menu.addAction(hardy_weinberg_act)

        fft_act = QAction("📊 Biến đổi Fourier Nhanh FFT & Phổ Tần số (Fast Fourier Transform)", self)
        fft_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_act.triggered.connect(self._on_toggle_fft)
        view_menu.addAction(fft_act)

        fabry_perot_act = QAction("🔬 Giao thoa kế Fabry-Pérot & Hốc Cộng hưởng Laser (Fabry-Perot)", self)
        fabry_perot_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        fabry_perot_act.triggered.connect(self._on_toggle_fabry_perot)
        view_menu.addAction(fabry_perot_act)

        hplc_act = QAction("🧪 Sắc ký Lỏng HPLC & Hiệu suất Cột Van Deemter (HPLC)", self)
        hplc_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        hplc_act.triggered.connect(self._on_toggle_hplc)
        view_menu.addAction(hplc_act)

        photoelectric_act = QAction("⚡ Quang điện Einstein & Lượng tử Ánh sáng (Photoelectric)", self)
        photoelectric_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_act.triggered.connect(self._on_toggle_photoelectric)
        view_menu.addAction(photoelectric_act)

        pcr_act = QAction("🧬 Phản ứng Chuỗi Polymerase PCR & Điện di Gel (PCR)", self)
        pcr_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_act.triggered.connect(self._on_toggle_pcr)
        view_menu.addAction(pcr_act)

        thin_lens_act = QAction("🔍 Hệ Thấu kính Mỏng & Vẽ Tia sáng (Thin Lens)", self)
        thin_lens_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        thin_lens_act.triggered.connect(self._on_toggle_thin_lens)
        view_menu.addAction(thin_lens_act)

        bragg_act = QAction("💎 Nhiễu xạ Tia X Bragg & Tinh thể học (Bragg XRD)", self)
        bragg_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+B"))
        bragg_act.triggered.connect(self._on_toggle_bragg_diffraction)
        view_menu.addAction(bragg_act)

        buffer_act = QAction("🧪 Hệ Đệm Axit-Bazơ & Dung lượng Van Slyke (Buffer System)", self)
        buffer_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+H"))
        buffer_act.triggered.connect(self._on_toggle_buffer_system)
        view_menu.addAction(buffer_act)

        sg_act = QAction("⚛️ Thí nghiệm Stern-Gerlach & Spin 1/2 (Stern-Gerlach)", self)
        sg_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+S"))
        sg_act.triggered.connect(self._on_toggle_stern_gerlach)
        view_menu.addAction(sg_act)

        enzyme_act = QAction("🧪 Động học Enzyme Michaelis-Menten & Lineweaver-Burk (Enzyme Kinetics)", self)
        enzyme_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+E"))
        enzyme_act.triggered.connect(self._on_toggle_enzyme_kinetics)
        view_menu.addAction(enzyme_act)

        poiseuille_act = QAction("🌊 Dòng Chảy Nhớt Poiseuille & Số Reynolds (Poiseuille Flow)", self)
        poiseuille_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+P"))
        poiseuille_act.triggered.connect(self._on_toggle_poiseuille_flow)
        view_menu.addAction(poiseuille_act)

        cosmic_act = QAction("🌌 Định Luật Hubble & Vũ Trụ Giãn Nở (Cosmic Expansion)", self)
        cosmic_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+U"))
        cosmic_act.triggered.connect(self._on_toggle_cosmic_expansion)
        view_menu.addAction(cosmic_act)

        seismic_act = QAction("🌍 Sóng Địa Chấn & Định Vị Tâm Chấn (Seismic Waves)", self)
        seismic_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+Q"))
        seismic_act.triggered.connect(self._on_toggle_seismic_waves)
        view_menu.addAction(seismic_act)

        spectrometer_act = QAction("⚡ Máy Khối Phổ & Bộ Lọc Vận Tốc Wien (Mass Spectrometer)", self)
        spectrometer_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+M"))
        spectrometer_act.triggered.connect(self._on_toggle_mass_spectrometer)
        view_menu.addAction(spectrometer_act)

        spectrophotometry_act = QAction("🧪 Quang Phổ Hấp Thụ & Định Luật Beer-Lambert (Spectrophotometry)", self)
        spectrophotometry_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+L"))
        spectrophotometry_act.triggered.connect(self._on_toggle_spectrophotometry)
        view_menu.addAction(spectrophotometry_act)

        nuclear_binding_act = QAction("⚛ Năng Lượng Liên Kết Hạt Nhân & SEMF (Nuclear Binding Energy)", self)
        nuclear_binding_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+B"))
        nuclear_binding_act.triggered.connect(self._on_toggle_nuclear_binding_energy)
        view_menu.addAction(nuclear_binding_act)

        compton_act = QAction("⚛ Tán Xạ Compton Relativistic (Compton Scattering)", self)
        compton_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+Q"))
        compton_act.triggered.connect(self._on_toggle_compton_scattering)
        view_menu.addAction(compton_act)

        lensing_act = QAction("🌌 Thấu Kính Hấp Dẫn & Nhẫn Einstein (Gravitational Lensing)", self)
        lensing_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+G"))
        lensing_act.triggered.connect(self._on_toggle_gravitational_lensing)
        view_menu.addAction(lensing_act)

        equilibrium_act = QAction("⚗ Cân Bằng Hóa Học & Le Chatelier (Chemical Equilibrium)", self)
        equilibrium_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+E"))
        equilibrium_act.triggered.connect(self._on_toggle_chemical_equilibrium)
        view_menu.addAction(equilibrium_act)

        hodgkin_act = QAction("🧠 Điện Sinh Lý Nơ-ron Hodgkin-Huxley (Action Potential)", self)
        hodgkin_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+H"))
        hodgkin_act.triggered.connect(self._on_toggle_hodgkin_huxley)
        view_menu.addAction(hodgkin_act)

        mst_act = QAction("🌲 Cây Khung Nhỏ Nhất Kruskal & Prim (Minimum Spanning Tree)", self)
        mst_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+M"))
        mst_act.triggered.connect(self._on_toggle_mst_visualizer)
        view_menu.addAction(mst_act)

        hall_act = QAction("⚡ Hiệu Ứng Hall & Vật Lý Bán Dẫn (Hall Effect)", self)
        hall_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+F"))
        hall_act.triggered.connect(self._on_toggle_hall_effect)
        view_menu.addAction(hall_act)

        rad_act = QAction("☢ Độ Xuyên Sâu Phóng Xạ & Tấm Chắn HVL (Radiation Shielding)", self)
        rad_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+R"))
        rad_act.triggered.connect(self._on_toggle_radiation_shielding)
        view_menu.addAction(rad_act)

        maxwell_act = QAction("🔥 Phân Bố Vận Tốc Phân Tử Maxwell-Boltzmann", self)
        maxwell_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+X"))
        maxwell_act.triggered.connect(self._on_toggle_maxwell_boltzmann)
        view_menu.addAction(maxwell_act)

        mo_act = QAction("⚛ Obitan Phân Tử & Thuyết MO-LCAO (Molecular Orbital)", self)
        mo_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+O"))
        mo_act.triggered.connect(self._on_toggle_molecular_orbital)
        view_menu.addAction(mo_act)

        rlc_act = QAction("⚡ Cộng Hưởng Điện RLC & Giản Đồ Phasor", self)
        rlc_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+C"))
        rlc_act.triggered.connect(self._on_toggle_rlc_resonance)
        view_menu.addAction(rlc_act)

        stress_act = QAction("⚙ Đường Cong Ứng Suất - Biến Dạng (Stress-Strain Curve)", self)
        stress_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+T"))
        stress_act.triggered.connect(self._on_toggle_stress_strain)
        view_menu.addAction(stress_act)

        kepler_act = QAction("🪐 Quỹ Đạo Hấp Dẫn & 3 Định Luật Kepler (Kepler Orbits)", self)
        kepler_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+K"))
        kepler_act.triggered.connect(self._on_toggle_kepler_orbits)
        view_menu.addAction(kepler_act)

        logic_act = QAction("🔌 Điện Tử Số & Cổng Logic (Digital Logic & Circuits)", self)
        logic_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+G"))
        logic_act.triggered.connect(self._on_toggle_logic_circuits)
        view_menu.addAction(logic_act)

        laser_color_act = QAction("🔴 Đổi màu tia Laser (Laser Color)", self)
        laser_color_act.setShortcut(QKeySequence("Ctrl+L"))
        laser_color_act.triggered.connect(self._on_cycle_laser_color)
        view_menu.addAction(laser_color_act)

        tools_menu = mb.addMenu("Công cụ")
        wb_act = QAction("✨ Trạm soạn thảo & Trợ giảng AI (AI Workbench)…", self)
        wb_act.triggered.connect(self._on_open_ai_workbench)
        tools_menu.addAction(wb_act)

        diag_act = QAction("🔬 Chẩn đoán & Đo lường Camera…", self)
        diag_act.triggered.connect(self._on_open_camera_diagnostics)
        tools_menu.addAction(diag_act)

        trainer_act = QAction(_get_icon("camera"), "Luyện tập & Hiệu chuẩn Cử chỉ…", self)
        trainer_act.setShortcut(Qt.Key.Key_T)
        trainer_act.triggered.connect(self._on_open_gesture_trainer)
        tools_menu.addAction(trainer_act)

        calib_act = QAction("🎯 Hiệu chuẩn Cử chỉ & Khoảng cách (Gesture Ergonomics)…", self)
        calib_act.triggered.connect(self._on_open_gesture_calibration)
        tools_menu.addAction(calib_act)

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

        export_img_tb_act = QAction("🖼️ Xuất ảnh", self)
        export_img_tb_act.setToolTip("Xuất ảnh bảng vẽ độ phân giải cao (1080p / 4K UHD, PNG / JPG) (Ctrl+Shift+E)")
        export_img_tb_act.triggered.connect(self._on_export_image)
        nav_tb.addAction(export_img_tb_act)

        board_theme_tb_act = QAction("🎨 Bảng", self)
        board_theme_tb_act.setToolTip("Chuyển đổi Bảng trắng / Bảng xanh phấn / Vở 4 ô ly / Bảng tối (Ctrl+B)")
        board_theme_tb_act.triggered.connect(self._on_toggle_board_theme)
        nav_tb.addAction(board_theme_tb_act)

        slide_tb_act = QAction("🎞️ Slides", self)
        slide_tb_act.setToolTip("Bật/tắt ngăn kéo danh sách slide bài giảng (F2)")
        slide_tb_act.triggered.connect(self._on_toggle_slide_dock)
        nav_tb.addAction(slide_tb_act)

        sticky_tb_act = QAction("📝 Ghi chú", self)
        sticky_tb_act.setToolTip("Thêm thẻ ghi chú dán Acrylic trên bảng (Ctrl+N)")
        sticky_tb_act.triggered.connect(self._on_add_sticky_note)
        nav_tb.addAction(sticky_tb_act)

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

        grid_tb_act = QAction(_get_icon("grid"), "Lưới Oxy", self)
        grid_tb_act.setToolTip("Bật/Tắt hệ trục tọa độ Oxy & lưới ô vuông (Ctrl+G)")
        grid_tb_act.triggered.connect(self._on_toggle_grid)
        nav_tb.addAction(grid_tb_act)

        ruler_tb_act = QAction("📐 Thước kẻ", self)
        ruler_tb_act.setToolTip("Bật/Tắt thước kẻ đo centimet trên bảng (Ctrl+R)")
        ruler_tb_act.triggered.connect(self._on_toggle_ruler)
        nav_tb.addAction(ruler_tb_act)

        protractor_tb_act = QAction("🧭 Đo độ", self)
        protractor_tb_act.setToolTip("Bật/Tắt thước đo góc 180° trên bảng (Ctrl+Shift+R)")
        protractor_tb_act.triggered.connect(self._on_toggle_protractor)
        nav_tb.addAction(protractor_tb_act)

        compass_tb_act = QAction("📐 Compa", self)
        compass_tb_act.setToolTip("Bật/Tắt compa vẽ đường tròn & cung tròn chính xác (Ctrl+Shift+C)")
        compass_tb_act.setShortcut(QKeySequence("Ctrl+Shift+C"))
        compass_tb_act.triggered.connect(self._on_toggle_compass)
        nav_tb.addAction(compass_tb_act)

        spotlight_tb_act = QAction("🔦 Đèn rọi", self)
        spotlight_tb_act.setToolTip("Bật/Tắt chế độ đèn rọi Spotlight tập trung sự chú ý (Phím S)")
        spotlight_tb_act.setShortcut(QKeySequence("S"))
        spotlight_tb_act.triggered.connect(self._on_toggle_spotlight)
        nav_tb.addAction(spotlight_tb_act)

        loupe_tb_act = QAction("🔍 Kính lúp", self)
        loupe_tb_act.setToolTip("Bật/Tắt kính lúp phóng đại chi tiết (Phím M)")
        loupe_tb_act.setShortcut(QKeySequence("M"))
        loupe_tb_act.triggered.connect(self._on_toggle_loupe)
        nav_tb.addAction(loupe_tb_act)

        timer_tb_act = QAction("⏱️ Hẹn giờ", self)
        timer_tb_act.setToolTip("Bật/Tắt đồng hồ đếm ngược bài giảng & bấm giờ thảo luận (Phím T)")
        timer_tb_act.setShortcut(QKeySequence("T"))
        timer_tb_act.triggered.connect(self._on_toggle_timer)
        nav_tb.addAction(timer_tb_act)

        metronome_tb_act = QAction("🎵 Nhịp", self)
        metronome_tb_act.setToolTip("Máy đếm nhịp & luyện đọc theo tiết tấu (Ctrl+M)")
        metronome_tb_act.setShortcut(QKeySequence("Ctrl+M"))
        metronome_tb_act.triggered.connect(self._on_toggle_metronome)
        nav_tb.addAction(metronome_tb_act)

        minimap_tb_act = QAction("🗺️ Bản đồ", self)
        minimap_tb_act.setToolTip("Bật/Tắt bản đồ thu nhỏ toàn cảnh (Ctrl+Shift+V)")
        minimap_tb_act.setShortcut(QKeySequence("Ctrl+Shift+V"))
        minimap_tb_act.triggered.connect(self._on_toggle_minimap)
        nav_tb.addAction(minimap_tb_act)

        coord_tb_act = QAction("📍 Tọa độ Oxy", self)
        coord_tb_act.setToolTip("Bật/Tắt hiển thị tọa độ Oxy thời gian thực (Ctrl+Shift+X)")
        coord_tb_act.setShortcut(QKeySequence("Ctrl+Shift+X"))
        coord_tb_act.triggered.connect(self._on_toggle_coordinate_hud)
        nav_tb.addAction(coord_tb_act)

        angle_tb_act = QAction("📐 Đo góc động", self)
        angle_tb_act.setToolTip("Bật/Tắt thước đo góc hình học động (Ctrl+Shift+A)")
        angle_tb_act.setShortcut(QKeySequence("Ctrl+Shift+A"))
        angle_tb_act.triggered.connect(self._on_toggle_angle_tool)
        nav_tb.addAction(angle_tb_act)

        triangle_tb_act = QAction("🔺 Tam giác động", self)
        triangle_tb_act.setToolTip("Bật/Tắt tam giác hình học động & đường Euler (Ctrl+Shift+T)")
        triangle_tb_act.setShortcut(QKeySequence("Ctrl+Shift+T"))
        triangle_tb_act.triggered.connect(self._on_toggle_triangle_tool)
        nav_tb.addAction(triangle_tb_act)

        func_tb_act = QAction("📈 Khảo sát f(x)", self)
        func_tb_act.setToolTip("Bật/Tắt đồ thị hàm số & tiếp tuyến động (Ctrl+Shift+F)")
        func_tb_act.setShortcut(QKeySequence("Ctrl+Shift+F"))
        func_tb_act.triggered.connect(self._on_toggle_function_plotter)
        nav_tb.addAction(func_tb_act)

        incline_tb_act = QAction("⚡ Hệ lực dốc", self)
        incline_tb_act.setToolTip("Bật/Tắt hệ lực mặt phẳng nghiêng (Ctrl+Shift+I)")
        incline_tb_act.setShortcut(QKeySequence("Ctrl+Shift+I"))
        incline_tb_act.triggered.connect(self._on_toggle_inclined_plane)
        nav_tb.addAction(incline_tb_act)

        optics_tb_act = QAction("🌈 Khúc xạ ánh sáng", self)
        optics_tb_act.setToolTip("Bật/Tắt mô phỏng khúc xạ ánh sáng & phản xạ toàn phần (Ctrl+Shift+O)")
        optics_tb_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        optics_tb_act.triggered.connect(self._on_toggle_optics_refraction)
        nav_tb.addAction(optics_tb_act)

        wave_tb_act = QAction("🌊 Sóng dừng", self)
        wave_tb_act.setToolTip("Bật/Tắt mô phỏng sóng cơ & sóng dừng (Ctrl+Shift+W)")
        wave_tb_act.setShortcut(QKeySequence("Ctrl+Shift+W"))
        wave_tb_act.triggered.connect(self._on_toggle_standing_wave)
        nav_tb.addAction(wave_tb_act)

        vsepr_tb_act = QAction("⚛️ Hình học VSEPR", self)
        vsepr_tb_act.setToolTip("Bật/Tắt mô phỏng 3D hình học phân tử VSEPR & Lai hóa orbital (Ctrl+Shift+Y)")
        vsepr_tb_act.setShortcut(QKeySequence("Ctrl+Shift+Y"))
        vsepr_tb_act.triggered.connect(self._on_toggle_vsepr_geometry)
        nav_tb.addAction(vsepr_tb_act)

        dna_tb_act = QAction("🧬 Tự nhân đôi DNA", self)
        dna_tb_act.setToolTip("Bật/Tắt mô phỏng cơ chế tự nhân đôi DNA & nguyên tắc bổ sung (Ctrl+Shift+D)")
        dna_tb_act.setShortcut(QKeySequence("Ctrl+Shift+D"))
        dna_tb_act.triggered.connect(self._on_toggle_dna_replication)
        nav_tb.addAction(dna_tb_act)

        kepler_tb_act = QAction("🪐 Quỹ đạo Kepler", self)
        kepler_tb_act.setToolTip("Bật/Tắt mô phỏng 3 định luật Kepler & chuyển động thiên thể (Ctrl+Shift+K)")
        kepler_tb_act.setShortcut(QKeySequence("Ctrl+Shift+K"))
        kepler_tb_act.triggered.connect(self._on_toggle_kepler_orbit)
        nav_tb.addAction(kepler_tb_act)

        graph_tb_act = QAction("🕸️ Đồ thị Dijkstra", self)
        graph_tb_act.setToolTip("Bật/Tắt mô phỏng Lý thuyết đồ thị & Thuật toán tìm đường ngắn nhất Dijkstra / A* (Ctrl+Shift+P)")
        graph_tb_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
        graph_tb_act.triggered.connect(self._on_toggle_graph_pathfinding)
        nav_tb.addAction(graph_tb_act)

        electric_field_tb_act = QAction("⚡ Điện trường Coulomb", self)
        electric_field_tb_act.setToolTip("Bật/Tắt mô phỏng Đường sức điện trường & Định luật Coulomb tĩnh điện (Ctrl+Shift+E)")
        electric_field_tb_act.setShortcut(QKeySequence("Ctrl+Shift+E"))
        electric_field_tb_act.triggered.connect(self._on_toggle_electric_field)
        nav_tb.addAction(electric_field_tb_act)

        lorentz_tb_act = QAction("🧲 Lực Lorentz", self)
        lorentz_tb_act.setToolTip("Bật/Tắt mô phỏng Từ trường & Lực từ Lorentz làm lệch hạt mang điện (Ctrl+Shift+B)")
        lorentz_tb_act.setShortcut(QKeySequence("Ctrl+Shift+B"))
        lorentz_tb_act.triggered.connect(self._on_toggle_lorentz_force)
        nav_tb.addAction(lorentz_tb_act)

        trophic_tb_act = QAction("🌿 Lưới thức ăn", self)
        trophic_tb_act.setToolTip("Bật/Tắt mô phỏng Lưới thức ăn & Cân bằng sinh thái Lotka-Volterra (Ctrl+Shift+U)")
        trophic_tb_act.setShortcut(QKeySequence("Ctrl+Shift+U"))
        trophic_tb_act.triggered.connect(self._on_toggle_trophic_web)
        nav_tb.addAction(trophic_tb_act)

        gas_tb_act = QAction("🎈 Khí lý tưởng", self)
        gas_tb_act.setToolTip("Bật/Tắt mô phỏng Định luật khí lý tưởng Clapeyron & Động học phân tử Maxwell-Boltzmann (Ctrl+Shift+J)")
        gas_tb_act.setShortcut(QKeySequence("Ctrl+Shift+J"))
        gas_tb_act.triggered.connect(self._on_toggle_gas_laws)
        nav_tb.addAction(gas_tb_act)

        titration_tb_act = QAction("🧪 Chuẩn độ pH", self)
        titration_tb_act.setToolTip("Bật/Tắt mô phỏng Chuẩn độ Axit - Bazơ & Đường cong pH đệm Henderson-Hasselbalch (Ctrl+Shift+Q)")
        titration_tb_act.setShortcut(QKeySequence("Ctrl+Shift+Q"))
        titration_tb_act.triggered.connect(self._on_toggle_titration)
        nav_tb.addAction(titration_tb_act)

        oscillator_tb_act = QAction("🧲 Dao động cơ", self)
        oscillator_tb_act.setToolTip("Bật/Tắt mô phỏng Dao động điều hòa, Tắt dần & Cộng hưởng cơ học (Ctrl+Shift+H)")
        oscillator_tb_act.setShortcut(QKeySequence("Ctrl+Shift+H"))
        oscillator_tb_act.triggered.connect(self._on_toggle_harmonic_oscillator)
        nav_tb.addAction(oscillator_tb_act)

        decay_tb_act = QAction("☢️ Phóng xạ", self)
        decay_tb_act.setToolTip("Bật/Tắt mô phỏng Định luật phóng xạ hạt nhân & Chuỗi phân rã Monte Carlo (Ctrl+Alt+R)")
        decay_tb_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        decay_tb_act.triggered.connect(self._on_toggle_radioactive_decay)
        nav_tb.addAction(decay_tb_act)

        kinetics_tb_act = QAction("🧪 Động hóa học", self)
        kinetics_tb_act.setToolTip("Bật/Tắt mô phỏng Tốc độ phản ứng, Năng lượng hoạt hóa Arrhenius & Cân bằng hóa học (Ctrl+Alt+K)")
        kinetics_tb_act.setShortcut(QKeySequence("Ctrl+Alt+K"))
        kinetics_tb_act.triggered.connect(self._on_toggle_chemical_kinetics)
        nav_tb.addAction(kinetics_tb_act)

        crypto_tb_act = QAction("🔐 Mật mã RSA", self)
        crypto_tb_act.setToolTip("Bật/Tắt mô phỏng Mật mã khóa công khai RSA & Vòng tròn số học đồng dư (Ctrl+Alt+C)")
        crypto_tb_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        crypto_tb_act.triggered.connect(self._on_toggle_cryptography)
        nav_tb.addAction(crypto_tb_act)

        orbits_tb_act = QAction("🌌 Quỹ đạo hấp dẫn", self)
        orbits_tb_act.setToolTip("Bật/Tắt mô phỏng Quỹ đạo vạn vật hấp dẫn N-vật thể & Điểm Lagrange (Ctrl+Alt+G)")
        orbits_tb_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        orbits_tb_act.triggered.connect(self._on_toggle_gravitational_orbits)
        nav_tb.addAction(orbits_tb_act)

        relativity_tb_act = QAction("⚡ Minkowski", self)
        relativity_tb_act.setToolTip("Bật/Tắt mô phỏng Không-thời gian Minkowski & Biến đổi Lorentz (Ctrl+Alt+M)")
        relativity_tb_act.setShortcut(QKeySequence("Ctrl+Alt+M"))
        relativity_tb_act.triggered.connect(self._on_toggle_spacetime_relativity)
        nav_tb.addAction(relativity_tb_act)

        young_tb_act = QAction("🌈 Giao thoa sóng", self)
        young_tb_act.setToolTip("Bật/Tắt mô phỏng Giao thoa sóng ánh sáng qua khe Young & Cách tử nhiễu xạ (Ctrl+Alt+Y)")
        young_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Y"))
        young_tb_act.triggered.connect(self._on_toggle_young_interference)
        nav_tb.addAction(young_tb_act)

        rlc_tb_act = QAction("⚡ Mạch RLC", self)
        rlc_tb_act.setToolTip("Bật/Tắt mô phỏng Mạch điện xoay chiều RLC nối tiếp & Giản đồ Fresnel (Ctrl+Alt+E)")
        rlc_tb_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        rlc_tb_act.triggered.connect(self._on_toggle_rlc_circuit)
        nav_tb.addAction(rlc_tb_act)

        thin_lens_tb_act = QAction("🔬 Thấu kính mỏng", self)
        thin_lens_tb_act.setToolTip("Bật/Tắt mô phỏng Thấu kính mỏng & Băng quang học vẽ tia sáng (Ctrl+Alt+O)")
        thin_lens_tb_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        thin_lens_tb_act.triggered.connect(self._on_toggle_thin_lens)
        nav_tb.addAction(thin_lens_tb_act)

        photoelectric_tb_act = QAction("⚡ Quang điện", self)
        photoelectric_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu ứng Quang điện & Thuyết Lượng tử Ánh sáng Einstein (Ctrl+Alt+P)")
        photoelectric_tb_act.setShortcut(QKeySequence("Ctrl+Alt+P"))
        photoelectric_tb_act.triggered.connect(self._on_toggle_photoelectric_effect)
        nav_tb.addAction(photoelectric_tb_act)

        carnot_tb_act = QAction("🔥 Chu trình Carnot", self)
        carnot_tb_act.setToolTip("Bật/Tắt mô phỏng Động cơ nhiệt lý tưởng Carnot & Đồ thị PV / TS (Ctrl+Alt+T)")
        carnot_tb_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_tb_act.triggered.connect(self._on_toggle_carnot_engine)
        nav_tb.addAction(carnot_tb_act)
        bohr_tb_act = QAction("⚛️ Nguyên tử Bohr", self)
        bohr_tb_act.setToolTip("Bật/Tắt mô phỏng Mô hình Nguyên tử Bohr & Dãy vạch quang phổ Hydro (Ctrl+Alt+B)")
        bohr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_tb_act.triggered.connect(self._on_toggle_bohr_atom)
        nav_tb.addAction(bohr_tb_act)
        automata_tb_act = QAction("🤖 Máy trạng thái", self)
        automata_tb_act.setToolTip("Bật/Tắt mô phỏng Máy trạng thái hữu hạn & Automata DFA/NFA (Ctrl+Alt+A)")
        automata_tb_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        automata_tb_act.triggered.connect(self._on_toggle_automata)
        nav_tb.addAction(automata_tb_act)
        sorting_tb_act = QAction("📊 Sắp xếp Big-O", self)
        sorting_tb_act.setToolTip("Bật/Tắt mô phỏng Thuật toán sắp xếp & Bảng độ phức tạp Big-O (Ctrl+Alt+S)")
        sorting_tb_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        sorting_tb_act.triggered.connect(self._on_toggle_sorting)
        nav_tb.addAction(sorting_tb_act)
        punnett_tb_act = QAction("🧬 Di truyền Mendel", self)
        punnett_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền Mendel & Khung Punnett Square (Ctrl+Alt+G)")
        punnett_tb_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        punnett_tb_act.triggered.connect(self._on_toggle_punnett_genetics)
        nav_tb.addAction(punnett_tb_act)
        organic_tb_act = QAction("🧪 Đồng phân Hóa học", self)
        organic_tb_act.setToolTip("Bật/Tắt mô phỏng Đồng phân Hóa học Hữu cơ & Nhóm chức (Ctrl+Alt+H)")
        organic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_tb_act.triggered.connect(self._on_toggle_organic_chemistry)
        nav_tb.addAction(organic_tb_act)

        fluid_tb_act = QAction("🌊 Cơ học chất lưu", self)
        fluid_tb_act.setToolTip("Bật/Tắt mô phỏng Cơ học chất lưu & Ống Venturi Bernoulli (Ctrl+Alt+F)")
        fluid_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_tb_act.triggered.connect(self._on_toggle_fluid_venturi)
        nav_tb.addAction(fluid_tb_act)

        tree_tb_act = QAction("🌳 Cây nhị phân", self)
        tree_tb_act.setToolTip("Bật/Tắt mô phỏng Cây nhị phân tìm kiếm & Mã hóa Huffman (Ctrl+Alt+U)")
        tree_tb_act.setShortcut(QKeySequence("Ctrl+Alt+U"))
        tree_tb_act.triggered.connect(self._on_toggle_binary_tree)
        nav_tb.addAction(tree_tb_act)

        electro_tb_act = QAction("⚡ Điện hóa & Điện phân", self)
        electro_tb_act.setToolTip("Bật/Tắt mô phỏng Pin Galvani & Bình điện phân (Ctrl+Alt+E)")
        electro_tb_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        electro_tb_act.triggered.connect(self._on_toggle_electrochemistry)
        nav_tb.addAction(electro_tb_act)

        circ_tb_act = QAction("🔄 Chuyển động tròn", self)
        circ_tb_act.setToolTip("Bật/Tắt mô phỏng Chuyển động tròn đều & Lực hướng tâm (Ctrl+Alt+C)")
        circ_tb_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        circ_tb_act.triggered.connect(self._on_toggle_circular_motion)
        nav_tb.addAction(circ_tb_act)

        titr_tb_act = QAction("🧪 Chuẩn độ pH", self)
        titr_tb_act.setToolTip("Bật/Tắt mô phỏng Chuẩn độ Axit - Bazơ & Đường cong pH (Ctrl+Alt+I)")
        titr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+I"))
        titr_tb_act.triggered.connect(self._on_toggle_ph_titration)
        nav_tb.addAction(titr_tb_act)

        ind_tb_act = QAction("🧲 Cảm ứng điện từ", self)
        ind_tb_act.setToolTip("Bật/Tắt mô phỏng Cảm ứng điện từ & Định luật Faraday (Ctrl+Alt+M)")
        ind_tb_act.setShortcut(QKeySequence("Ctrl+Alt+M"))
        ind_tb_act.triggered.connect(self._on_toggle_electromagnetic_induction)
        nav_tb.addAction(ind_tb_act)

        osc_tb_act = QAction("🌀 Dao động & Cộng hưởng", self)
        osc_tb_act.setToolTip("Bật/Tắt mô phỏng Dao động cơ học: Điều hòa, Tắt dần & Cộng hưởng (Ctrl+Alt+W)")
        osc_tb_act.setShortcut(QKeySequence("Ctrl+Alt+W"))
        osc_tb_act.triggered.connect(self._on_toggle_harmonic_oscillator)
        nav_tb.addAction(osc_tb_act)

        gcms_tb_act = QAction("🧪 Sắc ký GC-MS", self)
        gcms_tb_act.setToolTip("Bật/Tắt mô phỏng Sắc ký khí & Khối phổ GC-MS 70 eV (Ctrl+Alt+Y)")
        gcms_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Y"))
        gcms_tb_act.triggered.connect(self._on_toggle_gcms)
        nav_tb.addAction(gcms_tb_act)

        photo_tb_act = QAction("🍃 Quang hợp & Calvin", self)
        photo_tb_act.setToolTip("Bật/Tắt mô phỏng Quang hợp: Pha sáng Thylakoid & Chu trình Calvin (Ctrl+Alt+Q)")
        photo_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photo_tb_act.triggered.connect(self._on_toggle_photosynthesis)
        nav_tb.addAction(photo_tb_act)

        proj_tb_act = QAction("🚀 Chuyển động ném", self)
        proj_tb_act.setToolTip("Bật/Tắt mô phỏng Chuyển động ném & Động học 2D (Ctrl+Alt+K)")
        proj_tb_act.setShortcut(QKeySequence("Ctrl+Alt+K"))
        proj_tb_act.triggered.connect(self._on_toggle_projectile_motion)
        nav_tb.addAction(proj_tb_act)

        dp_tb_act = QAction("📐 Quy hoạch động", self)
        dp_tb_act.setToolTip("Bật/Tắt mô phỏng Quy hoạch động (Knapsack, LCS, Coin Change) (Ctrl+Alt+D)")
        dp_tb_act.setShortcut(QKeySequence("Ctrl+Alt+D"))
        dp_tb_act.triggered.connect(self._on_toggle_dynamic_programming)
        nav_tb.addAction(dp_tb_act)

        cnn_tb_act = QAction("🧠 Mạng nơ-ron CNN", self)
        cnn_tb_act.setToolTip("Bật/Tắt mô phỏng Mạng nơ-ron CNN & Tích chập 2D (Ctrl+Alt+N)")
        cnn_tb_act.setShortcut(QKeySequence("Ctrl+Alt+N"))
        cnn_tb_act.triggered.connect(self._on_toggle_cnn_visualizer)
        nav_tb.addAction(cnn_tb_act)

        airfoil_tb_act = QAction("✈️ Khí động học cánh", self)
        airfoil_tb_act.setToolTip("Bật/Tắt mô phỏng Khí động học cánh máy bay & Lực nâng/cản NACA (Ctrl+Alt+V)")
        airfoil_tb_act.setShortcut(QKeySequence("Ctrl+Alt+V"))
        airfoil_tb_act.triggered.connect(self._on_toggle_airfoil_aerodynamics)
        nav_tb.addAction(airfoil_tb_act)

        periodic_tb_act = QAction("⚛️ Bảng tuần hoàn", self)
        periodic_tb_act.setToolTip("Bật/Tắt mô phỏng Bảng tuần hoàn nguyên tố & Cấu hình electron Bohr (Ctrl+Alt+J)")
        periodic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+J"))
        periodic_tb_act.triggered.connect(self._on_toggle_periodic_table)
        nav_tb.addAction(periodic_tb_act)

        respiration_tb_act = QAction("⚡ Hô hấp tế bào", self)
        respiration_tb_act.setToolTip("Bật/Tắt mô phỏng Hô hấp tế bào, Krebs & Chuỗi chuyền electron ti thể ETC (Ctrl+Alt+Z)")
        respiration_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Z"))
        respiration_tb_act.triggered.connect(self._on_toggle_cellular_respiration)
        nav_tb.addAction(respiration_tb_act)

        protein_tb_act = QAction("🧬 Gấp nếp Protein", self)
        protein_tb_act.setToolTip("Bật/Tắt mô phỏng Cấu trúc bậc hai Protein & Đồ thị Ramachandran (Ctrl+Alt+X)")
        protein_tb_act.setShortcut(QKeySequence("Ctrl+Alt+X"))
        protein_tb_act.triggered.connect(self._on_toggle_protein_folding)
        nav_tb.addAction(protein_tb_act)

        mlp_tb_act = QAction("🧠 Mạng MLP Backprop", self)
        mlp_tb_act.setToolTip("Bật/Tắt mô phỏng Mạng nơ-ron MLP, Lan truyền ngược & Biên quyết định (Ctrl+Alt+L)")
        mlp_tb_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        mlp_tb_act.triggered.connect(self._on_toggle_mlp_backprop)
        nav_tb.addAction(mlp_tb_act)

        max_flow_tb_act = QAction("🌊 Luồng cực đại", self)
        max_flow_tb_act.setToolTip("Bật/Tắt mô phỏng Luồng cực đại trên mạng & Lát cắt Min-Cut (Ctrl+Alt+1)")
        max_flow_tb_act.setShortcut(QKeySequence("Ctrl+Alt+1"))
        max_flow_tb_act.triggered.connect(self._on_toggle_max_flow)
        nav_tb.addAction(max_flow_tb_act)

        oxyz_tb_act = QAction("📐 Hình học Oxyz", self)
        oxyz_tb_act.setToolTip("Bật/Tắt mô phỏng Hình học không gian Oxyz & Tích có hướng Vector (Ctrl+Alt+2)")
        oxyz_tb_act.setShortcut(QKeySequence("Ctrl+Alt+2"))
        oxyz_tb_act.triggered.connect(self._on_toggle_oxyz_geometry)
        nav_tb.addAction(oxyz_tb_act)

        diode_tb_act = QAction("⚡ Điốt Bán dẫn", self)
        diode_tb_act.setToolTip("Bật/Tắt mô phỏng Bán dẫn P-N, Điốt Shockley & Vùng nghèo (Ctrl+Alt+3)")
        diode_tb_act.setShortcut(QKeySequence("Ctrl+Alt+3"))
        diode_tb_act.triggered.connect(self._on_toggle_semiconductor_diode)
        nav_tb.addAction(diode_tb_act)

        nuclear_tb_act = QAction("⚛️ Hạt nhân", self)
        nuclear_tb_act.setToolTip("Bật/Tắt mô phỏng Vật lý Hạt nhân, Eb/A, Phân hạch & Nhiệt hạch (Ctrl+Alt+4)")
        nuclear_tb_act.setShortcut(QKeySequence("Ctrl+Alt+4"))
        nuclear_tb_act.triggered.connect(self._on_toggle_nuclear_reactions)
        nav_tb.addAction(nuclear_tb_act)

        doppler_tb_act = QAction("📢 Sóng Doppler", self)
        doppler_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu ứng Doppler, Nén bước sóng & Nón Mach (Ctrl+Alt+5)")
        doppler_tb_act.setShortcut(QKeySequence("Ctrl+Alt+5"))
        doppler_tb_act.triggered.connect(self._on_toggle_doppler_effect)
        nav_tb.addAction(doppler_tb_act)

        rb_tree_tb_act = QAction("🌲 Cây Đỏ-Đen", self)
        rb_tree_tb_act.setToolTip("Bật/Tắt mô phỏng Cây tự cân bằng Đỏ-Đen & CLRS Invariants (Ctrl+Alt+6)")
        rb_tree_tb_act.setShortcut(QKeySequence("Ctrl+Alt+6"))
        rb_tree_tb_act.triggered.connect(self._on_toggle_red_black_tree)
        nav_tb.addAction(rb_tree_tb_act)

        millikan_tb_act = QAction("💧 Giọt dầu Millikan", self)
        millikan_tb_act.setToolTip("Bật/Tắt mô phỏng Giọt dầu Millikan, Lực cản Stokes & Điện tích e (Ctrl+Alt+7)")
        millikan_tb_act.setShortcut(QKeySequence("Ctrl+Alt+7"))
        millikan_tb_act.triggered.connect(self._on_toggle_millikan_oil_drop)
        nav_tb.addAction(millikan_tb_act)

        quantum_well_tb_act = QAction("⚛️ Giếng lượng tử", self)
        quantum_well_tb_act.setToolTip("Bật/Tắt mô phỏng Giếng thế vô hạn, Sóng dừng ψ và Xác suất |ψ|² (Ctrl+Alt+8)")
        quantum_well_tb_act.setShortcut(QKeySequence("Ctrl+Alt+8"))
        quantum_well_tb_act.triggered.connect(self._on_toggle_quantum_well)
        nav_tb.addAction(quantum_well_tb_act)

        hohmann_tb_act = QAction("🚀 Quỹ đạo Hohmann", self)
        hohmann_tb_act.setToolTip("Bật/Tắt mô phỏng Cơ học thiên thể Kepler & Chuyển quỹ đạo Hohmann (Ctrl+Alt+9)")
        hohmann_tb_act.setShortcut(QKeySequence("Ctrl+Alt+9"))
        hohmann_tb_act.triggered.connect(self._on_toggle_hohmann_transfer)
        nav_tb.addAction(hohmann_tb_act)

        photoelectric_tb_act = QAction("⚡ Quang điện Ngoài", self)
        photoelectric_tb_act.setToolTip("Bật/Tắt mô phỏng Hiện tượng Quang điện Ngoài, Công thoát & Photon Einstein (Ctrl+Alt+0)")
        photoelectric_tb_act.setShortcut(QKeySequence("Ctrl+Alt+0"))
        photoelectric_tb_act.triggered.connect(self._on_toggle_photoelectric_effect)
        nav_tb.addAction(photoelectric_tb_act)

        punnett_tb_act = QAction("🧬 Di truyền Punnett", self)
        punnett_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền Mendel, Khung Punnett & Nhóm máu ABO (Ctrl+Shift+G)")
        punnett_tb_act.setShortcut(QKeySequence("Ctrl+Shift+G"))
        punnett_tb_act.triggered.connect(self._on_toggle_punnett_genetics)
        nav_tb.addAction(punnett_tb_act)

        kinetics_tb_act = QAction("🧪 Động Hóa học", self)
        kinetics_tb_act.setToolTip("Bật/Tắt mô phỏng Động hóa học, Năng lượng hoạt hóa & Xúc tác Arrhenius (Ctrl+Shift+K)")
        kinetics_tb_act.setShortcut(QKeySequence("Ctrl+Shift+K"))
        kinetics_tb_act.triggered.connect(self._on_toggle_reaction_kinetics)
        nav_tb.addAction(kinetics_tb_act)

        calculus_tb_act = QAction("📐 Tích phân Riemann", self)
        calculus_tb_act.setToolTip("Bật/Tắt mô phỏng Tích phân Riemann, Cầu phương Simpson & Chuỗi Taylor (Ctrl+Shift+R)")
        calculus_tb_act.setShortcut(QKeySequence("Ctrl+Shift+R"))
        calculus_tb_act.triggered.connect(self._on_toggle_calculus_taylor)
        nav_tb.addAction(calculus_tb_act)

        michelson_tb_act = QAction("🔬 Giao thoa Michelson", self)
        michelson_tb_act.setToolTip("Bật/Tắt mô phỏng Giao thoa kế Michelson & Sóng Hấp dẫn LIGO (Ctrl+Shift+M)")
        michelson_tb_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
        michelson_tb_act.triggered.connect(self._on_toggle_michelson_interferometer)
        nav_tb.addAction(michelson_tb_act)

        huffman_tb_act = QAction("🌲 Mã hóa Huffman", self)
        huffman_tb_act.setToolTip("Bật/Tắt mô phỏng Cây tiền tố tối ưu Huffman & Entropy Shannon (Ctrl+Shift+H)")
        huffman_tb_act.setShortcut(QKeySequence("Ctrl+Shift+H"))
        huffman_tb_act.triggered.connect(self._on_toggle_huffman_coding)
        nav_tb.addAction(huffman_tb_act)

        action_potential_tb_act = QAction("⚡ Điện thế Hoạt động", self)
        action_potential_tb_act.setToolTip("Bật/Tắt mô phỏng Điện thế Hoạt động & Mô hình Hodgkin-Huxley (Ctrl+Shift+P)")
        action_potential_tb_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
        action_potential_tb_act.triggered.connect(self._on_toggle_action_potential)
        nav_tb.addAction(action_potential_tb_act)

        cyclotron_tb_act = QAction("⚛️ Gia tốc Cyclotron", self)
        cyclotron_tb_act.setToolTip("Bật/Tắt mô phỏng Máy gia tốc hạt Cyclotron & Synchrocyclotron (Ctrl+Alt+C)")
        cyclotron_tb_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        cyclotron_tb_act.triggered.connect(self._on_toggle_cyclotron)
        nav_tb.addAction(cyclotron_tb_act)

        aho_corasick_tb_act = QAction("🔤 Otomat Aho-Corasick", self)
        aho_corasick_tb_act.setToolTip("Bật/Tắt mô phỏng Máy tự động tìm kiếm chuỗi Aho-Corasick O(n+m+z) (Ctrl+Alt+A)")
        aho_corasick_tb_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        aho_corasick_tb_act.triggered.connect(self._on_toggle_aho_corasick)
        nav_tb.addAction(aho_corasick_tb_act)

        phase_diagram_tb_act = QAction("🌡️ Giản đồ Pha P-T", self)
        phase_diagram_tb_act.setToolTip("Bật/Tắt mô phỏng Giản đồ pha nhiệt động P-T, điểm tam giác & tới hạn (Ctrl+Alt+P)")
        phase_diagram_tb_act.setShortcut(QKeySequence("Ctrl+Alt+P"))
        phase_diagram_tb_act.triggered.connect(self._on_toggle_phase_diagram)
        nav_tb.addAction(phase_diagram_tb_act)

        em_wave_tb_act = QAction("📡 Sóng Điện từ", self)
        em_wave_tb_act.setToolTip("Bật/Tắt mô phỏng Sóng điện từ Maxwell, phân cực & vectơ Poynting (Ctrl+Alt+E)")
        em_wave_tb_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        em_wave_tb_act.triggered.connect(self._on_toggle_em_wave)
        nav_tb.addAction(em_wave_tb_act)

        hardy_weinberg_tb_act = QAction("🧬 Hardy-Weinberg", self)
        hardy_weinberg_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền học quần thể Hardy-Weinberg & phiêu bạt gen (Ctrl+Alt+H)")
        hardy_weinberg_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_tb_act.triggered.connect(self._on_toggle_hardy_weinberg)
        nav_tb.addAction(hardy_weinberg_tb_act)

        fft_tb_act = QAction("📊 Phân tích Phổ FFT", self)
        fft_tb_act.setToolTip("Bật/Tắt mô phỏng Biến đổi Fourier Nhanh Cooley-Tukey FFT & Phân tích Phổ tần số (Ctrl+Alt+F)")
        fft_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_tb_act.triggered.connect(self._on_toggle_fft)
        nav_tb.addAction(fft_tb_act)

        fabry_perot_tb_act = QAction("🔬 Fabry-Pérot", self)
        fabry_perot_tb_act.setToolTip("Bật/Tắt mô phỏng Giao thoa kế Fabry-Pérot & Hốc quang học Finesse cao (Ctrl+Alt+O)")
        fabry_perot_tb_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        fabry_perot_tb_act.triggered.connect(self._on_toggle_fabry_perot)
        nav_tb.addAction(fabry_perot_tb_act)

        hplc_tb_act = QAction("🧪 Sắc ký HPLC", self)
        hplc_tb_act.setToolTip("Bật/Tắt mô phỏng Sắc ký lỏng HPLC & Hiệu suất cột Van Deemter (Ctrl+Alt+L)")
        hplc_tb_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        hplc_tb_act.triggered.connect(self._on_toggle_hplc)
        nav_tb.addAction(hplc_tb_act)

        photoelectric_tb_act = QAction("⚡ Quang điện", self)
        photoelectric_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu ứng quang điện ngoài Einstein & Công thoát kim loại (Ctrl+Alt+Q)")
        photoelectric_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_tb_act.triggered.connect(self._on_toggle_photoelectric)
        nav_tb.addAction(photoelectric_tb_act)

        pcr_tb_act = QAction("🧬 PCR & Điện di", self)
        pcr_tb_act.setToolTip("Bật/Tắt mô phỏng Phản ứng chuỗi Polymerase PCR & Điện di gel agarose (Ctrl+Alt+R)")
        pcr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_tb_act.triggered.connect(self._on_toggle_pcr)
        nav_tb.addAction(pcr_tb_act)

        thin_lens_tb_act = QAction("🔍 Thấu kính Mỏng", self)
        thin_lens_tb_act.setToolTip("Bật/Tắt mô phỏng Hệ thấu kính mỏng, tạo ảnh thật/ảo & ma trận ABCD (Ctrl+Alt+T)")
        thin_lens_tb_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        thin_lens_tb_act.triggered.connect(self._on_toggle_thin_lens)
        nav_tb.addAction(thin_lens_tb_act)

        bragg_tb_act = QAction("💎 Nhiễu xạ Bragg", self)
        bragg_tb_act.setToolTip("Bật/Tắt mô phỏng Nhiễu xạ Tia X Bragg, Phổ XRD & Làm rộng vạch Scherrer (Ctrl+Alt+Shift+B)")
        bragg_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+B"))
        bragg_tb_act.triggered.connect(self._on_toggle_bragg_diffraction)
        nav_tb.addAction(bragg_tb_act)

        buffer_tb_act = QAction("🧪 Hệ Đệm Axit-Bazơ", self)
        buffer_tb_act.setToolTip("Bật/Tắt mô phỏng Hệ đệm axit-bazơ, Phương trình Henderson-Hasselbalch & Dung lượng Van Slyke (Ctrl+Alt+Shift+H)")
        buffer_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+H"))
        buffer_tb_act.triggered.connect(self._on_toggle_buffer_system)
        nav_tb.addAction(buffer_tb_act)

        sg_tb_act = QAction("⚛️ Stern-Gerlach", self)
        sg_tb_act.setToolTip("Bật/Tắt mô phỏng Thí nghiệm Stern-Gerlach, Lượng tử hóa không gian & Spin 1/2 (Ctrl+Alt+Shift+S)")
        sg_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+S"))
        sg_tb_act.triggered.connect(self._on_toggle_stern_gerlach)
        nav_tb.addAction(sg_tb_act)

        enzyme_tb_act = QAction("🧪 Động học Enzyme", self)
        enzyme_tb_act.setToolTip("Bật/Tắt mô phỏng Động học Enzyme Michaelis-Menten & Đồ thị Lineweaver-Burk (Ctrl+Alt+Shift+E)")
        enzyme_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+E"))
        enzyme_tb_act.triggered.connect(self._on_toggle_enzyme_kinetics)
        nav_tb.addAction(enzyme_tb_act)

        poiseuille_tb_act = QAction("🌊 Dòng Poiseuille", self)
        poiseuille_tb_act.setToolTip("Bật/Tắt mô phỏng Dòng Chảy Nhớt Hagen-Poiseuille & Số Reynolds (Ctrl+Alt+Shift+P)")
        poiseuille_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+P"))
        poiseuille_tb_act.triggered.connect(self._on_toggle_poiseuille_flow)
        nav_tb.addAction(poiseuille_tb_act)

        cosmic_tb_act = QAction("🌌 Vũ Trụ Giãn Nở", self)
        cosmic_tb_act.setToolTip("Bật/Tắt mô phỏng Định Luật Hubble & Vũ Trụ Giãn Nở (Ctrl+Alt+Shift+U)")
        cosmic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+U"))
        cosmic_tb_act.triggered.connect(self._on_toggle_cosmic_expansion)
        nav_tb.addAction(cosmic_tb_act)

        seismic_tb_act = QAction("🌍 Sóng Địa Chấn", self)
        seismic_tb_act.setToolTip("Bật/Tắt mô phỏng Sóng Địa Chấn & Định Vị Tâm Chấn (Ctrl+Alt+Shift+Q)")
        seismic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+Q"))
        seismic_tb_act.triggered.connect(self._on_toggle_seismic_waves)
        nav_tb.addAction(seismic_tb_act)

        spectrometer_tb_act = QAction("⚡ Máy Khối Phổ", self)
        spectrometer_tb_act.setToolTip("Bật/Tắt mô phỏng Máy Khối Phổ & Bộ Lọc Vận Tốc Wien (Ctrl+Alt+Shift+M)")
        spectrometer_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+M"))
        spectrometer_tb_act.triggered.connect(self._on_toggle_mass_spectrometer)
        nav_tb.addAction(spectrometer_tb_act)

        spectrophotometry_tb_act = QAction("🧪 Quang Phổ Beer-Lambert", self)
        spectrophotometry_tb_act.setToolTip("Bật/Tắt mô phỏng Quang Phổ Hấp Thụ & Định Luật Beer-Lambert (Ctrl+Alt+Shift+L)")
        spectrophotometry_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+L"))
        spectrophotometry_tb_act.triggered.connect(self._on_toggle_spectrophotometry)
        nav_tb.addAction(spectrophotometry_tb_act)

        nuclear_binding_tb_act = QAction("⚛ Liên Kết Hạt Nhân", self)
        nuclear_binding_tb_act.setToolTip("Bật/Tắt mô phỏng đường cong năng lượng liên kết hạt nhân & công thức SEMF (Ctrl+Alt+Shift+B)")
        nuclear_binding_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+B"))
        nuclear_binding_tb_act.triggered.connect(self._on_toggle_nuclear_binding_energy)
        nav_tb.addAction(nuclear_binding_tb_act)

        compton_tb_act = QAction("⚛ Tán Xạ Compton", self)
        compton_tb_act.setToolTip("Bật/Tắt mô phỏng tán xạ Compton & va chạm relativistic photon-electron (Ctrl+Alt+Shift+Q)")
        compton_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+Q"))
        compton_tb_act.triggered.connect(self._on_toggle_compton_scattering)
        nav_tb.addAction(compton_tb_act)

        lensing_tb_act = QAction("🌌 Thấu Kính Hấp Dẫn", self)
        lensing_tb_act.setToolTip("Bật/Tắt mô phỏng Thấu Kính Hấp Dẫn & Nhẫn Einstein (Ctrl+Alt+Shift+G)")
        lensing_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+G"))
        lensing_tb_act.triggered.connect(self._on_toggle_gravitational_lensing)
        nav_tb.addAction(lensing_tb_act)

        equilibrium_tb_act = QAction("⚗ Cân Bằng Hóa Học", self)
        equilibrium_tb_act.setToolTip("Bật/Tắt mô phỏng Cân Bằng Hóa Học & Nguyên Lý Le Chatelier (Ctrl+Alt+Shift+E)")
        equilibrium_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+E"))
        equilibrium_tb_act.triggered.connect(self._on_toggle_chemical_equilibrium)
        nav_tb.addAction(equilibrium_tb_act)

        hodgkin_tb_act = QAction("🧠 Nơ-ron Hodgkin-Huxley", self)
        hodgkin_tb_act.setToolTip("Bật/Tắt mô phỏng Điện Sinh Lý Nơ-ron Hodgkin-Huxley (Ctrl+Alt+Shift+H)")
        hodgkin_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+H"))
        hodgkin_tb_act.triggered.connect(self._on_toggle_hodgkin_huxley)
        nav_tb.addAction(hodgkin_tb_act)

        mst_tb_act = QAction("🌲 Cây Khung MST", self)
        mst_tb_act.setToolTip("Bật/Tắt mô phỏng Cây Khung Nhỏ Nhất Kruskal & Prim (Ctrl+Alt+Shift+M)")
        mst_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+M"))
        mst_tb_act.triggered.connect(self._on_toggle_mst_visualizer)
        nav_tb.addAction(mst_tb_act)

        hall_tb_act = QAction("⚡ Hiệu Ứng Hall", self)
        hall_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu Ứng Hall & Vật Lý Bán Dẫn (Ctrl+Alt+Shift+F)")
        hall_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+F"))
        hall_tb_act.triggered.connect(self._on_toggle_hall_effect)
        nav_tb.addAction(hall_tb_act)

        rad_tb_act = QAction("☢ Che Chắn Phóng Xạ", self)
        rad_tb_act.setToolTip("Bật/Tắt mô phỏng Độ Xuyên Sâu Phóng Xạ & Tấm Chắn HVL (Ctrl+Alt+Shift+R)")
        rad_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+R"))
        rad_tb_act.triggered.connect(self._on_toggle_radiation_shielding)
        nav_tb.addAction(rad_tb_act)

        maxwell_tb_act = QAction("🔥 Phân Bố Maxwell", self)
        maxwell_tb_act.setToolTip("Bật/Tắt mô phỏng Phân Bố Vận Tốc Phân Tử Maxwell-Boltzmann (Ctrl+Alt+Shift+X)")
        maxwell_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+X"))
        maxwell_tb_act.triggered.connect(self._on_toggle_maxwell_boltzmann)
        nav_tb.addAction(maxwell_tb_act)

        mo_tb_act = QAction("⚛ Obitan Phân Tử", self)
        mo_tb_act.setToolTip("Bật/Tắt mô phỏng Obitan Phân Tử & Thuyết MO-LCAO (Ctrl+Alt+Shift+O)")
        mo_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+O"))
        mo_tb_act.triggered.connect(self._on_toggle_molecular_orbital)
        nav_tb.addAction(mo_tb_act)

        rlc_tb_act = QAction("⚡ Cộng Hưởng RLC", self)
        rlc_tb_act.setToolTip("Bật/Tắt mô phỏng Cộng Hưởng Điện RLC & Giản Đồ Phasor (Ctrl+Alt+Shift+C)")
        rlc_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+C"))
        rlc_tb_act.triggered.connect(self._on_toggle_rlc_resonance)
        nav_tb.addAction(rlc_tb_act)

        stress_tb_act = QAction("⚙ Đường Cong Ứng Suất", self)
        stress_tb_act.setToolTip("Bật/Tắt mô phỏng Đường Cong Ứng Suất - Biến Dạng & Kéo Thử Cơ Học (Ctrl+Alt+Shift+T)")
        stress_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+T"))
        stress_tb_act.triggered.connect(self._on_toggle_stress_strain)
        nav_tb.addAction(stress_tb_act)

        kepler_tb_act = QAction("🪐 Quỹ Đạo Kepler", self)
        kepler_tb_act.setToolTip("Bật/Tắt mô phỏng Quỹ Đạo Hấp Dẫn & 3 Định Luật Kepler (Ctrl+Alt+Shift+K)")
        kepler_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+K"))
        kepler_tb_act.triggered.connect(self._on_toggle_kepler_orbits)
        nav_tb.addAction(kepler_tb_act)

        logic_tb_act = QAction("🔌 Cổng Logic & Adder", self)
        logic_tb_act.setToolTip("Bật/Tắt mô phỏng Điện Tử Số, Cổng Logic & Mạch Cộng Nhị Phân (Ctrl+Alt+Shift+G)")
        logic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+G"))
        logic_tb_act.triggered.connect(self._on_toggle_logic_circuits)
        nav_tb.addAction(logic_tb_act)

        desktop_tb_act = QAction("🖥️ Vẽ Desktop", self)
        desktop_tb_act.setToolTip("Chuyển đổi Chế độ vẽ đè màn hình Desktop trong suốt (Phím F11)")
        desktop_tb_act.setShortcut(QKeySequence("F11"))
        desktop_tb_act.triggered.connect(self._on_toggle_desktop_overlay)
        nav_tb.addAction(desktop_tb_act)

        guide_tb_act = QAction("💡 Hướng dẫn cử chỉ", self)
        guide_tb_act.setToolTip("Mở thẻ hướng dẫn thao tác cử chỉ tay nhanh (Phím F1)")
        guide_tb_act.setShortcut(QKeySequence("F1"))
        guide_tb_act.triggered.connect(self._on_toggle_gesture_guide)
        nav_tb.addAction(guide_tb_act)

        clear_tb_act = QAction(_get_icon("clear"), "Xóa nét", self)
        clear_tb_act.setToolTip("Xóa toàn bộ nét vẽ trên trang hiện tại")
        clear_tb_act.triggered.connect(self._on_clear_all)
        nav_tb.addAction(clear_tb_act)

        # Spacer to push profile badge and logout to right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        nav_tb.addWidget(spacer)

        self._cam_diag_btn = QPushButton("📹 Camera: Khởi tạo…")
        self._cam_diag_btn.setToolTip("Mở bảng Chẩn đoán Camera & Đo lường chất lượng hình ảnh")
        self._cam_diag_btn.setStyleSheet("""
            QPushButton {
                background: #f1f3f4;
                color: #3c4043;
                border: 1px solid #dadce0;
                border-radius: 12px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
                margin-right: 4px;
            }
            QPushButton:hover {
                background: #e8eaed;
            }
        """)
        self._cam_diag_btn.clicked.connect(self._on_open_camera_diagnostics)
        nav_tb.addWidget(self._cam_diag_btn)

        self._ai_workbench_btn = QPushButton("✨ AI Workbench")
        self._ai_workbench_btn.setToolTip("Mở Trạm soạn thảo & Trợ lý Giảng dạy AI (Soạn đề, Giáo án, Tóm tắt)")
        self._ai_workbench_btn.setStyleSheet("""
            QPushButton {
                background: #e8f0fe;
                color: #1967d2;
                border: 1px solid #c2e7ff;
                border-radius: 12px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
                margin-right: 6px;
            }
            QPushButton:hover {
                background: #d2e3fc;
            }
        """)
        self._ai_workbench_btn.clicked.connect(self._on_open_ai_workbench)
        nav_tb.addWidget(self._ai_workbench_btn)

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

        menu_act = QAction(_get_icon("menu"), "Menu không chạm (M)", self)
        menu_act.setShortcut(Qt.Key.Key_M)
        menu_act.setToolTip("Mở / Đóng Menu điều khiển không chạm trong không gian (M)")
        menu_act.triggered.connect(self._toggle_floating_menu)
        canvas_tb.addAction(menu_act)

        canvas_tb.addSeparator()

        snap_act = QAction(_get_icon("shape"), "Nắn hình học (G)", self)
        snap_act.setCheckable(True)
        snap_act.setChecked(self._snapping_on)
        snap_act.setShortcut(Qt.Key.Key_G)
        snap_act.triggered.connect(self._toggle_shape_snapping)
        canvas_tb.addAction(snap_act)

        canvas_tb.addSeparator()

        self._draw_mode_btn = QPushButton()
        self._draw_mode_btn.setToolTip("Chuyển đổi Chế độ vẽ: Chỉ tay (Air Pen) vs Véo ngón (Pinch-to-draw)")
        self._draw_mode_btn.setStyleSheet("""
            QPushButton {
                background: #f8f9fa;
                color: #202124;
                border: 1px solid #dadce0;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e8f0fe;
                border-color: #1a73e8;
            }
        """)
        self._draw_mode_btn.clicked.connect(self._toggle_draw_mode)
        canvas_tb.addWidget(self._draw_mode_btn)
        self._apply_draw_mode()

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

        # 3. STEM Subject Tools Toolbar
        stem_tb = QToolBar("Bộ công cụ Chuyên đề STEM")
        stem_tb.setMovable(False)
        stem_tb.setIconSize(QSize(20, 20))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, stem_tb)

        stem_tb.addWidget(QLabel(" 🎓 Chuyên đề STEM: "))

        self._stem_subject_combo = QComboBox()
        self._stem_subject_combo.addItem("📐 Toán học", "math")
        self._stem_subject_combo.addItem("⚡ Vật lý", "physics")
        self._stem_subject_combo.addItem("🧪 Hóa học", "chemistry")
        self._stem_subject_combo.addItem("🧬 Sinh học", "biology")
        self._stem_subject_combo.addItem("💻 Tin học", "informatics")
        self._stem_subject_combo.currentIndexChanged.connect(self._on_stem_subject_changed)
        stem_tb.addWidget(self._stem_subject_combo)

        stem_tb.addSeparator()

        # Math Actions
        self._stem_oxy_act = QAction("📐 Hệ Oxy", self)
        self._stem_oxy_act.setToolTip("Chèn hệ trục tọa độ vuông góc Oxy chuẩn vào bài giảng")
        self._stem_oxy_act.triggered.connect(self._on_insert_stem_oxy)
        stem_tb.addAction(self._stem_oxy_act)

        self._stem_func_act = QAction("📈 Đồ thị f(x)", self)
        self._stem_func_act.setToolTip("Vẽ đồ thị hàm số (x^2, sin, cos, exp...)")
        self._stem_func_act.triggered.connect(self._on_insert_stem_function)
        stem_tb.addAction(self._stem_func_act)

        self._stem_triangle_act = QAction("📐 Tam giác ABC", self)
        self._stem_triangle_act.setToolTip("Chèn tam giác hình học (vuông, cân, đều, thường) kèm nhãn đỉnh")
        self._stem_triangle_act.triggered.connect(self._on_insert_stem_triangle)
        stem_tb.addAction(self._stem_triangle_act)

        self._stem_circle_rad_act = QAction("⭕ Đường tròn & R", self)
        self._stem_circle_rad_act.setToolTip("Chèn đường tròn kèm tâm O, bán kính nét đứt và nhãn số đo")
        self._stem_circle_rad_act.triggered.connect(self._on_insert_stem_circle_radius)
        stem_tb.addAction(self._stem_circle_rad_act)

        self._stem_angle_act = QAction("∠ Cung đo Góc", self)
        self._stem_angle_act.setToolTip("Chèn cung góc đo và nhãn góc α giữa 2 tia")
        self._stem_angle_act.triggered.connect(self._on_insert_stem_angle)
        stem_tb.addAction(self._stem_angle_act)

        # Physics Actions
        self._stem_force_act = QAction("➡️ Vectơ Lực F⃗", self)
        self._stem_force_act.setToolTip("Chèn vectơ lực có phương, chiều và độ lớn")
        self._stem_force_act.triggered.connect(self._on_insert_stem_force)
        stem_tb.addAction(self._stem_force_act)

        self._stem_decomp_act = QAction("🔲 Phân tích Lực", self)
        self._stem_decomp_act.setToolTip("Phân tích vectơ lực thành 2 thành phần Fx và Fy vuông góc")
        self._stem_decomp_act.triggered.connect(self._on_insert_stem_force_decomp)
        stem_tb.addAction(self._stem_decomp_act)

        self._stem_circuit_act = QAction("⚡ Mạch điện", self)
        self._stem_circuit_act.setToolTip("Chèn linh kiện mạch điện (Điện trở, Tụ điện, Nguồn, Công tắc)")
        self._stem_circuit_act.triggered.connect(self._on_insert_stem_circuit)
        stem_tb.addAction(self._stem_circuit_act)

        self._stem_optics_act = QAction("🔍 Quang hình / Thấu kính", self)
        self._stem_optics_act.setToolTip("Chèn sơ đồ tia sáng qua thấu kính mỏng hội tụ / phân kỳ")
        self._stem_optics_act.triggered.connect(self._on_insert_stem_optics)
        stem_tb.addAction(self._stem_optics_act)

        # Chemistry Actions
        self._stem_chem_act = QAction("🧪 Công thức Hóa", self)
        self._stem_chem_act.setToolTip("Định dạng và chèn công thức hóa học tự động (H2SO4 -> H₂SO₄)")
        self._stem_chem_act.triggered.connect(self._on_insert_stem_chem_formula)
        stem_tb.addAction(self._stem_chem_act)

        self._stem_arrow_act = QAction("⇄ Mũi tên Phản ứng", self)
        self._stem_arrow_act.setToolTip("Chèn mũi tên phản ứng 1 chiều hoặc thuận nghịch có kèm điều kiện nhiệt độ/xúc tác")
        self._stem_arrow_act.triggered.connect(self._on_insert_stem_reaction_arrow)
        stem_tb.addAction(self._stem_arrow_act)

        self._stem_benzene_act = QAction("⬡ Vòng Benzen", self)
        self._stem_benzene_act.setToolTip("Chèn vòng nhân thơm Benzen chuẩn")
        self._stem_benzene_act.triggered.connect(self._on_insert_stem_benzene)
        stem_tb.addAction(self._stem_benzene_act)

        self._stem_bond_act = QAction("⚗️ Liên kết Hóa học", self)
        self._stem_bond_act.setToolTip("Chèn liên kết cộng hóa trị (đơn, đôi, ba)")
        self._stem_bond_act.triggered.connect(self._on_insert_stem_organic_bond)
        stem_tb.addAction(self._stem_bond_act)

        self._stem_molecule_act = QAction("⚛️ Hình học Phân tử", self)
        self._stem_molecule_act.setToolTip("Chèn mô hình phân tử 3D quả cầu & que nối (CH₄, H₂O, CO₂, BF₃, NH₃, SF₆)")
        self._stem_molecule_act.triggered.connect(self._on_insert_stem_molecule)
        stem_tb.addAction(self._stem_molecule_act)

        # Biology Actions
        self._stem_bio_act = QAction("📍 Ghim Chú thích", self)
        self._stem_bio_act.setToolTip("Chèn kim chỉ dẫn chú thích bộ phận giải phẫu hoặc tế bào")
        self._stem_bio_act.triggered.connect(self._on_insert_stem_biology_callout)
        stem_tb.addAction(self._stem_bio_act)

        self._stem_bio_dna_act = QAction("🧬 Xoắn kép DNA", self)
        self._stem_bio_dna_act.setToolTip("Chèn chuỗi xoắn kép DNA với 2 mạch xoắn và các cặp bazơ nitơ bổ sung")
        self._stem_bio_dna_act.triggered.connect(self._on_insert_stem_dna)
        stem_tb.addAction(self._stem_bio_dna_act)

        self._stem_bio_punnett_act = QAction("▦ Khung Punnett", self)
        self._stem_bio_punnett_act.setToolTip("Chèn khung di truyền Menđen Punnett 2x2 với các giao tử và kiểu gen con")
        self._stem_bio_punnett_act.triggered.connect(self._on_insert_stem_punnett)
        stem_tb.addAction(self._stem_bio_punnett_act)

        # Informatics Actions
        self._stem_flow_act = QAction("🔄 Khối Thuật toán", self)
        self._stem_flow_act.setToolTip("Chèn khối sơ đồ thuật toán (Thực thi, Điều kiện, Vào/Ra, Bắt đầu/Kết thúc)")
        self._stem_flow_act.triggered.connect(self._on_insert_stem_flowchart)
        stem_tb.addAction(self._stem_flow_act)
        self._stem_logic_gate_act = QAction("🔲 Cổng Logic", self)
        self._stem_logic_gate_act.setToolTip("Chèn cổng logic số (AND, OR, NOT, NAND, NOR, XOR)")
        self._stem_logic_gate_act.triggered.connect(self._on_insert_stem_logic_gate)
        stem_tb.addAction(self._stem_logic_gate_act)

        self._stem_flow_arrow_act = QAction("➡️ Luồng Rẽ nhánh", self)
        self._stem_flow_arrow_act.setToolTip("Chèn mũi tên nối luồng thuật toán thẳng hoặc gấp khúc vuông góc kèm điều kiện")
        self._stem_flow_arrow_act.triggered.connect(self._on_insert_stem_flow_arrow)
        stem_tb.addAction(self._stem_flow_arrow_act)

        self._stem_code_act = QAction("💻 Khối Code", self)
        self._stem_code_act.setToolTip("Chèn khung hiển thị mã nguồn lập trình monospaced")
        self._stem_code_act.triggered.connect(self._on_insert_stem_code_block)
        stem_tb.addAction(self._stem_code_act)

        self._on_stem_subject_changed(0)

    def _add_stem_item_to_current_page(self, item_or_items: Any) -> None:
        """Add STEM items to the active canvas and track them for per-page lifecycle and undo."""
        if isinstance(item_or_items, list):
            items = [itm for itm in item_or_items if itm is not None]
        elif item_or_items is not None:
            items = [item_or_items]
        else:
            items = []

        if not items:
            return

        stem_list = self._page_stem_items.setdefault(self._current_page_index, [])
        for itm in items:
            if itm.scene() is None:
                self._scene.addItem(itm)
            itm.setVisible(True)
            stem_list.append(itm)

        self._page_undo_stack.setdefault(self._current_page_index, []).append(("stem", items))
        self._page_redo_stack.setdefault(self._current_page_index, []).clear()

    # ── STEM Subject Tools Slots ──────────────────────────────────────────────

    @Slot(int)
    def _on_stem_subject_changed(self, index: int) -> None:
        """Toggle visibility of STEM tool actions based on selected subject workspace."""
        is_math = (index == 0)
        is_phys = (index == 1)
        is_chem = (index == 2)
        is_bio  = (index == 3)
        is_info = (index == 4)

        self._stem_oxy_act.setVisible(is_math)
        self._stem_func_act.setVisible(is_math)
        self._stem_triangle_act.setVisible(is_math)
        self._stem_circle_rad_act.setVisible(is_math)
        self._stem_angle_act.setVisible(is_math)

        self._stem_force_act.setVisible(is_phys)
        self._stem_decomp_act.setVisible(is_phys)
        self._stem_circuit_act.setVisible(is_phys)
        self._stem_optics_act.setVisible(is_phys)

        self._stem_chem_act.setVisible(is_chem)
        self._stem_arrow_act.setVisible(is_chem)
        self._stem_benzene_act.setVisible(is_chem)
        self._stem_bond_act.setVisible(is_chem)

        self._stem_bio_act.setVisible(is_bio)
        self._stem_bio_dna_act.setVisible(is_bio)
        self._stem_bio_punnett_act.setVisible(is_bio)

        self._stem_flow_act.setVisible(is_info)
        self._stem_logic_gate_act.setVisible(is_info)
        self._stem_flow_arrow_act.setVisible(is_info)
        self._stem_code_act.setVisible(is_info)

    @Slot()
    def _on_insert_stem_oxy(self) -> None:
        """Insert Cartesian Oxy coordinate system with grid and axis labels."""
        dlg = StemToolDialog("Thiết lập Hệ trục Oxy", "math_oxy", self)
        if dlg.exec():
            try:
                x_r = float(dlg.inp_x_range.text().strip() or "8")
                y_r = float(dlg.inp_y_range.text().strip() or "5")
            except ValueError:
                x_r, y_r = 8.0, 5.0
            items = MathTools.create_cartesian_axes(
                origin_x=960.0, origin_y=540.0,
                x_min=-x_r, x_max=x_r,
                y_min=-y_r, y_max=y_r,
                grid_step_px=60.0,
                show_grid=True,
                pen_color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage("Đã chèn hệ trục tọa độ Oxy chuẩn", 3000)

    @Slot()
    def _on_insert_stem_function(self) -> None:
        """Insert mathematical function graph."""
        dlg = StemToolDialog("Vẽ đồ thị hàm số f(x)", "math_func", self)
        if dlg.exec():
            expr = dlg.inp_expr.text().strip() or "x**2 - 2"
            item = MathTools.create_function_graph(
                expr=expr,
                origin_x=960.0, origin_y=540.0,
                x_min=-6.0, x_max=6.0,
                grid_step_px=60.0,
                line_color=self._ink_color,
            )
            if item is not None:
                self._add_stem_item_to_current_page(item)
                self._status.showMessage(f"Đã vẽ đồ thị hàm số y = {expr}", 3000)
            else:
                QMessageBox.warning(self, "Lỗi biểu thức", f"Không thể vẽ hàm số: {expr}")

    @Slot()
    def _on_insert_stem_force(self) -> None:
        """Insert physics force vector with magnitude, angle and label."""
        dlg = StemToolDialog("Vectơ Lực F⃗", "physics_force", self)
        if dlg.exec():
            label = dlg.inp_label.text().strip() or "F⃗"
            try:
                mag = float(dlg.inp_mag.text().strip() or "120")
                angle = float(dlg.inp_angle.text().strip() or "45")
            except ValueError:
                mag, angle = 120.0, 45.0
            rad = math.radians(angle)
            p0 = (960.0, 540.0)
            p1 = (960.0 + mag * math.cos(rad), 540.0 - mag * math.sin(rad))
            items = PhysicsTools.create_force_vector(p0, p1, label=label, color=self._ink_color)
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn vectơ lực {label}", 3000)

    @Slot()
    def _on_insert_stem_force_decomp(self) -> None:
        """Insert physics vector orthogonal decomposition (Fx, Fy)."""
        origin = (860.0, 600.0)
        vec_end = (1080.0, 420.0)
        items = PhysicsTools.create_vector_decomposition(
            origin=origin,
            vector_end=vec_end,
            label="F⃗",
            color_main=self._ink_color,
            color_comp=QColor("#1a73e8"),
        )
        self._add_stem_item_to_current_page(items)
        self._status.showMessage("Đã chèn phân tích vectơ lực Fx, Fy", 3000)

    @Slot()
    def _on_insert_stem_circuit(self) -> None:
        """Insert standard electrical schematic component with interactive dialog."""
        dlg = StemToolDialog("Chèn linh kiện mạch điện", "physics_circuit", self)
        if dlg.exec():
            raw_comp = dlg.combo_component.currentText().lower()
            label = dlg.inp_label.text().strip()
            if "resistor" in raw_comp or "điện trở" in raw_comp:
                comp = "resistor"
            elif "capacitor" in raw_comp or "tụ điện" in raw_comp:
                comp = "capacitor"
            elif "battery" in raw_comp or "nguồn" in raw_comp:
                comp = "battery"
            elif "inductor" in raw_comp or "cuộn cảm" in raw_comp:
                comp = "inductor"
            elif "ground" in raw_comp or "nối đất" in raw_comp:
                comp = "ground"
            elif "switch" in raw_comp or "công tắc" in raw_comp:
                comp = "switch"
            else:
                comp = "resistor"

            items = PhysicsTools.create_circuit_component(
                component=comp,
                start=(820.0, 540.0),
                end=(1100.0, 540.0),
                label=label,
                color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn linh kiện {dlg.combo_component.currentText()}", 3000)

    @Slot()
    def _on_insert_stem_optics(self) -> None:
        """Insert geometric optics ray tracing diagram."""
        dlg = StemToolDialog("Sơ đồ Quang hình & Thấu kính", "physics_optics", self)
        if dlg.exec():
            lens_type = dlg.combo_lens_type.currentText()
            try:
                focal = float(dlg.inp_focal.text().strip() or "120")
            except ValueError:
                focal = 120.0
            try:
                obj_dist = float(dlg.inp_obj_dist.text().strip() or "200")
            except ValueError:
                obj_dist = 200.0
            try:
                obj_h = float(dlg.inp_obj_height.text().strip() or "60")
            except ValueError:
                obj_h = 60.0

            items = PhysicsTools.create_optics_ray_diagram(
                center=(960.0, 540.0),
                lens_type=lens_type,
                focal_length=focal,
                object_dist=obj_dist,
                object_height=obj_h,
                color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn sơ đồ quang học ({lens_type})", 3000)

    @Slot()
    def _on_insert_stem_chem_formula(self) -> None:
        """Insert auto-formatted chemical equation or formula."""
        dlg = StemToolDialog("Công thức & Phương trình Hóa học", "chem_formula", self)
        if dlg.exec():
            raw = dlg.inp_chem.text().strip() or "H2SO4 + 2NaOH -> Na2SO4 + 2H2O"
            formatted = format_chemical_formula(raw)
            txt = QGraphicsTextItem(formatted)
            txt.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
            txt.setDefaultTextColor(self._ink_color)
            txt.setPos(760.0, 500.0)
            txt.setZValue(12.0)
            self._add_stem_item_to_current_page(txt)
            self._status.showMessage(f"Đã chèn công thức hóa học: {formatted}", 3000)

    @Slot()
    def _on_insert_stem_reaction_arrow(self) -> None:
        """Insert chemical reaction reversible/irreversible arrow with conditions."""
        items = ChemistryTools.create_reaction_arrow(
            start=(820.0, 540.0),
            end=(1100.0, 540.0),
            reversible=True,
            condition_top="t°, V₂O₅",
            condition_bottom="p",
            color=self._ink_color,
        )
        self._add_stem_item_to_current_page(items)
        self._status.showMessage("Đã chèn mũi tên phản ứng thuận nghịch ⇄", 3000)

    @Slot()
    def _on_insert_stem_benzene(self) -> None:
        """Insert standard benzene organic ring."""
        items = ChemistryTools.create_benzene_ring(
            center=(960.0, 540.0),
            radius=60.0,
            color=self._ink_color,
        )
        self._add_stem_item_to_current_page(items)
        self._status.showMessage("Đã chèn vòng thơm Benzen C₆H₆", 3000)

    @Slot()
    def _on_insert_stem_biology_callout(self) -> None:
        """Insert anatomical pointer leader line with circular pin and callout text."""
        items = BiologyTools.create_callout_pin(
            target_pt=(900.0, 520.0),
            label_pt=(1060.0, 460.0),
            text="Nhân tế bào (Nucleus)",
            color=self._ink_color,
        )
        self._add_stem_item_to_current_page(items)
        self._status.showMessage("Đã chèn ghim chú thích Sinh học", 3000)

    @Slot()
    def _on_insert_stem_dna(self) -> None:
        """Insert DNA double helix segment."""
        dlg = StemToolDialog("Chuỗi Xoắn Kép DNA", "bio_dna", self)
        if dlg.exec():
            try:
                length = float(dlg.inp_dna_len.text().strip() or "320")
            except ValueError:
                length = 320.0
            try:
                turns = float(dlg.inp_turns.text().strip() or "2.5")
            except ValueError:
                turns = 2.5
            try:
                amplitude = float(dlg.inp_amp.text().strip() or "28")
            except ValueError:
                amplitude = 28.0

            x_start = 800.0
            y_start = 540.0
            items = BiologyTools.create_dna_double_helix(
                start=(x_start, y_start),
                end=(x_start + length, y_start),
                turns=turns,
                amplitude=amplitude,
                rungs=max(6, int(turns * 6)),
                color1=QColor("#1a73e8"),
                color2=QColor("#ea4335"),
                show_labels=True,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn chuỗi xoắn kép DNA ({turns} chu kỳ)", 3000)

    @Slot()
    def _on_insert_stem_punnett(self) -> None:
        """Insert Mendelian genetics 2x2 Punnett square table."""
        dlg = StemToolDialog("Khung Di Truyền Punnett", "bio_punnett", self)
        if dlg.exec():
            raw_male = [x.strip() for x in dlg.inp_male.text().split(",") if x.strip()]
            raw_fem = [x.strip() for x in dlg.inp_female.text().split(",") if x.strip()]
            male_g = (raw_male[0] if len(raw_male) > 0 else "A", raw_male[1] if len(raw_male) > 1 else "a")
            fem_g = (raw_fem[0] if len(raw_fem) > 0 else "A", raw_fem[1] if len(raw_fem) > 1 else "a")
            title = dlg.inp_cross_title.text().strip() or "Khung Punnett"

            items = BiologyTools.create_punnett_square(
                center=(960.0, 540.0),
                cell_size=72.0,
                male_gametes=male_g,
                female_gametes=fem_g,
                title=title,
                color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn {title}", 3000)

    @Slot()
    def _on_insert_stem_flowchart(self) -> None:
        """Insert algorithm flowchart shape."""
        dlg = StemToolDialog("Khối Sơ đồ Thuật toán", "info_flowchart", self)
        if dlg.exec():
            shape_type = dlg.combo_type.currentText()
            label = dlg.inp_text.text().strip() or "Bước xử lý"
            items = InformaticsTools.create_flowchart_shape(
                shape_type=shape_type,
                center=(960.0, 540.0),
                width=160.0,
                height=76.0,
                label=label,
                color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn khối thuật toán {shape_type.split()[0]}", 3000)

    @Slot()
    def _on_insert_stem_logic_gate(self) -> None:
        """Insert digital logic gate (AND, OR, NOT, NAND, NOR, XOR)."""
        dlg = StemToolDialog("Chèn Cổng Logic Số", "info_logic_gate", self)
        if dlg.exec():
            gate_text = dlg.combo_gate_type.currentText()
            gate_type = gate_text.split()[0]
            inputs = tuple(x.strip() for x in dlg.inp_inputs.text().split(",") if x.strip())
            output = dlg.inp_output.text().strip() or "Y"

            gate = InformaticsTools.create_interactive_logic_gate(
                gate_type=gate_type,
                center=(960.0, 540.0),
                input_labels=inputs,
                output_label=output,
            )
            gate.state_changed.connect(
                lambda gt, out: self._status.showMessage(f"Cổng logic {gt}: Ngõ ra = {out}", 2000)
            )
            self._add_stem_item_to_current_page(gate)
            self._status.showMessage(f"Đã chèn cổng logic tương tác {gate_type}", 3000)

    @Slot()
    def _on_insert_stem_flow_arrow(self) -> None:
        """Insert flowchart flow/branch connector arrow."""
        dlg = StemToolDialog("Chèn Mũi tên Rẽ nhánh Sơ đồ", "info_flow_arrow", self)
        if dlg.exec():
            is_elbow = dlg.combo_arrow_type.currentIndex() == 1
            arr_type = "elbow" if is_elbow else "straight"
            label = dlg.combo_condition.currentText().strip()

            items = InformaticsTools.create_flow_arrow(
                start=(860.0, 480.0),
                end=(1060.0, 600.0) if is_elbow else (1060.0, 480.0),
                arrow_type=arr_type,
                label=label,
                color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage("Đã chèn đường nối luồng thuật toán", 3000)

    @Slot()
    def _on_insert_stem_code_block(self) -> None:
        """Insert professional styled dark IDE code block."""
        default_code = "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1"
        items = InformaticsTools.create_code_block(
            top_left=(720.0, 420.0),
            width=520.0,
            height=260.0,
            code_text=default_code,
            title="binary_search.py",
        )
        self._add_stem_item_to_current_page(items)
        self._status.showMessage("Đã chèn khung mã nguồn Python", 3000)

    @Slot()
    def _on_insert_stem_triangle(self) -> None:
        """Insert geometric triangle with vertex labels and optional right-angle marker."""
        dlg = StemToolDialog("Chèn Tam giác Hình học", "math_triangle", self)
        if dlg.exec():
            try:
                base = float(dlg.inp_base.text().strip() or "240")
                height = float(dlg.inp_height.text().strip() or "160")
            except ValueError:
                base, height = 240.0, 160.0

            raw_labels = dlg.inp_labels.text().split(",")
            lbl_a = raw_labels[0].strip() if len(raw_labels) > 0 and raw_labels[0].strip() else "A"
            lbl_b = raw_labels[1].strip() if len(raw_labels) > 1 and raw_labels[1].strip() else "B"
            lbl_c = raw_labels[2].strip() if len(raw_labels) > 2 and raw_labels[2].strip() else "C"
            labels = (lbl_a, lbl_b, lbl_c)

            tri_type_idx = dlg.combo_tri_type.currentIndex()
            is_right = (tri_type_idx == 0)

            cx, cy = 960.0, 540.0
            if tri_type_idx == 0:  # Right triangle at A
                v0 = (cx - base / 2.0, cy + height / 2.0)  # A (right angle)
                v1 = (cx + base / 2.0, cy + height / 2.0)  # B
                v2 = (cx - base / 2.0, cy - height / 2.0)  # C
            elif tri_type_idx == 1:  # Isosceles triangle at A
                v0 = (cx, cy - height / 2.0)  # Top vertex A
                v1 = (cx - base / 2.0, cy + height / 2.0)  # B
                v2 = (cx + base / 2.0, cy + height / 2.0)  # C
            elif tri_type_idx == 2:  # Equilateral triangle
                h_eq = base * 0.866025
                v0 = (cx, cy - h_eq * 0.6)
                v1 = (cx - base / 2.0, cy + h_eq * 0.4)
                v2 = (cx + base / 2.0, cy + h_eq * 0.4)
            else:  # Scalene triangle
                v0 = (cx - 40.0, cy - height / 2.0)
                v1 = (cx - base / 2.0, cy + height / 2.0)
                v2 = (cx + base / 2.0, cy + height / 2.0)

            items = MathTools.create_geometric_triangle(
                v0=v0, v1=v1, v2=v2,
                labels=labels,
                show_right_angle=is_right,
                right_angle_vertex_idx=0,
                pen_color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn tam giác {lbl_a}{lbl_b}{lbl_c}", 3000)

    @Slot()
    def _on_insert_stem_circle_radius(self) -> None:
        """Insert circle with center dot, dashed radius line, and measurement label."""
        dlg = StemToolDialog("Chèn Đường tròn & Bán kính", "math_circle_radius", self)
        if dlg.exec():
            try:
                rad = float(dlg.inp_radius.text().strip() or "120")
            except ValueError:
                rad = 120.0
            lbl_rad = dlg.inp_rad_lbl.text().strip() or "R"
            items = MathTools.create_circle_with_radius(
                center=(960.0, 540.0),
                radius=rad,
                label=lbl_rad,
                pen_color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn đường tròn bán kính {lbl_rad}", 3000)

    @Slot()
    def _on_insert_stem_angle(self) -> None:
        """Insert angle arc with vertex, rays, and degree/label badge."""
        dlg = StemToolDialog("Chèn Cung đo Góc", "math_angle", self)
        if dlg.exec():
            try:
                ang_deg = float(dlg.inp_angle_deg.text().strip() or "60")
                ray_len = float(dlg.inp_ray_len.text().strip() or "150")
                arc_r = float(dlg.inp_arc_radius.text().strip() or "45")
            except ValueError:
                ang_deg, ray_len, arc_r = 60.0, 150.0, 45.0
            lbl = dlg.inp_angle_label.text().strip() or f"α = {int(ang_deg)}°"

            vertex = (900.0, 560.0)
            p1 = (vertex[0] + ray_len, vertex[1])
            rad_calc = math.radians(-ang_deg)
            p2 = (vertex[0] + ray_len * math.cos(rad_calc), vertex[1] + ray_len * math.sin(rad_calc))

            # Ray lines
            ray1 = QGraphicsLineItem(vertex[0], vertex[1], p1[0], p1[1])
            ray1.setPen(QPen(self._ink_color, 2.0))
            ray1.setZValue(11.0)

            ray2 = QGraphicsLineItem(vertex[0], vertex[1], p2[0], p2[1])
            ray2.setPen(QPen(self._ink_color, 2.0))
            ray2.setZValue(11.0)

            arc_items = MathTools.create_angle_arc(
                vertex=vertex, p1=p1, p2=p2,
                radius=arc_r, label=lbl, color=self._ink_color,
            )
            all_items = [ray1, ray2] + arc_items
            self._add_stem_item_to_current_page(all_items)
            self._status.showMessage(f"Đã chèn cung đo góc {lbl}", 3000)

    @Slot()
    def _on_insert_stem_molecule(self) -> None:
        """Insert 3D ball-and-stick molecular geometry model."""
        dlg = StemToolDialog("Mô hình Hình học Phân tử 3D", "chem_molecule", self)
        if dlg.exec():
            mol_idx = dlg.combo_mol_type.currentIndex()
            type_keys = ["tetrahedral", "bent", "linear", "trigonal_planar", "trigonal_pyramidal", "octahedral"]
            mol_type = type_keys[mol_idx] if 0 <= mol_idx < len(type_keys) else "tetrahedral"
            try:
                scale = float(dlg.inp_scale.text().strip() or "80")
            except ValueError:
                scale = 80.0
            title = dlg.inp_title.text().strip()

            items = ChemistryTools.create_molecular_geometry(
                center=(960.0, 520.0),
                geometry_type=mol_type,
                scale=scale,
                title=title,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn mô hình phân tử ({mol_type})", 3000)

    @Slot()
    def _on_insert_stem_organic_bond(self) -> None:
        """Insert single, double, or triple covalent bond."""
        dlg = StemToolDialog("Chèn Liên kết Hóa học", "chem_bond", self)
        if dlg.exec():
            b_idx = dlg.combo_bond_type.currentIndex()
            bond_type = "single" if b_idx == 0 else ("double" if b_idx == 1 else "triple")
            try:
                length = float(dlg.inp_bond_len.text().strip() or "80")
            except ValueError:
                length = 80.0

            ang_idx = dlg.combo_bond_angle.currentIndex()
            angle_map = {0: 0.0, 1: 60.0, 2: -60.0, 3: 90.0, 4: 30.0}
            deg = angle_map.get(ang_idx, 0.0)
            rad = math.radians(deg)

            start = (920.0, 540.0)
            end = (start[0] + length * math.cos(rad), start[1] - length * math.sin(rad))

            items = ChemistryTools.create_organic_bond(
                start=start, end=end, bond_type=bond_type, color=self._ink_color,
            )
            self._add_stem_item_to_current_page(items)
            self._status.showMessage(f"Đã chèn liên kết hóa học ({bond_type})", 3000)

    # ── Camera / Hand tracking ─────────────────────────────────────────────────

    def _connect_camera(self, camera_index: int) -> None:
        """Start the _QtCameraWorker on a background thread."""
        if self._cam_thread is not None:
            self._disconnect_camera()

        if camera_index < 0:
            logger.info("Camera index is negative (%d); camera thread not started.", camera_index)
            self._cam_widget.setText("📷 Camera offline")
            return

        worker = _QtCameraWorker(
            camera_index=camera_index,
            lighting_mode=self._camera_lighting_mode,
            multi_hand_mode=self._multi_hand_mode,
        )
        mode_enum = DrawMode.PRECISION_PINCH if self._draw_mode == "precision_pinch" else DrawMode.INDEX_UP
        worker.set_draw_mode(mode_enum)
        worker.set_dominant_hand(self._dominant_hand)
        worker.set_air_click_type(self._air_click_type)
        thread = QThread(self)
        worker.moveToThread(thread)

        # Frame display
        worker.frame_ready.connect(self._on_camera_frame, Qt.ConnectionType.QueuedConnection)
        # Gesture events
        worker.gesture_event.connect(self._on_gesture_event, Qt.ConnectionType.QueuedConnection)
        worker.multi_gesture_event.connect(self._on_multi_gesture_event, Qt.ConnectionType.QueuedConnection)
        # Camera Diagnostics
        worker.diagnostics_ready.connect(self._on_camera_diagnostics, Qt.ConnectionType.QueuedConnection)
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
        Handle gesture state changes from the camera worker (single-hand fallback).
        cx, cy are canvas coordinates of the index fingertip.
        """
        if self._multi_hand_active:
            return

        if state == GestureState.NONE:
            self._gesture_hud.setText("Mode: NO HAND")
            self._status_mode.setText("Mode: NO HAND")
            self._scene.move_laser(0.0, 0.0, visible=False)
            if self._current_stroke_id is not None:
                self._finalize_stroke()
            return

        bx, by = self._scene_coords(cx, cy)

        mode_name = state.name
        self._gesture_hud.setText(f"Mode: {mode_name}")
        self._status_mode.setText(f"Mode: {mode_name}")

        # Handle explicit manual teaching tool overrides
        if self._teaching_mode == "SPOTLIGHT":
            self._scene.move_spotlight(bx, by)
            self._scene.move_laser(bx, by, visible=True)
            if self._current_stroke_id is not None:
                self._finalize_stroke()
            return
        elif self._teaching_mode == "LOUPE":
            self._scene.move_loupe(bx, by)
            self._scene.move_laser(bx, by, visible=True)
            if self._current_stroke_id is not None:
                self._finalize_stroke()
            return
        elif self._teaching_mode == "LASER":
            if state in (GestureState.POINT, GestureState.DRAW):
                self._scene.move_laser(bx, by, visible=True)
                if self._current_stroke_id is not None:
                    self._finalize_stroke()
            return
        elif self._teaching_mode in ("ERASER", "ERASER_OBJECT", "ERASER_AREA"):
            self._scene.move_laser(bx, by, visible=False)
            if state == GestureState.DRAW:
                erased = self._scene.erase_strokes_at(bx, by, radius=self._eraser_radius)
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
                self._selected_items = []
                self._selected_strokes = []
            else:
                rect = QRectF(self._select_start, QPointF(bx, by)).normalized()
                self._scene.show_selection_rect(rect)

        elif state == GestureState.SWIPE_LEFT:
            self._on_prev_page()

        elif state == GestureState.SWIPE_RIGHT:
            self._on_next_page()

        elif state == GestureState.CONFIRM:
            rect = None
            if self._scene._select_rect_item and self._scene._select_rect_item.isVisible():
                rect = self._scene._select_rect_item.rect()
            if rect is not None and not rect.isEmpty():
                self._selected_items = self._scene.get_items_in_rect(rect)
                self._selected_strokes = self._scene.get_strokes_in_rect(rect)
            self._select_start = None
            self._trigger_ocr_on_selection()

    def _handle_selection_drag(self, bx: float, by: float, is_pinched: bool) -> None:
        """Handle in-air dragging/translation of selected items with pinch gesture."""
        if not self._selected_items:
            return

        if is_pinched:
            if not self._is_dragging_selection:
                self._is_dragging_selection = True
                self._drag_last_pos = QPointF(bx, by)
                self._drag_total_delta = QPointF(0.0, 0.0)
            else:
                if self._drag_last_pos is not None:
                    dx = bx - self._drag_last_pos.x()
                    dy = by - self._drag_last_pos.y()
                    self._scene.translate_items(self._selected_items, dx, dy)
                    self._drag_total_delta += QPointF(dx, dy)
                    self._drag_last_pos = QPointF(bx, by)
                    if self._scene._select_rect_item and self._scene._select_rect_item.isVisible():
                        self._scene._select_rect_item.setRect(
                            self._scene._select_rect_item.rect().translated(dx, dy)
                        )
        else:
            if self._is_dragging_selection:
                total_dx = self._drag_total_delta.x()
                total_dy = self._drag_total_delta.y()
                if abs(total_dx) > 1.0 or abs(total_dy) > 1.0:
                    undo_list = self._page_undo_stack.setdefault(self._current_page_index, [])
                    undo_list.append(("translate", (list(self._selected_items), total_dx, total_dy)))
                    self._page_redo_stack.setdefault(self._current_page_index, []).clear()
                    self._status.showMessage(
                        f"Đã di chuyển {len(self._selected_items)} đối tượng ({total_dx:+.0f}, {total_dy:+.0f})", 2000
                    )
                self._is_dragging_selection = False
                self._drag_last_pos = None
                self._drag_total_delta = QPointF(0.0, 0.0)

    def _delete_selected_items(self) -> None:
        """Delete currently selected items/strokes from the scene with undo/redo."""
        if not self._selected_items:
            return
        del_strokes, del_items = self._scene.delete_items(self._selected_items)
        if del_strokes or del_items:
            undo_list = self._page_undo_stack.setdefault(self._current_page_index, [])
            undo_list.append(("delete", (del_strokes, del_items)))
            self._page_redo_stack.setdefault(self._current_page_index, []).clear()
            self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
            self._status.showMessage(
                f"Đã xóa {len(del_strokes) + len(del_items)} đối tượng đã chọn", 2000
            )
        self._selected_items.clear()
        self._selected_strokes.clear()
        self._scene.hide_selection_rect()

    # ── Multi-Hand Coordination & Air Gestures ────────────────────────────────

    @Slot(list)
    def _on_multi_gesture_event(self, hands: list[HandLandmarkData]) -> None:
        """
        Handle real-time multi-hand coordination events:
          - Concurrent dual-hand drawing
          - Asymmetric interactions (Hand A draws, Hand B erases or navigates floating menu)
          - In-air menu opening, hover tracking, dwell selection, air tap / pinch clicks
        """
        self._multi_hand_active = True

        if not hands:
            # Finalize any ongoing hand strokes
            for hid in list(self._hand_strokes.keys()):
                self._finalize_hand_stroke(hid)
            self._scene.hide_laser()
            self._scene.hide_eraser_reticle()
            self._gesture_hud.setText("Mode: NO HAND")
            self._status_mode.setText("Mode: NO HAND")
            return

        active_track_ids = {h.track_id for h in hands}
        for hid in list(self._hand_strokes.keys()):
            if hid not in active_track_ids:
                self._finalize_hand_stroke(hid)
                self._scene.hide_laser(hid)
                self._scene.hide_eraser_reticle(hid)

        # Update HUD with tracking status
        hand_count = len(hands)
        primary_h = next((h for h in hands if h.role == HandRole.PRIMARY), hands[0])
        hud_txt = f"Hands: {hand_count} | Mode: {primary_h.state.name}"
        self._gesture_hud.setText(hud_txt)
        self._status_mode.setText(hud_txt)
        if hasattr(self, "_gesture_guide_overlay") and self._gesture_guide_overlay is not None and self._gesture_guide_overlay.isVisible():
            self._gesture_guide_overlay.update_active_gesture(primary_h.state, is_pinched=primary_h.is_pinched)

        # Update floating Cartesian Oxy coordinate HUD with primary hand position
        if hasattr(self._scene, "update_coordinate_hud"):
            self._scene.update_coordinate_hud(primary_h.canvas_pos[0], primary_h.canvas_pos[1])

        # Step 1: Floating Action Menu Interaction
        menu_interacting_hand_id: Optional[str] = None
        tapped_hand_ids: set[str] = set()
        if self._floating_menu.isVisible():
            # Control menu: prefer secondary hand if present
            menu_hand = next((h for h in hands if h.role == HandRole.SECONDARY), hands[0])
            bx, by = menu_hand.canvas_pos
            menu_interacting_hand_id = menu_hand.track_id

            # Update pointer & dwell progress
            self._floating_menu.update_pointer(bx, by, dwell_progress=menu_hand.dwell_progress)

            # Air Click or Air Tap
            if menu_hand.is_air_tap or menu_hand.is_pinch_click:
                self._floating_menu.trigger_click()
        else:
            # Check if air tap/click is over an interactive gate or canvas item
            item_consumed_tap = False
            for h in hands:
                if h.is_air_tap or h.is_pinch_click:
                    from src.ui.interactive_logic_gate import InteractiveLogicGateItem
                    from src.ui.classroom_metronome import ClassroomMetronomeItem
                    bx, by = h.canvas_pos
                    for item in self._scene.items():
                        if isinstance(item, InteractiveLogicGateItem):
                            if item.air_click_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#10b981", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                        elif isinstance(item, ClassroomMetronomeItem):
                            if item.isVisible() and item.air_click_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#10b981", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                        elif item.__class__.__name__ == "VirtualCompassItem":
                            if item.isVisible() and hasattr(item, "air_click_at") and item.air_click_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#3b82f6", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                        elif item.__class__.__name__ == "InteractiveCircuitWireItem":
                            if hasattr(item, "air_click_at") and item.air_click_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#10b981", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                        elif item.__class__.__name__ == "InteractiveAngleMeasureItem":
                            if item.isVisible() and hasattr(item, "get_handle_at") and item.get_handle_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#3b82f6", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                        elif item.__class__.__name__ == "InteractiveTriangleGeometryItem":
                            if item.isVisible() and hasattr(item, "get_handle_at") and item.get_handle_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#3b82f6", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                        elif item.__class__.__name__ == "InteractiveFunctionPlotterItem":
                            if item.isVisible() and hasattr(item, "probe_scene_pos"):
                                pos = item.probe_scene_pos()
                                if pos is not None and math.hypot(bx - pos.x(), by - pos.y()) <= 24.0:
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#f59e0b", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveInclinedPlaneItem":
                            if item.isVisible() and hasattr(item, "air_click_at") and item.air_click_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#06b6d4", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                        elif item.__class__.__name__ in ("InteractiveTransformerAttentionItem", "InteractiveSpecialRelativityItem", "InteractiveQuantumTunnelingItem", "InteractiveCoriolisEarthItem", "InteractiveAcousticSonarItem", "InteractiveCloudChamberItem", "InteractiveOrbitalMechanicsItem", "ChemicalReactionCardItem", "MathDerivationCardItem"):

                            if item.isVisible() and hasattr(item, "handle_air_tap"):

                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):

                                    item_consumed_tap = True

                                    tapped_hand_ids.add(h.track_id)

                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#38bdf8", radius=9.0)

                                    self._finalize_hand_stroke(h.track_id)

                                    break

                        elif item.__class__.__name__ == "InteractiveOpticsRefractionItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#38bdf8", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveStandingWaveItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#38bdf8", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveVseprGeometryItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#8b5cf6", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveDnaReplicationItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#ec4899", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveKeplerOrbitItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#38bdf8", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveGraphPathfindingItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#06b6d4", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveElectricFieldItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#38bdf8", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break
                        elif item.__class__.__name__ == "InteractiveLorentzForceItem":
                            if item.isVisible() and hasattr(item, "handle_air_tap"):
                                if item.handle_air_tap(QPointF(bx, by), board=self._scene):
                                    item_consumed_tap = True
                                    tapped_hand_ids.add(h.track_id)
                                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#ef4444", radius=9.0)
                                    self._finalize_hand_stroke(h.track_id)
                                    break

                    # Also check floating view overlays (Mini-Map)
                    if not item_consumed_tap and hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
                        minimap = self._zoom_ctrl.get_minimap()
                        if minimap is not None and minimap.isVisible():
                            if minimap.air_click_at(bx, by):
                                item_consumed_tap = True
                                tapped_hand_ids.add(h.track_id)
                                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#3b82f6", radius=9.0)
                                self._finalize_hand_stroke(h.track_id)
                                break
                    if item_consumed_tap:
                        break

            if not item_consumed_tap:
                # Menu is closed: check if any hand triggers opening in air
                for h in hands:
                    can_open = (
                        h.role == HandRole.SECONDARY or
                        (len(hands) == 1 and h.state in (GestureState.POINT, GestureState.NONE))
                    )
                    if can_open and (h.is_air_tap or h.is_pinch_click or h.dwell_progress >= 0.85):
                        bx, by = h.canvas_pos
                        self._floating_menu.show_at(bx, by)
                        menu_interacting_hand_id = h.track_id
                        break

        # Step 1.4: Bimanual STEM Object Manipulation (Two-Hand Scale, Rotate, Translate)
        if hasattr(self, "_bimanual_manipulator") and self._bimanual_manipulator is not None:
            if len(hands) >= 2 and not self._floating_menu.isVisible():
                h1_m, h2_m = hands[0], hands[1]
                both_m_pinched = (h1_m.is_pinched or h1_m.state == GestureState.SELECT) and (h2_m.is_pinched or h2_m.state == GestureState.SELECT)
                p1_sc = (h1_m.canvas_pos[0], h1_m.canvas_pos[1])
                p2_sc = (h2_m.canvas_pos[0], h2_m.canvas_pos[1])
                mid_sc = QPointF((p1_sc[0] + p2_sc[0]) * 0.5, (p1_sc[1] + p2_sc[1]) * 0.5)
                cand_items = self._scene.items(mid_sc) if hasattr(self, "_scene") and self._scene else []
                stem_cands = [
                    itm for itm in cand_items
                    if itm not in (getattr(self._scene, "_background_item", None), getattr(self._scene, "_bg_item", None), getattr(self._scene, "_grid_item", None))
                    and not getattr(itm, "_is_stroke_item", False)
                    and (hasattr(itm, "rotate_by") or hasattr(itm, "air_click_at") or type(itm).__name__.startswith("Interactive") or type(itm).__name__.startswith("Virtual") or type(itm).__name__.endswith("CardItem"))
                ]
                if (self._bimanual_manipulator.is_manipulating() or stem_cands) and both_m_pinched:
                    t_stem = self._bimanual_manipulator.process_hands(
                        hand1_pos=p1_sc,
                        hand2_pos=p2_sc,
                        hand1_active=True,
                        hand2_active=True,
                        candidate_items=stem_cands,
                    )
                    if t_stem is not None:
                        for h in hands:
                            self._finalize_hand_stroke(h.track_id)
                            self._scene.move_laser(h.canvas_pos[0], h.canvas_pos[1], visible=True, hand_id=h.track_id, color="#f59e0b", radius=8.0)
                        return
                elif self._bimanual_manipulator.is_manipulating():
                    self._bimanual_manipulator.end_manipulation()

        # Step 1.5: Bimanual Two-Hand Pinch-to-Zoom & Touchless Air Panning
        if hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
            # Check two-hand pinch-to-zoom
            if len(hands) >= 2 and not self._floating_menu.isVisible():
                h1, h2 = hands[0], hands[1]
                both_pinched = (h1.is_pinched or h1.state == GestureState.SELECT) and (h2.is_pinched or h2.state == GestureState.SELECT)
                if both_pinched:
                    p1 = h1.index_tip_norm
                    p2 = h2.index_tip_norm
                    dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                    last_dist = getattr(self, "_last_bimanual_pinch_dist", None)
                    if last_dist is not None and last_dist > 0.001:
                        delta = dist - last_dist
                        self._zoom_ctrl.apply_pinch_delta(delta)
                    self._last_bimanual_pinch_dist = dist
                    for h in hands:
                        self._finalize_hand_stroke(h.track_id)
                        self._scene.move_laser(h.canvas_pos[0], h.canvas_pos[1], visible=True, hand_id=h.track_id, color="#6366f1", radius=7.0)
                    return
                else:
                    self._last_bimanual_pinch_dist = None
            else:
                self._last_bimanual_pinch_dist = None

            # Check touchless air pan mode
            if self._zoom_ctrl.is_pan_mode() and not self._floating_menu.isVisible():
                if primary_h.state in (GestureState.POINT, GestureState.DRAW, GestureState.NONE) or primary_h.is_pinched:
                    last_pan = getattr(self, "_last_air_pan_pos", None)
                    curr_pos = (primary_h.canvas_pos[0], primary_h.canvas_pos[1])
                    if last_pan is not None:
                        dx = curr_pos[0] - last_pan[0]
                        dy = curr_pos[1] - last_pan[1]
                        self._zoom_ctrl.pan_by(dx, dy)
                    self._last_air_pan_pos = curr_pos
                else:
                    self._last_air_pan_pos = None
                for h in hands:
                    self._finalize_hand_stroke(h.track_id)
                    self._scene.move_laser(h.canvas_pos[0], h.canvas_pos[1], visible=True, hand_id=h.track_id, color="#10b981", radius=8.0)
                return
            else:
                self._last_air_pan_pos = None

        # Step 2: Process hands for canvas operations
        for h in hands:
            if h.track_id in tapped_hand_ids:
                continue

            if h.track_id == menu_interacting_hand_id and self._floating_menu.isVisible():
                # Hand is busy controlling menu
                self._scene.move_laser(h.canvas_pos[0], h.canvas_pos[1], visible=True, hand_id=h.track_id, color="#38bdf8", radius=8.0)
                self._finalize_hand_stroke(h.track_id)
                continue

            bx, by = h.canvas_pos

            # Asymmetric role: Secondary hand in DRAW_AND_ERASE mode
            if h.role == HandRole.SECONDARY and self._multi_hand_mode == "draw_and_erase":
                self._scene.move_eraser_reticle(bx, by, radius=self._eraser_radius, visible=True, hand_id=h.track_id)
                self._finalize_hand_stroke(h.track_id)
                if h.state == GestureState.SWIPE_LEFT:
                    self._on_prev_page()
                elif h.state == GestureState.SWIPE_RIGHT:
                    self._on_next_page()
                elif h.state == GestureState.DRAW or h.is_pinched:
                    erased = self._scene.erase_strokes_at(bx, by, radius=self._eraser_radius)
                    if erased:
                        self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
                continue

            # Asymmetric role: Secondary hand in DRAW_AND_TOOL mode
            if h.role == HandRole.SECONDARY and self._multi_hand_mode == "draw_and_tool":
                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#a855f7", radius=10.0)
                self._finalize_hand_stroke(h.track_id)
                if h.state == GestureState.SWIPE_LEFT:
                    self._on_prev_page()
                elif h.state == GestureState.SWIPE_RIGHT:
                    self._on_next_page()
                elif h.is_air_tap or h.is_pinch_click:
                    self._floating_menu.show_at(bx, by)
                continue

            # Drawing or teaching mode dispatch
            if self._teaching_mode == "SPOTLIGHT":
                self._scene.move_spotlight(bx, by)
                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id)
                self._finalize_hand_stroke(h.track_id)
                # Bimanual secondary hand aperture tuning
                if len(hands) >= 2:
                    sec_h = next((other for other in hands if other.track_id != h.track_id), None)
                    if sec_h is not None:
                        norm_y = sec_h.index_tip_norm[1]
                        radius = float(np.clip(50.0 + (1.0 - norm_y) * 260.0, 40.0, 350.0))
                        self._scene.set_spotlight_radius(radius)
            elif self._teaching_mode == "LOUPE":
                self._scene.move_loupe(bx, by)
                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id)
                self._finalize_hand_stroke(h.track_id)
                # Bimanual secondary hand zoom tuning
                if len(hands) >= 2:
                    sec_h = next((other for other in hands if other.track_id != h.track_id), None)
                    if sec_h is not None:
                        norm_y = sec_h.index_tip_norm[1]
                        zoom = float(np.clip(1.5 + (1.0 - norm_y) * 2.5, 1.5, 4.0))
                        self._scene.set_loupe_zoom(zoom)
            elif self._teaching_mode == "LASER":
                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id)
                self._finalize_hand_stroke(h.track_id)
            elif self._teaching_mode in ("ERASER", "ERASER_OBJECT", "ERASER_AREA"):
                self._scene.move_eraser_reticle(bx, by, radius=self._eraser_radius, visible=True, hand_id=h.track_id)
                self._finalize_hand_stroke(h.track_id)
                if h.state == GestureState.DRAW or h.is_pinched:
                    if self._teaching_mode == "ERASER":
                        erased = self._scene.erase_strokes_at(bx, by, radius=self._eraser_radius)
                    elif self._teaching_mode == "ERASER_OBJECT":
                        res = self._scene.erase_object_at(bx, by, radius=self._eraser_radius)
                        erased = [res] if res else []
                    else:
                        erased = self._scene.erase_area(QRectF(bx - self._eraser_radius, by - self._eraser_radius, 2 * self._eraser_radius, 2 * self._eraser_radius))
                    if erased:
                        self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
            elif self._teaching_mode == "SELECT":
                self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id, color="#38bdf8", radius=7.0)
                self._finalize_hand_stroke(h.track_id)
                if h.state == GestureState.SWIPE_LEFT:
                    self._on_prev_page()
                elif h.state == GestureState.SWIPE_RIGHT:
                    self._on_next_page()
                elif self._selected_items and (h.is_pinched or h.state == GestureState.DRAW):
                    self._handle_selection_drag(bx, by, is_pinched=True)
                else:
                    if self._is_dragging_selection:
                        self._handle_selection_drag(bx, by, is_pinched=False)
                    if h.state == GestureState.SELECT or h.is_pinched:
                        if self._select_start is None:
                            self._select_start = QPointF(bx, by)
                            self._selected_items = []
                            self._selected_strokes = []
                        else:
                            rect = QRectF(self._select_start, QPointF(bx, by)).normalized()
                            self._scene.show_selection_rect(rect)
                    elif h.state == GestureState.CONFIRM:
                        rect = None
                        if self._scene._select_rect_item and self._scene._select_rect_item.isVisible():
                            rect = self._scene._select_rect_item.rect()
                        if rect is not None and not rect.isEmpty():
                            self._selected_items = self._scene.get_items_in_rect(rect)
                            self._selected_strokes = self._scene.get_strokes_in_rect(rect)
                        self._select_start = None
                        self._trigger_ocr_on_selection()
            else:
                # Normal ink writing
                if h.state == GestureState.POINT:
                    self._scene.move_laser(bx, by, visible=True, hand_id=h.track_id)
                    self._finalize_hand_stroke(h.track_id)
                elif h.state == GestureState.DRAW:
                    self._scene.move_laser(bx, by, visible=False, hand_id=h.track_id)
                    t_ms = time.perf_counter() * 1000.0
                    stroke_color = self._ink_color if h.role == HandRole.PRIMARY else self._secondary_ink_color
                    if h.track_id not in self._hand_strokes:
                        self._start_new_hand_stroke(h.track_id, bx, by, t_ms, color=stroke_color)
                    else:
                        self._extend_hand_stroke(h.track_id, bx, by, t_ms)
                elif h.state == GestureState.PAUSE:
                    self._scene.move_laser(bx, by, visible=False, hand_id=h.track_id)
                    self._finalize_hand_stroke(h.track_id)
                elif h.state == GestureState.SELECT and h.role == HandRole.PRIMARY:
                    if self._select_start is None:
                        self._select_start = QPointF(bx, by)
                        self._selected_items = []
                        self._selected_strokes = []
                    else:
                        rect = QRectF(self._select_start, QPointF(bx, by)).normalized()
                        self._scene.show_selection_rect(rect)
                elif h.state == GestureState.SWIPE_LEFT:
                    self._on_prev_page()
                elif h.state == GestureState.SWIPE_RIGHT:
                    self._on_next_page()
                elif h.state == GestureState.CONFIRM and h.role == HandRole.PRIMARY:
                    rect = None
                    if self._scene._select_rect_item and self._scene._select_rect_item.isVisible():
                        rect = self._scene._select_rect_item.rect()
                    if rect is not None and not rect.isEmpty():
                        self._selected_items = self._scene.get_items_in_rect(rect)
                        self._selected_strokes = self._scene.get_strokes_in_rect(rect)
                    self._select_start = None
                    self._trigger_ocr_on_selection()

    # ── Floating Action Menu Slots ────────────────────────────────────────────

    def _toggle_floating_menu(self) -> None:
        """Toggle floating menu visibility centered on canvas."""
        if self._floating_menu.isVisible():
            self._floating_menu.close_menu()
        else:
            self._floating_menu.show_at(BOARD_WIDTH / 2.0, BOARD_HEIGHT / 2.0)

    @Slot(str, object)
    def _on_menu_action_triggered(self, cmd: str, val: Any) -> None:
        """Apply tool, color, eraser, or shape selected from in-air menu."""
        logger.info("Menu action received: %s, %s", cmd, val)
        if cmd == "tool":
            if val == "laser":
                self._set_teaching_mode("LASER")
                self._scene.hide_eraser_reticle()
            elif val == "spotlight":
                self._set_teaching_mode("SPOTLIGHT")
                self._scene.hide_eraser_reticle()
            elif val == "loupe":
                self._set_teaching_mode("LOUPE")
                self._scene.hide_eraser_reticle()
            elif val == "select":
                self._set_teaching_mode("SELECT")
                self._scene.hide_laser()
            elif val == "pen":
                self._set_teaching_mode("PEN")
        elif cmd == "color":
            self._set_ink_color(str(val))
            self._set_teaching_mode("PEN")
        elif cmd == "width":
            w = float(val)
            self._ink_min_width = max(1.0, w * 0.6)
            self._ink_max_width = w * 1.5
            self._status.showMessage(f"Độ dày nét: {w:.1f} px", 2000)
        elif cmd == "eraser":
            if val == "stroke":
                self._set_teaching_mode("ERASER")
            elif val == "object":
                self._set_teaching_mode("ERASER_OBJECT")
            elif val == "area":
                self._set_teaching_mode("ERASER_AREA")
            elif val == "clear_page":
                self._scene.clear_page_annotations()
                self._page_strokes[self._current_page_index] = []
                self._status.showMessage("Đã xóa toàn bộ nét vẽ trên trang", 2500)
        elif cmd == "action":
            if val == "undo":
                self._on_undo()
            elif val == "redo":
                self._on_redo()
            elif val == "toggle_desktop":
                self._on_toggle_desktop_overlay()
            elif val == "toggle_board_theme":
                self._on_toggle_board_theme()
            elif val == "export_image":
                self._on_export_image()
            elif val == "zoom_in":
                self._on_zoom_in()
            elif val == "zoom_out":
                self._on_zoom_out()
            elif val == "zoom_reset":
                self._on_zoom_reset()
            elif val == "pan":
                self._on_toggle_pan()
            elif val == "toggle_minimap":
                self._on_toggle_minimap()
            elif val == "toggle_coord_hud":
                self._on_toggle_coordinate_hud()
            elif val == "cycle_laser_color":
                self._on_cycle_laser_color()
        elif cmd == "shape":
            if val == "freeform":
                self._snapping_on = False
            else:
                self._snapping_on = True
        elif cmd == "instrument":
            if val == "ruler":
                self._on_toggle_ruler()
            elif val == "protractor":
                self._on_toggle_protractor()
            elif val == "compass":
                self._on_toggle_compass()
            elif val == "angle_tool":
                self._on_toggle_angle_tool()
            elif val == "triangle_tool":
                self._on_toggle_triangle_tool()
            elif val == "function_plotter":
                self._on_toggle_function_plotter()
            elif val == "inclined_plane":
                self._on_toggle_inclined_plane()
            elif val in ("optics_refraction", "optics"):
                self._on_toggle_optics_refraction()
            elif val in ("standing_wave", "wave"):
                self._on_toggle_standing_wave()
            elif val in ("vsepr_geometry", "vsepr", "molecular_geometry"):
                self._on_toggle_vsepr_geometry()
            elif val in ("dna_replication", "dna_double_helix", "dna"):
                self._on_toggle_dna_replication()
            elif val in ("kepler_orbit", "kepler", "orbit", "astronomy"):
                self._on_toggle_kepler_orbit()
            elif val in ("graph_pathfinding", "graph", "dijkstra", "astar", "pathfinding"):
                self._on_toggle_graph_pathfinding()
            elif val in ("electric_field", "coulomb", "charges", "electrostatic"):
                self._on_toggle_electric_field()
            elif val in ("lorentz_force", "lorentz", "magnetism", "b_field"):
                self._on_toggle_lorentz_force()
            elif val in ("trophic_web", "ecosystem", "lotka_volterra", "food_web", "trophic"):
                self._on_toggle_trophic_web()
            elif val in ("gas_laws", "ideal_gas", "gas", "maxwell_boltzmann", "thermodynamics", "gas_chamber"):
                self._on_toggle_gas_laws()
            elif val in ("titration", "acid_base", "ph_buffer", "henderson_hasselbalch", "ph_curve"):
                self._on_toggle_titration()
            elif val in ("harmonic_oscillator", "oscillator", "resonance", "spring_mass", "dao_dong", "shm"):
                self._on_toggle_harmonic_oscillator()
            elif val in ("radioactive_decay", "decay", "nuclear", "phong_xa", "half_life"):
                self._on_toggle_radioactive_decay()
            elif val in ("chemical_kinetics", "kinetics", "arrhenius", "equilibrium", "can_bang"):
                self._on_toggle_chemical_kinetics()
            elif val in ("cryptography", "rsa", "diffie_hellman", "crypto", "mat_ma", "modular"):
                self._on_toggle_cryptography()
            elif val in ("gravitational_orbits", "orbits", "lagrange", "n_body", "kepler_nbody", "hap_dan"):
                self._on_toggle_gravitational_orbits()
            elif val in ("spacetime_relativity", "minkowski", "relativity", "lorentz", "tuong_doi", "khong_thoi_gian"):
                self._on_toggle_spacetime_relativity()
            elif val in ("young_interference", "young", "interference", "giao_thoa", "nhieu_xa", "grating", "cach_tu", "quang_song"):
                self._on_toggle_young_interference()
            elif val in ("rlc_circuit", "rlc", "fresnel", "phasor", "xoay_chieu", "dien_xoay_chieu", "cong_huong"):
                self._on_toggle_rlc_circuit()
            elif val in ("thin_lens", "lens", "thau_kinh", "optical_bench", "bang_quang_hoc", "kinh_lup"):
                self._on_toggle_thin_lens()
            elif val in ("bragg", "xrd", "x_ray_diffraction", "nhieu_xa_bragg", "tinh_the"):
                self._on_toggle_bragg_diffraction()
            elif val in ("buffer", "buffer_system", "he_dem", "henderson", "van_slyke"):
                self._on_toggle_buffer_system()
            elif val in ("stern_gerlach", "spin", "spin_half", "gerlach", "luong_tu_hoa_khong_gian"):
                self._on_toggle_stern_gerlach()
            elif val in ("enzyme", "enzyme_kinetics", "michaelis_menten", "lineweaver_burk", "dong_hoc_enzyme"):
                self._on_toggle_enzyme_kinetics()
            elif val in ("poiseuille", "hagen_poiseuille", "dong_chay_nhot", "reynolds", "viscous_flow"):
                self._on_toggle_poiseuille_flow()
            elif val in ("hubble", "cosmic", "universe", "redshift", "vu_tru_gian_no", "cosmology"):
                self._on_toggle_cosmic_expansion()
            elif val in ("seismic", "earthquake", "song_dia_chan", "tam_chan", "epicenter", "seismology"):
                self._on_toggle_seismic_waves()
            elif val in ("spectrometer", "mass_spectrometer", "wien", "khoi_pho", "dong_vi", "isotope"):
                self._on_toggle_mass_spectrometer()
            elif val in ("spectrophotometry", "beer_lambert", "absorbance", "quang_pho_hap_thu", "beer", "cuvette", "do_hap_thu"):
                self._on_toggle_spectrophotometry()
            elif val in ("lien_ket_hat_nhan", "binding_energy", "semf", "do_hut_khoi", "nuclear_binding"):
                self._on_toggle_nuclear_binding_energy()
            elif val in ("compton", "tan_xa_compton", "compton_scattering", "photon_electron"):
                self._on_toggle_compton_scattering()
            elif val in ("gravitational_lensing", "thau_kinh_hap_dan", "einstein_ring", "nhan_einstein", "einstein_cross", "tu_ong_quang"):
                self._on_toggle_gravitational_lensing()
            elif val in ("chemical_equilibrium", "can_bang_hoa_hoc", "le_chatelier", "haber_bosch", "van_t_hoff", "nhiet_dong_hoa_hoc"):
                self._on_toggle_chemical_equilibrium()
            elif val in ("hodgkin_huxley", "hodgkin", "action_potential", "dien_the_hoat_dong", "dien_sinh_ly", "neuron", "axon", "ttx", "tea"):
                self._on_toggle_hodgkin_huxley()
            elif val in ("mst", "cay_khung_nho_nhat", "minimum_spanning_tree", "kruskal", "prim", "dsu", "union_find", "graph_algorithm"):
                self._on_toggle_mst_visualizer()
            elif val in ("hall", "hall_effect", "hieu_ung_hall", "ban_dan", "semiconductor", "lorentz_drift"):
                self._on_toggle_hall_effect()
            elif val in ("radiation", "phong_xa", "shielding", "hvl", "gamma", "alpha", "beta", "che_chan_phong_xa", "geiger"):
                self._on_toggle_radiation_shielding()
            elif val in ("maxwell", "boltzmann", "maxwell_boltzmann", "phan_bo_van_toc", "kinetic_theory", "thuyet_dong_hoc"):
                self._on_toggle_maxwell_boltzmann()
            elif val in ("orbital", "molecular_orbital", "obitan", "mo_lcao", "hybridization", "lai_hoa", "obitan_phan_tu"):
                self._on_toggle_molecular_orbital()
            elif val in ("rlc", "resonance", "cong_huong", "cong_huong_rlc", "phasor", "freznel", "impedance", "tong_tro"):
                self._on_toggle_rlc_resonance()
            elif val in ("stress_strain", "ung_suat", "bien_dang", "young_modulus", "dan_hoi", "keo_dut", "tensile_test", "do_ben_keo"):
                self._on_toggle_stress_strain()
            elif val in ("kepler", "orbit", "quy_dao", "hanh_tinh", "ve_tinh", "vis_viva", "apogee", "perigee", "can_nhat", "vien_nhat"):
                self._on_toggle_kepler_orbits()
            elif val in ("logic", "logic_gates", "cong_logic", "half_adder", "full_adder", "mach_cong", "truth_table", "bang_chan_tri", "boole"):
                self._on_toggle_logic_circuits()
            elif val in ("photoelectric", "photoelectric_effect", "quang_dien", "einstein", "luong_tu", "hieu_ung_quang_dien"):
                self._on_toggle_photoelectric_effect()
            elif val in ("carnot", "carnot_engine", "dong_co_nhiet", "chu_trinh_carnot", "nhiet_dong_luc"):
                self._on_toggle_carnot_engine()
            elif val in ("bohr", "bohr_atom", "nguyen_tu_bohr", "quang_pho_hydro", "lyman", "balmer", "paschen"):
                self._on_toggle_bohr_atom()
            elif val in ("automata", "fsm", "dfa", "nfa", "finite_state", "may_trang_thai", "automaton"):
                self._on_toggle_automata()
            elif val in ("sorting", "sort", "sap_xep", "big_o", "bubble_sort", "quick_sort", "merge_sort"):
                self._on_toggle_sorting()
            elif val in ("punnett", "genetics", "mendel", "di_truyen", "khung_punnett", "lai_giong"):
                self._on_toggle_punnett_genetics()
            elif val in ("organic", "organic_chem", "isomer", "dong_phan", "hoa_huu_co", "hydrocarbon"):
                self._on_toggle_organic_chemistry()
            elif val in ("fluid", "fluid_venturi", "venturi", "bernoulli", "chat_luu", "ap_suat_tinh", "manometer"):
                self._on_toggle_fluid_venturi()
            elif val in ("tree", "binary_tree", "bst", "huffman", "cay_nhi_phan", "duyet_cay"):
                self._on_toggle_binary_tree()
            elif val in ("electro", "galvanic", "daniell", "electrolysis", "pin_dien_hoa", "dien_phan", "nernst"):
                self._on_toggle_electrochemistry()
            elif val in ("circular", "circular_motion", "centripetal", "tron_deu", "huong_tam", "banked_curve", "vong_xiec"):
                self._on_toggle_circular_motion()
            elif val in ("titration", "ph_titration", "chuan_do", "duong_cong_ph", "axit_bazo", "burette", "chi_thi"):
                self._on_toggle_ph_titration()
            elif val in ("induction", "faraday", "lenz", "cam_ung_dien_tu", "tu_thong", "may_phat_dien", "bien_ap"):
                self._on_toggle_electromagnetic_induction()
            elif val in ("dp", "dynamic_programming", "knapsack", "cai_tui", "quy_hoach_dong", "lcs", "coin_change", "bang_dp"):
                self._on_toggle_dynamic_programming()
            elif val == "timer":
                self._on_toggle_timer()
            elif val == "metronome":
                self._on_toggle_metronome()
        elif cmd == "stem":
            if val == "oxy":
                self._on_insert_stem_oxy()
            elif val == "function":
                self._on_insert_stem_function()
            elif val == "force":
                self._on_insert_stem_force()
            elif val == "circuit":
                self._on_insert_stem_circuit()
            elif val == "optics":
                self._on_insert_stem_optics()
            elif val == "chem_formula":
                self._on_insert_stem_chem_formula()
            elif val == "benzene":
                self._on_insert_stem_benzene()
            elif val == "molecule":
                self._on_insert_stem_molecule()
            elif val == "biology_callout":
                self._on_insert_stem_biology_callout()
            elif val == "dna":
                self._on_insert_stem_dna()
            elif val == "punnett":
                self._on_insert_stem_punnett()
            elif val == "flowchart":
                self._on_insert_stem_flowchart()
            elif val == "code_block":
                self._on_insert_stem_code_block()
            elif val == "triangle":
                self._on_insert_stem_triangle()
            elif val == "circle_radius":
                self._on_insert_stem_circle_radius()
            elif val == "angle":
                self._on_insert_stem_angle()
            elif val == "bond":
                self._on_insert_stem_organic_bond()
            elif val == "logic_gate":
                self._on_insert_stem_logic_gate()
            elif val == "flow_arrow":
                self._on_insert_stem_flow_arrow()

    @Slot()
    def _on_menu_closed(self) -> None:
        self._status.showMessage("Đã đóng Menu điều khiển không chạm", 2000)

    def _scene_coords(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert gesture coordinates to board scene coordinates.
        Supports normalized [0, 1] input and pixel-space coordinates within [0, BOARD_WIDTH/HEIGHT].
        """
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            sx = x * BOARD_WIDTH
            sy = y * BOARD_HEIGHT
        else:
            sx = x
            sy = y
        return max(0.0, min(float(BOARD_WIDTH), sx)), max(0.0, min(float(BOARD_HEIGHT), sy))

    # ── Multi-Hand Stroke Building ────────────────────────────────────────────

    def _start_new_hand_stroke(
        self,
        hand_id: str,
        x: float,
        y: float,
        t_ms: float,
        color: Optional[QColor] = None,
        width: float = 2.5,
    ) -> None:
        stroke_id = str(uuid.uuid4())
        f2d = OneEuroFilter2D(beta=getattr(self, "_one_euro_beta", 0.35))
        fx, fy = f2d.filter(x, y, t_ms)
        c = color or (self._ink_color if hand_id in ("primary", "hand_0") else self._secondary_ink_color)
        self._hand_strokes[hand_id] = HandLiveStroke(
            stroke_id=stroke_id,
            hand_id=hand_id,
            filter_2d=f2d,
            raw_points=[(x, y, t_ms)],
            color=c,
            width=width,
        )
        self._scene.begin_stroke_preview(fx, fy, c, hand_id=hand_id, width=width)
        if hand_id in ("primary", "hand_0"):
            self._current_stroke_id = stroke_id
            self._live_raw_points = [(x, y, t_ms)]
            self._filter = f2d

    def _extend_hand_stroke(self, hand_id: str, x: float, y: float, t_ms: float) -> None:
        hs = self._hand_strokes.get(hand_id)
        if hs is None:
            return
        hs.raw_points.append((x, y, t_ms))
        fx, fy = hs.filter_2d.filter(x, y, t_ms)
        self._scene.update_stroke_preview(fx, fy, hand_id=hand_id)
        if hand_id in ("primary", "hand_0"):
            self._live_raw_points.append((x, y, t_ms))

    def _finalize_hand_stroke(self, hand_id: str) -> None:
        hs = self._hand_strokes.pop(hand_id, None)
        if hs is None or len(hs.raw_points) < 2:
            self._scene.end_stroke_preview(hand_id)
            if hand_id in ("primary", "hand_0"):
                self._current_stroke_id = None
                self._live_raw_points.clear()
            return

        raw_pts = np.array([(p[0], p[1]) for p in hs.raw_points])
        t_ms = [p[2] for p in hs.raw_points]

        simplified = ramer_douglas_peucker(raw_pts, epsilon=self._rdp_epsilon)
        dt = np.diff(t_ms) if len(t_ms) > 1 else np.array([1.0])
        dt = np.where(dt < 1, 1, dt)
        d = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
        vels = d / dt

        segments = fit_cubic_bezier_spline(
            simplified,
            velocities=vels if len(vels) == len(simplified) else None,
            width_min=self._ink_min_width,
            width_max=self._ink_max_width,
        )
        widths = [seg.width_start for seg in segments] if segments else [hs.width]

        snap: Optional[SnapResult] = None
        if self._snapping_on and len(simplified) >= 2:
            try:
                snap = self._snapper.snap(simplified)
                if snap.shape_type == ShapeType.FREEFORM:
                    snap = None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Snap failed: %s", exc)

        stroke = InkStroke(
            stroke_id=hs.stroke_id,
            segments=segments,
            widths=widths,
            snap=snap,
            color=hs.color,
            timestamps_ms=t_ms,
            hand_id=hand_id,
        )
        self._scene.commit_stroke(stroke)
        self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
        self._page_undo_stack.setdefault(self._current_page_index, []).append(("stroke", stroke))
        self._page_redo_stack.setdefault(self._current_page_index, []).clear()

        if hand_id in ("primary", "hand_0"):
            self._current_stroke_id = None
            self._live_raw_points.clear()

    def _start_new_stroke(self, x: float, y: float, t_ms: float) -> None:
        self._start_new_hand_stroke("primary", x, y, t_ms, color=self._ink_color)

    def _extend_stroke(self, x: float, y: float, t_ms: float) -> None:
        self._extend_hand_stroke("primary", x, y, t_ms)

    def _finalize_stroke(self) -> None:
        self._finalize_hand_stroke("primary")

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _get_strokes_in_rect(self, rect: QRectF) -> list[Stroke]:
        """Convert InkStrokes intersecting rect into sampled Stroke objects with complete trajectory for OCR."""
        ocr_strokes: list[Stroke] = []
        for ink in self._scene._strokes.values():
            stroke_pts: list[StrokePoint] = []
            intersects = False
            for seg in ink.segments:
                p0 = QPointF(seg.p0[0], seg.p0[1])
                p3 = QPointF(seg.p3[0], seg.p3[1])
                if rect.contains(p0) or rect.contains(p3):
                    intersects = True
                for t in (0.0, 0.33, 0.67):
                    pt = seg.evaluate(t)
                    stroke_pts.append(StrokePoint(float(pt[0]), float(pt[1]), 0.0))
            if ink.segments:
                last_pt = ink.segments[-1].evaluate(1.0)
                stroke_pts.append(StrokePoint(float(last_pt[0]), float(last_pt[1]), 0.0))

            if intersects and stroke_pts:
                w = getattr(ink, "widths", [2.5])[0] if getattr(ink, "widths", None) else 2.5
                ocr_strokes.append(Stroke(points=stroke_pts, color_bgr=(0, 0, 0), width=float(w)))
        return ocr_strokes

    def _trigger_ocr_on_selection(self) -> None:
        """Run OCR on strokes within the selected region."""
        sel_item = self._scene._select_rect_item
        if sel_item is None:
            return
        rect = sel_item.rect()
        ocr_strokes = self._get_strokes_in_rect(rect)

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

            new_item = None
            if result.is_formula:
                render_rgba = result.render_image_rgba
                if latex_or_text and latex_or_text != getattr(result, "latex", ""):
                    try:
                        from src.services.smart_ink_ocr import LaTeXRenderer  # noqa: PLC0415
                        fresh_rgba = LaTeXRenderer.render(latex_or_text, dpi=120, fontsize=16)
                        if fresh_rgba is not None:
                            render_rgba = fresh_rgba
                    except Exception as exc:
                        logger.warning("Re-rendering edited LaTeX formula failed: %s", exc)
                if render_rgba is not None:
                    new_item = self._scene.add_latex_overlay(render_rgba, x, y)
                elif result.text or latex_or_text:
                    new_item = self._scene.add_text_overlay(latex_or_text or result.text, x, y)
            elif result.text or latex_or_text:
                new_item = self._scene.add_text_overlay(latex_or_text or result.text, x, y)

            if new_item is not None:
                new_item.setData(1, {
                    "text": result.text or latex_or_text,
                    "latex": (latex_or_text if result.is_formula else getattr(result, "latex", "")),
                    "is_formula": result.is_formula,
                    "conf": float(getattr(result, "confidence", 0.95)),
                })
                self._add_stem_item_to_current_page(new_item)
            self._status.showMessage("Đã chèn nội dung nhận dạng lên bảng.", 3000)

        def _on_derivation_confirmed(deriv_res, replace_ink: bool):
            try:
                from src.ui.math_derivation_card import MathDerivationCardItem  # noqa: PLC0415
                card = MathDerivationCardItem(deriv_res)
                card.setPos(x + 20.0, y + 80.0)
                self._scene.addItem(card)
                card.setData(1, {
                    "text": deriv_res.solution_raw,
                    "latex": deriv_res.solution_latex,
                    "is_formula": True,
                    "is_derivation_card": True,
                    "conf": 1.0,
                })
                self._add_stem_item_to_current_page(card)
                self._status.showMessage(f"Đã chèn thẻ lời giải ({deriv_res.operation.upper()}) lên bảng.", 3000)
            except Exception as exc:
                logger.warning("Failed to add MathDerivationCardItem: %s", exc)

        dlg.result_confirmed.connect(_on_confirmed)
        def _on_reaction_confirmed(react_res, replace_ink: bool):

            try:

                from src.ui.chemical_reaction_card import ChemicalReactionCardItem  # noqa: PLC0415

                card = ChemicalReactionCardItem(react_res)

                card.setPos(x + 20.0, y + 80.0)

                self._scene.addItem(card)

                card.setData(1, {

                    "text": react_res.balanced_equation_text,

                    "latex": react_res.balanced_equation_latex,

                    "is_formula": True,

                    "is_chemical_card": True,

                    "conf": 1.0,

                })

                self._add_stem_item_to_current_page(card)

                self._status.showMessage("Đã chèn thẻ phản ứng hóa học lên bảng.", 3000)
            except Exception as exc:

                logger.warning("Failed to add ChemicalReactionCardItem: %s", exc)


        dlg.derivation_confirmed.connect(_on_derivation_confirmed)

        dlg.reaction_confirmed.connect(_on_reaction_confirmed)
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
        ocr_strokes = self._get_strokes_in_rect(rect)
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
        """Save current canvas strokes and hide STEM items and sticky notes for the active page."""
        self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
        for itm in self._page_stem_items.get(self._current_page_index, []):
            if itm.scene() is self._scene:
                itm.setVisible(False)
        notes = self._scene.get_sticky_notes()
        if notes:
            self._page_sticky_notes[self._current_page_index] = list(notes)
            for n in notes:
                n.setVisible(False)

    def _restore_current_page_strokes(self) -> None:
        """Clear canvas and restore strokes, STEM items, and sticky notes for the active page."""
        self._scene.clear_all_strokes()
        saved_strokes = self._page_strokes.get(self._current_page_index, [])
        for stroke in saved_strokes:
            self._scene.commit_stroke(stroke)
        for itm in self._page_stem_items.get(self._current_page_index, []):
            if itm.scene() is None:
                self._scene.addItem(itm)
            itm.setVisible(True)
        for n in self._page_sticky_notes.get(self._current_page_index, []):
            if n.scene() is None:
                self._scene.addItem(n)
            n.setVisible(True)

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
                    sp = getattr(rec, "shape_params", None)
                    conf = float(getattr(rec, "snap_confidence", 1.0))
                    if sp:
                        snap = SnapResult(shape_type=st, params=sp, confidence=conf)
                    elif getattr(rec, "points", None) and len(rec.points) >= 2:
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
            self._page_pixmap_cache.clear()
            self._scene.clear_all_strokes()
            self._pages = self._doc_manager.import_document(Path(path))
            self._current_page_index = 0
            if hasattr(self, "_slide_drawer") and self._slide_drawer is not None:
                self._slide_drawer.set_pages(self._pages, self._current_page_index)
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
        self._load_airteach_package(Path(path))

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
                    shape_params = ink.snap.to_dict()["params"] if (ink.snap and ink.snap.is_geometric()) else {}
                    snap_conf = float(ink.snap.confidence) if ink.snap else 0.0
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
                        shape_params=shape_params,
                        snap_confidence=snap_conf,
                    ))
            page_image_paths = [
                Path(p.image_path) for p in getattr(self, "_pages", [])
                if Path(p.image_path).exists()
            ]
            recognized_records: list[RecognizedItem] = []
            for p_idx, p_items in self._page_stem_items.items():
                for itm in p_items:
                    data = itm.data(1)
                    brect = itm.sceneBoundingRect()
                    bbox = [float(brect.left()), float(brect.top()), float(brect.right()), float(brect.bottom())]
                    if isinstance(data, dict):
                        recognized_records.append(RecognizedItem(
                            item_id=str(uuid.uuid4())[:8],
                            page_index=p_idx,
                            text=str(data.get("text", "")),
                            latex=str(data.get("latex", "")),
                            confidence=float(data.get("conf", 0.95)),
                            bounding_box=bbox,
                            is_formula=bool(data.get("is_formula", False)),
                        ))
                    elif isinstance(itm, QGraphicsTextItem):
                        recognized_records.append(RecognizedItem(
                            item_id=str(uuid.uuid4())[:8],
                            page_index=p_idx,
                            text=itm.toPlainText(),
                            latex="",
                            confidence=0.95,
                            bounding_box=bbox,
                            is_formula=False,
                        ))
                    else:
                        stem_title = getattr(itm, "title", None) or getattr(itm, "name", None) or type(itm).__name__
                        recognized_records.append(RecognizedItem(
                            item_id=str(uuid.uuid4())[:8],
                            page_index=p_idx,
                            text=str(stem_title),
                            latex="",
                            confidence=0.95,
                            bounding_box=bbox,
                            is_formula=False,
                        ))

            self._doc_manager.save_lesson(
                output_path=Path(path),
                manifest=manifest,
                page_image_paths=page_image_paths,
                strokes=stroke_records,
                recognized=recognized_records,
            )
            self._status.showMessage(f"Đã lưu: {path}", 4000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi lưu", str(exc))

    def _on_export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất PDF", str(Path.home() / "bai_giang_chuthich.pdf"),
            "PDF (*.pdf)"
        )
        if not path:
            return
        try:
            from PIL import Image as _PILImage  # noqa: PLC0415

            self._stash_current_page_strokes()
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            pages = getattr(self, "_pages", [])
            pil_pages: list[_PILImage.Image] = []

            if pages:
                original_idx = self._current_page_index
                try:
                    for p_idx in range(len(pages)):
                        self._current_page_index = p_idx
                        self._show_current_page()

                        page_img = QImage(1920, 1080, QImage.Format.Format_ARGB32)
                        page_img.fill(Qt.GlobalColor.white)
                        p = QPainter(page_img)
                        p.setRenderHint(QPainter.RenderHint.Antialiasing)
                        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                        self._scene.render(p, QRectF(0, 0, 1920, 1080), self._scene.sceneRect())
                        p.end()

                        ptr = page_img.bits()
                        arr = np.frombuffer(ptr, np.uint8).reshape((1080, 1920, 4))
                        rgb = arr[:, :, [2, 1, 0]].copy()
                        pil_pages.append(_PILImage.fromarray(rgb))
                finally:
                    self._current_page_index = original_idx
                    self._show_current_page()
            else:
                # Whiteboard export: render current whiteboard scene
                page_img = QImage(1920, 1080, QImage.Format.Format_ARGB32)
                page_img.fill(Qt.GlobalColor.white)
                p = QPainter(page_img)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                self._scene.render(p, QRectF(0, 0, 1920, 1080), self._scene.sceneRect())
                p.end()

                ptr = page_img.bits()
                arr = np.frombuffer(ptr, np.uint8).reshape((1080, 1920, 4))
                rgb = arr[:, :, [2, 1, 0]].copy()
                pil_pages.append(_PILImage.fromarray(rgb))

            if not pil_pages:
                QMessageBox.warning(self, "Xuất PDF", "Không có nội dung trang để xuất.")
                return

            pil_pages[0].save(
                str(out_path),
                format="PDF",
                save_all=True,
                append_images=pil_pages[1:],
                resolution=144,
            )
            self._status.showMessage(f"Đã xuất PDF: {path}", 4000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi xuất PDF", str(exc))

    @Slot()
    def _on_export_image(self) -> None:
        """Open high-resolution canvas export dialog."""
        from src.ui.canvas_exporter import ExportImageDialog
        dlg = ExportImageDialog(self)
        dlg.exec()

    @Slot()
    def _on_toggle_board_theme(self) -> None:
        """Cycle through whiteboard / chalkboard / Vietnamese ô ly / slate themes."""
        from src.ui.lesson_board import THEME_CONFIGS
        next_theme = self._scene.cycle_board_theme()
        cfg = THEME_CONFIGS.get(next_theme, {})
        theme_name = cfg.get("name", next_theme)
        default_ink = cfg.get("default_ink")
        if default_ink:
            self._ink_color = default_ink
        try:
            from src.utils.audio_feedback import AudioFeedback
            AudioFeedback.play(AudioFeedback.SoundType.CLICK)
        except Exception:
            pass
        self._status.showMessage(f"Giao diện bảng: {theme_name}", 3000)

    @Slot(str, QColor)
    def _on_board_theme_changed(self, theme: str, default_ink: QColor) -> None:
        """Handle theme changes emitted by LessonBoard."""
        self._ink_color = default_ink

    def _show_current_page(self) -> None:
        pages = getattr(self, "_pages", [])
        if not pages:
            return
        idx = max(0, min(self._current_page_index, len(pages) - 1))
        self._current_page_index = idx
        page = pages[idx]

        # Use bounded LRU cache for page pixmaps
        if idx in self._page_pixmap_cache:
            pixmap = self._page_pixmap_cache[idx]
            self._page_pixmap_cache.move_to_end(idx)
            self._scene.set_background_page(pixmap)
        else:
            import cv2  # noqa: PLC0415
            img = cv2.imread(page.image_path)
            if img is not None:
                from src.ui.lesson_board import _numpy_bgr_to_qpixmap  # noqa: PLC0415
                raw_pix = _numpy_bgr_to_qpixmap(img)
                scaled = raw_pix.scaled(
                    BOARD_WIDTH, BOARD_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._page_pixmap_cache[idx] = scaled
                if len(self._page_pixmap_cache) > self._MAX_PAGE_CACHE:
                    self._page_pixmap_cache.popitem(last=False)
                self._scene.set_background_page(scaled)

        self._restore_current_page_strokes()
        self._board_view.fitInView(
            self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self._chat_panel.set_current_page_text(page.text)
        if hasattr(self, "_slide_drawer") and self._slide_drawer is not None:
            self._slide_drawer.set_current_page(self._current_page_index)
        self._update_status_bar()

    # ── Page navigation ───────────────────────────────────────────────────────

    def _on_prev_page(self) -> None:
        if self._current_page_index > 0:
            self._stash_current_page_strokes()
            self._current_page_index -= 1
            self._show_current_page()
            try:
                from src.utils.audio_feedback import AudioFeedbackManager
                AudioFeedbackManager.instance().play_page_turn()
            except Exception:
                pass

    def _on_next_page(self) -> None:
        pages = getattr(self, "_pages", [])
        if self._current_page_index < len(pages) - 1:
            self._stash_current_page_strokes()
            self._current_page_index += 1
            self._show_current_page()
            try:
                from src.utils.audio_feedback import AudioFeedbackManager
                AudioFeedbackManager.instance().play_page_turn()
            except Exception:
                pass

    @Slot(int)
    def _go_to_page(self, idx: int) -> None:
        """Jump directly to target page index."""
        pages = getattr(self, "_pages", [])
        if not pages:
            return
        target_idx = max(0, min(idx, len(pages) - 1))
        if target_idx == self._current_page_index:
            return
        self._stash_current_page_strokes()
        self._current_page_index = target_idx
        self._show_current_page()
        try:
            from src.utils.audio_feedback import AudioFeedback
            AudioFeedback.play(AudioFeedback.SoundType.PAGE_TURN)
        except Exception:
            pass

    @Slot()
    def _on_toggle_slide_dock(self) -> None:
        """Toggle visibility of the Slide Thumbnail Drawer dock."""
        if hasattr(self, "_slide_dock") and self._slide_dock is not None:
            self._slide_dock.setHidden(not self._slide_dock.isHidden())

    @Slot()
    def _on_add_sticky_note(self) -> None:
        """Add an acrylic floating sticky note to the current page."""
        from src.ui.sticky_note_item import StickyNoteEditDialog
        dlg = StickyNoteEditDialog("Lưu ý bài học", "", "yellow", self)
        if dlg.exec():
            title, text = dlg.get_data()
            note = self._scene.add_sticky_note(
                title=title or "Ghi chú",
                text=text,
                theme="yellow",
                x=600.0,
                y=180.0,
            )
            if self._current_page_index not in self._page_sticky_notes:
                self._page_sticky_notes[self._current_page_index] = []
            self._page_sticky_notes[self._current_page_index].append(note)
            try:
                from src.utils.audio_feedback import AudioFeedback
                AudioFeedback.play(AudioFeedback.SoundType.CLICK)
            except Exception:
                pass
            self._status.showMessage("Đã gắn thẻ ghi chú dán (Sticky Note)", 3000)

    @Slot()
    def _on_add_blank_page(self) -> None:
        """Add a new blank whiteboard slide to the presentation."""
        import time
        from src.services.document_manager import PageInfo
        self._stash_current_page_strokes()

        work_dir = Path.home() / ".airteach" / "temp_slides"
        work_dir.mkdir(parents=True, exist_ok=True)
        new_idx = len(self._pages)
        blank_p = work_dir / f"blank_slide_{new_idx}_{int(time.time())}.png"

        blank_img = QImage(1920, 1080, QImage.Format.Format_RGB32)
        blank_img.fill(Qt.GlobalColor.white)
        blank_img.save(str(blank_p), "PNG")

        new_page = PageInfo(
            page_index=new_idx,
            image_path=str(blank_p),
            width_px=1920,
            height_px=1080,
            text="",
        )
        self._pages.append(new_page)
        self._current_page_index = new_idx
        if hasattr(self, "_slide_drawer") and self._slide_drawer is not None:
            self._slide_drawer.set_pages(self._pages, new_idx)
        self._show_current_page()
        self._status.showMessage(f"Đã tạo trang trắng mới (Trang {new_idx + 1})", 3000)

    @Slot()
    def _on_delete_page(self) -> None:
        """Delete the current slide if more than 1 slide exists."""
        if len(self._pages) <= 1:
            QMessageBox.warning(self, "Xóa trang", "Không thể xóa trang duy nhất còn lại.")
            return

        del_idx = self._current_page_index
        del self._pages[del_idx]

        for i, p in enumerate(self._pages):
            p.page_index = i

        self._page_strokes.pop(del_idx, None)
        self._page_sticky_notes.pop(del_idx, None)

        self._current_page_index = max(0, min(del_idx, len(self._pages) - 1))
        if hasattr(self, "_slide_drawer") and self._slide_drawer is not None:
            self._slide_drawer.set_pages(self._pages, self._current_page_index)
        self._show_current_page()
        self._status.showMessage(f"Đã xóa trang {del_idx + 1}. Hiện có {len(self._pages)} trang.", 3000)

    @Slot()
    def _on_duplicate_page(self) -> None:
        """Duplicate active slide into a new slide right after it."""
        if not self._pages:
            return
        from src.services.document_manager import PageInfo
        self._stash_current_page_strokes()

        cur_p = self._pages[self._current_page_index]
        dup_idx = self._current_page_index + 1
        dup_page = PageInfo(
            page_index=dup_idx,
            image_path=cur_p.image_path,
            width_px=cur_p.width_px,
            height_px=cur_p.height_px,
            text=cur_p.text,
        )
        self._pages.insert(dup_idx, dup_page)
        for i, p in enumerate(self._pages):
            p.page_index = i

        cur_strokes = self._page_strokes.get(self._current_page_index, [])
        self._page_strokes[dup_idx] = list(cur_strokes)

        self._current_page_index = dup_idx
        if hasattr(self, "_slide_drawer") and self._slide_drawer is not None:
            self._slide_drawer.set_pages(self._pages, dup_idx)
        self._show_current_page()
        self._status.showMessage(f"Đã nhân bản slide thành Trang {dup_idx + 1}", 3000)

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _on_undo(self) -> None:
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_undo()
        except Exception:
            pass
        undo_list = self._page_undo_stack.setdefault(self._current_page_index, [])
        if undo_list:
            act_type, payload = undo_list.pop()
            redo_list = self._page_redo_stack.setdefault(self._current_page_index, [])
            if act_type == "stroke":
                stroke = payload
                self._scene.remove_stroke(stroke.stroke_id)
                redo_list.append(("stroke", stroke))
                self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
            elif act_type == "stem":
                items = payload
                stem_items = self._page_stem_items.setdefault(self._current_page_index, [])
                for itm in items:
                    if itm in stem_items:
                        stem_items.remove(itm)
                    if itm.scene() is self._scene:
                        itm.setVisible(False)
                redo_list.append(("stem", items))
            elif act_type == "translate":
                items, dx, dy = payload
                self._scene.translate_items(items, -dx, -dy)
                redo_list.append(("translate", (items, dx, dy)))
                self._status.showMessage("Hoàn tác di chuyển", 1500)
            elif act_type == "delete":
                del_strokes, del_items = payload
                for s in del_strokes:
                    self._scene.commit_stroke(s)
                stem_items = self._page_stem_items.setdefault(self._current_page_index, [])
                for itm in del_items:
                    if itm.scene() is None:
                        self._scene.addItem(itm)
                    itm.setVisible(True)
                    if itm not in stem_items:
                        stem_items.append(itm)
                self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
                redo_list.append(("delete", (del_strokes, del_items)))
                self._status.showMessage("Hoàn tác xóa đối tượng", 1500)
            return

        if self._scene._strokes:
            last_id = list(self._scene._strokes.keys())[-1]
            stroke = self._scene._strokes.get(last_id)
            if stroke is not None:
                self._redo_stack.append(stroke)
            self._scene.remove_stroke(last_id)
            self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())

    def _on_redo(self) -> None:
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_undo()
        except Exception:
            pass
        redo_list = self._page_redo_stack.setdefault(self._current_page_index, [])
        if redo_list:
            act_type, payload = redo_list.pop()
            undo_list = self._page_undo_stack.setdefault(self._current_page_index, [])
            if act_type == "stroke":
                stroke = payload
                self._scene.commit_stroke(stroke)
                undo_list.append(("stroke", stroke))
                self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
            elif act_type == "stem":
                items = payload
                stem_items = self._page_stem_items.setdefault(self._current_page_index, [])
                for itm in items:
                    if itm.scene() is None:
                        self._scene.addItem(itm)
                    itm.setVisible(True)
                    if itm not in stem_items:
                        stem_items.append(itm)
                undo_list.append(("stem", items))
            elif act_type == "translate":
                items, dx, dy = payload
                self._scene.translate_items(items, dx, dy)
                undo_list.append(("translate", (items, dx, dy)))
                self._status.showMessage("Làm lại di chuyển", 1500)
            elif act_type == "delete":
                del_strokes, del_items = payload
                self._scene.delete_items(del_items + [s._path_item for s in del_strokes if s._path_item])
                self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())
                undo_list.append(("delete", (del_strokes, del_items)))
                self._status.showMessage("Làm lại xóa đối tượng", 1500)
            return

        if self._redo_stack:
            stroke = self._redo_stack.pop()
            self._scene.commit_stroke(stroke)
            self._page_strokes[self._current_page_index] = list(self._scene._strokes.values())

    def _set_teaching_mode(self, mode: str) -> None:
        self._teaching_mode = mode
        logger.info("Teaching mode changed to %s", mode)
        self._status_mode.setText(f"Mode: {mode}")
        if mode == "SPOTLIGHT":
            self._scene.enable_spotlight(True)
            self._scene.enable_loupe(False)
            self._scene.hide_eraser_reticle()
        elif mode == "LOUPE":
            self._scene.enable_loupe(True)
            self._scene.enable_spotlight(False)
            self._scene.hide_eraser_reticle()
        else:
            self._scene.enable_spotlight(False)
            self._scene.enable_loupe(False)

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

    @Slot()
    def _on_open_gesture_calibration(self) -> None:
        """Open interactive touchless gesture ergonomics & distance calibration wizard."""
        from src.ui.gesture_calibration_dialog import GestureCalibrationDialog
        dlg = GestureCalibrationDialog(self)
        dlg.exec()

    def _on_open_gesture_trainer(self) -> None:
        from src.ui.gesture_trainer_dialog import GestureTrainerDialog  # noqa: PLC0415
        dlg = GestureTrainerDialog(parent=self)
        if self._cam_worker is not None:
            self._cam_worker.is_trainer_active = True
            self._cam_worker.trainer_frame_ready.connect(
                dlg.update_frame, Qt.ConnectionType.QueuedConnection
            )
            dlg.calibration_updated.connect(self._cam_worker.calibrate_perspective)
        dlg.exec()
        if self._cam_worker is not None:
            self._cam_worker.is_trainer_active = False
            try:
                self._cam_worker.trainer_frame_ready.disconnect(dlg.update_frame)
            except Exception:
                pass
            try:
                dlg.calibration_updated.disconnect(self._cam_worker.calibrate_perspective)
            except Exception:
                pass

    def _on_clear_all(self) -> None:
        reply = QMessageBox.question(
            self, "Xóa tất cả",
            "Bạn có chắc muốn xóa tất cả các nét mực không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._scene.clear_all_strokes()
            self._page_strokes[self._current_page_index] = []

    def _on_desktop_bar_pen(self) -> None:
        self._set_teaching_mode("PEN")
        self._scene.hide_eraser_reticle()

    def _on_desktop_bar_eraser(self) -> None:
        self._set_teaching_mode("ERASER")
        self._scene.hide_laser()

    def _on_desktop_bar_laser(self) -> None:
        self._set_teaching_mode("LASER")
        self._scene.hide_eraser_reticle()

    def _on_desktop_bar_clear(self) -> None:
        self._scene.clear_page_annotations()
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_erase()
        except Exception:
            pass
        self._status.showMessage("Đã xóa toàn bộ nét vẽ trên màn hình Desktop", 2000)

    @Slot()
    def _on_toggle_gesture_guide(self) -> bool:
        """Toggle in-air gesture reference guide card overlay."""
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_click()
        except Exception:
            pass

        if self._gesture_guide_overlay is None:
            from src.ui.gesture_guide_overlay import GestureGuideOverlay
            self._gesture_guide_overlay = GestureGuideOverlay(self)
            self._gesture_guide_overlay.setGeometry(30, max(60, self.height() - 340), 380, 310)

        is_visible = self._gesture_guide_overlay.toggle_guide()
        self._status.showMessage(f"Thẻ hướng dẫn cử chỉ: {'Hiển thị' if is_visible else 'Đã ẩn'}", 2000)
        return is_visible

    @Slot()
    def _on_toggle_desktop_overlay(self) -> bool:
        """Toggle transparent desktop annotation overlay mode."""
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_click()
        except Exception:
            pass

        if not self._is_desktop_overlay:
            self._is_desktop_overlay = True
            self._saved_geometry = self.saveGeometry()

            # Make canvas transparent
            self._scene.set_desktop_overlay_mode(True)

            # Hide distracting side panels
            if hasattr(self, "_sidebar") and self._sidebar is not None:
                self._sidebar.setVisible(False)
            if hasattr(self, "_chat_panel") and self._chat_panel is not None:
                self._chat_panel.setVisible(False)

            # Make views transparent
            self._board_view.setStyleSheet("background: transparent; border: none;")
            self._board_view.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self._board_view.viewport().setStyleSheet("background: transparent;")

            # Set translucent window flags
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setWindowFlags(
                self.windowFlags()
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )

            # Show floating HUD bar
            if self._desktop_overlay_bar is None:
                from src.ui.desktop_overlay_bar import DesktopOverlayBar
                self._desktop_overlay_bar = DesktopOverlayBar(self)
                self._desktop_overlay_bar.pen_requested.connect(self._on_desktop_bar_pen)
                self._desktop_overlay_bar.eraser_requested.connect(self._on_desktop_bar_eraser)
                self._desktop_overlay_bar.laser_requested.connect(self._on_desktop_bar_laser)
                self._desktop_overlay_bar.spotlight_requested.connect(self._on_toggle_spotlight)
                self._desktop_overlay_bar.loupe_requested.connect(self._on_toggle_loupe)
                self._desktop_overlay_bar.timer_requested.connect(self._on_toggle_timer)
                self._desktop_overlay_bar.undo_requested.connect(self._on_undo)
                self._desktop_overlay_bar.clear_requested.connect(self._on_desktop_bar_clear)
                self._desktop_overlay_bar.exit_requested.connect(self._on_toggle_desktop_overlay)

            # Position bar at top center
            bar_w = 680
            bar_h = 44
            screen_geom = self.screen().geometry() if self.screen() else self.geometry()
            bar_x = screen_geom.x() + (screen_geom.width() - bar_w) // 2
            bar_y = screen_geom.y() + 20
            self._desktop_overlay_bar.setGeometry(bar_x, bar_y, bar_w, bar_h)
            self._desktop_overlay_bar.show()
            self._desktop_overlay_bar.raise_()

            self.showFullScreen()
            self._status.showMessage("🖥️ Chế độ vẽ đè Desktop kích hoạt (Nhấn F11 hoặc nút 'Bảng dạy học' để thoát)", 3500)
            return True
        else:
            self._is_desktop_overlay = False

            if self._desktop_overlay_bar is not None:
                self._desktop_overlay_bar.hide()

            # Restore canvas
            self._scene.set_desktop_overlay_mode(False)

            # Restore styles
            self._board_view.setStyleSheet("")
            self._board_view.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self._board_view.viewport().setStyleSheet("")

            # Restore window flags
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setWindowFlags(Qt.WindowType.Window)

            # Restore panels
            if hasattr(self, "_sidebar") and self._sidebar is not None:
                self._sidebar.setVisible(True)
            if hasattr(self, "_chat_panel") and self._chat_panel is not None:
                self._chat_panel.setVisible(True)

            if self._saved_geometry is not None:
                self.restoreGeometry(self._saved_geometry)
            self.showMaximized()
            self._status.showMessage("Đã trở về Bảng dạy học tương tác", 2500)
            return False

    def is_desktop_overlay_active(self) -> bool:
        """Return True if transparent desktop annotation mode is currently enabled."""
        return getattr(self, "_is_desktop_overlay", False)

    @Slot()
    def _on_zoom_in(self) -> float:
        """Zoom into whiteboard canvas."""
        if hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
            return self._zoom_ctrl.zoom_in()
        return 1.0

    @Slot()
    def _on_zoom_out(self) -> float:
        """Zoom out of whiteboard canvas."""
        if hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
            return self._zoom_ctrl.zoom_out()
        return 1.0

    @Slot()
    def _on_zoom_reset(self) -> None:
        """Reset whiteboard canvas magnification back to 100%."""
        if hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
            self._zoom_ctrl.reset_zoom()

    @Slot()
    def _on_toggle_pan(self) -> bool:
        """Toggle whiteboard canvas pan mode."""
        if hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
            active = self._zoom_ctrl.toggle_pan_mode()
            self._status.showMessage(f"Chế độ cuộn bảng (Pan): {'Bật' if active else 'Tắt'}", 2000)
            return active
        return False

    @Slot()
    def _on_toggle_minimap(self) -> bool:
        """Toggle canvas bird's eye view mini-map HUD."""
        if hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
            is_visible = self._zoom_ctrl.toggle_minimap()
            self._status.showMessage(
                f"Bản đồ toàn cảnh (Mini-Map): {'Hiển thị (Ctrl+Shift+V)' if is_visible else 'Đã ẩn'}",
                2000,
            )
            return is_visible
        return False

    @Slot()
    def _on_toggle_coordinate_hud(self) -> bool:
        """Toggle real-time Cartesian Oxy coordinate HUD on LessonBoard."""
        is_visible = self._scene.toggle_coordinate_hud()
        self._status.showMessage(
            f"Tọa độ Oxy (HUD): {'Hiển thị (Ctrl+Shift+X)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_coordinate_snap(self) -> bool:
        """Toggle magnetic grid snapping in Cartesian Oxy coordinate HUD."""
        is_snap = self._scene.toggle_coordinate_hud_snap()
        self._status.showMessage(
            f"Bắt dính lưới Oxy: {'BẬT (🧲 SNAP)' if is_snap else 'TẮT'}",
            2000,
        )
        return is_snap

    @Slot()
    def _on_toggle_angle_tool(self) -> bool:
        """Toggle interactive dynamic angle measurement tool on LessonBoard."""
        is_visible = self._scene.toggle_angle_tool()
        self._status.showMessage(
            f"Thước đo góc động: {'Hiển thị (Ctrl+Shift+A)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_triangle_tool(self) -> bool:
        """Toggle interactive geometric triangle theorem solver on LessonBoard."""
        is_visible = self._scene.toggle_triangle_tool()
        self._status.showMessage(
            f"Tam giác hình học động: {'Hiển thị (Ctrl+Shift+T)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    @Slot()

    def _on_toggle_transformer_attention(self) -> bool:

        """Toggle interactive Transformer multi-head self-attention visualizer on LessonBoard."""

        is_visible = self._scene.toggle_transformer_attention()

        self._status.showMessage(

            f"Transformer Self-Attention: {'Hiển thị (Ctrl+Shift+A)' if is_visible else 'Đã ẩn'}",
            2000,

        )

        return is_visible

    @Slot()

    def _on_toggle_special_relativity(self) -> bool:

        """Toggle interactive Special Relativity & Minkowski spacetime visualizer on LessonBoard."""

        is_visible = self._scene.toggle_special_relativity()

        self._status.showMessage(

            f"Thuyết Tương đối Hẹp & Minkowski: {'Hiển thị (Ctrl+Shift+R)' if is_visible else 'Đã ẩn'}",
            2000,

        )

        return is_visible

    @Slot()
    def _on_toggle_quantum_tunneling(self) -> bool:
        """Toggle interactive Quantum Mechanics & Tunneling simulator on LessonBoard."""
        is_visible = self._scene.toggle_quantum_tunneling()
        self._status.showMessage(
            f"Cơ học Lượng tử & Xuyên hầm: {'Hiển thị (Ctrl+Shift+Q)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_coriolis_earth(self) -> bool:
        """Toggle interactive Coriolis Effect & Geostrophic Flow visualizer on LessonBoard."""
        is_visible = self._scene.toggle_coriolis_earth()
        self._status.showMessage(
            f"Hiệu ứng Coriolis & Khí quyển: {'Hiển thị (Ctrl+Shift+C)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_acoustic_sonar(self) -> bool:
        """Toggle interactive Acoustic Doppler & Sonar visualizer on LessonBoard."""
        is_visible = self._scene.toggle_acoustic_sonar()
        self._status.showMessage(
            f"Sóng âm Doppler & Sonar: {'Hiển thị (Ctrl+Shift+S)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_cloud_chamber(self) -> bool:
        """Toggle interactive Wilson Cloud Chamber & Particle Tracks visualizer on LessonBoard."""
        is_visible = self._scene.toggle_cloud_chamber()
        self._status.showMessage(
            f"Buồng sương Wilson & Hạt nhân: {'Hiển thị (Ctrl+Shift+N)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_orbital_mechanics(self) -> bool:
        """Toggle interactive Hohmann Transfer & Astrodynamics visualizer on LessonBoard."""
        is_visible = self._scene.toggle_orbital_mechanics()
        self._status.showMessage(
            f"Cơ học Quỹ đạo & Hohmann: {'Hiển thị (Ctrl+Shift+L)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_heat_transfer(self) -> bool:
        """Toggle interactive 2D Heat Transfer & Convection simulator on LessonBoard."""
        is_visible = self._scene.toggle_heat_transfer()
        self._status.showMessage(
            f"Truyền nhiệt Fourier 2D: {'Hiển thị (Ctrl+Shift+Z)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_quantum_diffraction(self) -> bool:
        """Toggle interactive Quantum Double-Slit & de Broglie matter wave visualizer on LessonBoard."""
        is_visible = self._scene.toggle_quantum_diffraction()
        self._status.showMessage(
            f"Giao thoa Lượng tử Khe kép: {'Hiển thị (Ctrl+Shift+1)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_laval_nozzle(self) -> bool:
        """Toggle interactive de Laval Nozzle & Supersonic shock diamonds on LessonBoard."""
        is_visible = self._scene.toggle_laval_nozzle()
        self._status.showMessage(
            f"Ống phun de Laval: {'Hiển thị (Ctrl+Shift+2)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_ramachandran_plot(self) -> bool:
        """Toggle interactive Ramachandran Plot & Protein Folding on LessonBoard."""
        is_visible = self._scene.toggle_ramachandran_plot()
        self._status.showMessage(
            f"Giản đồ Ramachandran: {'Hiển thị (Ctrl+Shift+3)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_double_pendulum(self) -> bool:
        """Toggle interactive Double Pendulum & Chaos Theory on LessonBoard."""
        is_visible = self._scene.toggle_double_pendulum()
        self._status.showMessage(
            f"Con lắc kép & Hỗn loạn: {'Hiển thị (Ctrl+Shift+4)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_superconductivity(self) -> bool:
        """Toggle interactive Superconductivity & Quantum Levitation on LessonBoard."""
        is_visible = self._scene.toggle_superconductivity()
        self._status.showMessage(
            f"Siêu dẫn & Bay lượng tử: {'Hiển thị (Ctrl+Shift+5)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible


    def _on_toggle_function_plotter(self) -> bool:
        """Toggle interactive function graph plotter & tangent inspector on LessonBoard."""
        is_visible = self._scene.toggle_function_plotter()
        self._status.showMessage(
            f"Đồ thị f(x) & Tiếp tuyến: {'Hiển thị (Ctrl+Shift+F)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_inclined_plane(self) -> bool:
        """Toggle interactive physics inclined plane vector decomposer on LessonBoard."""
        is_visible = self._scene.toggle_inclined_plane()
        self._status.showMessage(
            f"Hệ lực mặt phẳng nghiêng: {'Hiển thị (Ctrl+Shift+I)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_optics_refraction(self) -> bool:
        """Toggle interactive physics optics Snell's law refraction simulator on LessonBoard."""
        is_visible = self._scene.toggle_optics_refraction()
        self._status.showMessage(
            f"Khúc xạ ánh sáng (Snell's Law): {'Hiển thị (Ctrl+Shift+O)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_standing_wave(self) -> bool:
        """Toggle interactive physics standing wave simulator on LessonBoard."""
        is_visible = self._scene.toggle_standing_wave()
        self._status.showMessage(
            f"Sóng cơ & Sóng dừng: {'Hiển thị (Ctrl+Shift+W)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_vsepr_geometry(self) -> bool:
        """Toggle interactive 3D VSEPR molecular geometry simulator on LessonBoard."""
        is_visible = self._scene.toggle_vsepr_geometry()
        self._status.showMessage(
            f"Hình học phân tử 3D VSEPR: {'Hiển thị (Ctrl+Shift+Y)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_dna_replication(self) -> bool:
        """Toggle interactive biology DNA replication & base pairing simulator on LessonBoard."""
        is_visible = self._scene.toggle_dna_replication()
        self._status.showMessage(
            f"Tự nhân đôi DNA (Nguyên tắc bổ sung): {'Hiển thị (Ctrl+Shift+D)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_kepler_orbit(self) -> bool:
        """Toggle interactive astronomy Kepler orbit & planetary motion simulator on LessonBoard."""
        is_visible = self._scene.toggle_kepler_orbit()
        self._status.showMessage(
            f"Quỹ đạo hành tinh Kepler: {'Hiển thị (Ctrl+Shift+K)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_graph_pathfinding(self) -> bool:
        """Toggle interactive graph theory & Dijkstra / A* pathfinding simulator on LessonBoard."""
        is_visible = self._scene.toggle_graph_pathfinding()
        self._status.showMessage(
            f"Lý thuyết đồ thị & Đường đi ngắn nhất (Dijkstra/A*): {'Hiển thị (Ctrl+Shift+P)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_electric_field(self) -> bool:
        """Toggle interactive physics electric field lines & Coulomb simulator on LessonBoard."""
        is_visible = self._scene.toggle_electric_field()
        self._status.showMessage(
            f"Điện trường & Định luật Coulomb: {'Hiển thị (Ctrl+Shift+E)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_lorentz_force(self) -> bool:
        """Toggle interactive physics magnetism & Lorentz force simulator on LessonBoard."""
        is_visible = self._scene.toggle_lorentz_force()
        self._status.showMessage(
            f"Lực Lorentz & Từ trường: {'Hiển thị (Ctrl+Shift+B)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_trophic_web(self) -> bool:
        """Toggle interactive biology ecosystem trophic food web simulator on LessonBoard."""
        is_visible = self._scene.toggle_trophic_web()
        self._status.showMessage(
            f"Lưới thức ăn & Hệ sinh thái: {'Hiển thị (Ctrl+Shift+U)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_gas_laws(self) -> bool:
        """Toggle interactive ideal gas laws & kinetic theory simulator on LessonBoard."""
        is_visible = self._scene.toggle_gas_laws()
        self._status.showMessage(
            f"Định luật khí lý tưởng: {'Hiển thị (Ctrl+Shift+J)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_titration(self) -> bool:
        """Toggle interactive acid-base titration & pH buffer simulator on LessonBoard."""
        is_visible = self._scene.toggle_titration()
        self._status.showMessage(
            f"Chuẩn độ Axit - Bazơ & pH: {'Hiển thị (Ctrl+Shift+Q)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_harmonic_oscillator(self) -> bool:
        """Toggle interactive harmonic oscillator & resonance simulator on LessonBoard."""
        is_visible = self._scene.toggle_harmonic_oscillator()
        self._status.showMessage(
            f"Dao động điều hòa & Cộng hưởng: {'Hiển thị (Ctrl+Shift+H)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_radioactive_decay(self) -> bool:
        """Toggle interactive radioactive decay simulator on LessonBoard."""
        is_visible = self._scene.toggle_radioactive_decay()
        self._status.showMessage(
            f"Định luật phóng xạ & Chuỗi phân rã: {'Hiển thị (Ctrl+Alt+R)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_chemical_kinetics(self) -> bool:
        """Toggle interactive chemical kinetics & equilibrium simulator on LessonBoard."""
        is_visible = self._scene.toggle_chemical_kinetics()
        self._status.showMessage(
            f"Động hóa học & Cân bằng: {'Hiển thị (Ctrl+Alt+K)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_cryptography(self) -> bool:
        """Toggle interactive cryptography & modular arithmetic simulator on LessonBoard."""
        is_visible = self._scene.toggle_cryptography()
        self._status.showMessage(
            f"Mật mã học RSA: {'Hiển thị (Ctrl+Alt+C)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_gravitational_orbits(self) -> bool:
        """Toggle interactive gravitational orbits & Lagrange points simulator on LessonBoard."""
        is_visible = self._scene.toggle_gravitational_orbits()
        self._status.showMessage(
            f"Quỹ đạo hấp dẫn & Lagrange: {'Hiển thị (Ctrl+Alt+G)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_spacetime_relativity(self) -> bool:
        """Toggle interactive Special Relativity Minkowski Spacetime simulator on LessonBoard."""
        is_visible = self._scene.toggle_spacetime_relativity()
        self._status.showMessage(
            f"Không-thời gian Minkowski: {'Hiển thị (Ctrl+Alt+M)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_young_interference(self) -> bool:
        """Toggle interactive Wave Optics & Young's Double-Slit simulator on LessonBoard."""
        is_visible = self._scene.toggle_young_interference()
        self._status.showMessage(
            f"Giao thoa sóng khe Young: {'Hiển thị (Ctrl+Alt+Y)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_rlc_circuit(self) -> bool:
        """Toggle interactive AC Series RLC Circuit & Phasor simulator on LessonBoard."""
        is_visible = self._scene.toggle_rlc_circuit()
        self._status.showMessage(
            f"Mạch RLC xoay chiều & Fresnel: {'Hiển thị (Ctrl+Alt+E)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_thin_lens(self) -> bool:
        """Toggle interactive Geometric Optics Thin Lens & Optical Bench simulator on LessonBoard."""
        is_visible = self._scene.toggle_thin_lens()
        self._status.showMessage(
            f"Thấu kính mỏng & Băng quang học: {'Hiển thị (Ctrl+Alt+O)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_photoelectric_effect(self) -> bool:
        """Toggle interactive Modern Physics Photoelectric Effect simulator on LessonBoard."""
        is_visible = self._scene.toggle_photoelectric_effect()
        self._status.showMessage(
            f"Hiệu ứng Quang điện & Thuyết Lượng tử: {'Hiển thị (Ctrl+Alt+P)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_carnot_engine(self) -> bool:
        """Toggle interactive Thermodynamics Carnot Heat Engine simulator on LessonBoard."""
        is_visible = self._scene.toggle_carnot_engine()
        self._status.showMessage(
            f"Động cơ nhiệt Carnot (PV - TS): {'Hiển thị (Ctrl+Alt+T)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_bohr_atom(self) -> bool:
        """Toggle interactive Quantum Physics Bohr Atom simulator on LessonBoard."""
        is_visible = self._scene.toggle_bohr_atom()
        self._status.showMessage(
            f"Mô hình Nguyên tử Bohr: {'Hiển thị (Ctrl+Alt+B)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_automata(self) -> bool:
        """Toggle interactive Computer Science Finite State Machine (Automata) simulator on LessonBoard."""
        is_visible = self._scene.toggle_automata()
        self._status.showMessage(
            f"Máy trạng thái hữu hạn & Automata: {'Hiển thị (Ctrl+Alt+A)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_sorting(self) -> bool:
        """Toggle interactive Computer Science Sorting Algorithm visualizer on LessonBoard."""
        is_visible = self._scene.toggle_sorting()
        self._status.showMessage(
            f"Thuật toán sắp xếp & Big-O: {'Hiển thị (Ctrl+Alt+S)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_punnett_genetics(self) -> bool:
        """Toggle interactive Biology Mendel Genetics & Punnett Square visualizer on LessonBoard."""
        is_visible = self._scene.toggle_punnett_genetics()
        self._status.showMessage(
            f"Di truyền Mendel & Khung Punnett: {'Hiển thị (Ctrl+Alt+G)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_organic_chemistry(self) -> bool:
        """Toggle interactive Organic Chemistry Isomer simulator on LessonBoard."""
        is_visible = self._scene.toggle_organic_chemistry()
        self._status.showMessage(
            f"Đồng phân Hóa học Hữu cơ: {'Hiển thị (Ctrl+Alt+H)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_fluid_venturi(self) -> bool:
        """Toggle interactive Fluid Dynamics Venturi simulator on LessonBoard."""
        is_visible = self._scene.toggle_fluid_venturi()
        self._status.showMessage(
            f"Cơ học chất lưu Bernoulli: {'Hiển thị (Ctrl+Alt+F)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_binary_tree(self) -> bool:
        """Toggle interactive Binary Tree & Huffman simulator on LessonBoard."""
        is_visible = self._scene.toggle_binary_tree()
        self._status.showMessage(
            f"Cây nhị phân & Huffman: {'Hiển thị (Ctrl+Alt+U)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_electrochemistry(self) -> bool:
        """Toggle interactive Electrochemistry & Electrolysis simulator on LessonBoard."""
        is_visible = self._scene.toggle_electrochemistry()
        self._status.showMessage(
            f"Điện hóa & Điện phân Nernst: {'Hiển thị (Ctrl+Alt+E)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_circular_motion(self) -> bool:
        """Toggle interactive Circular Motion simulator on LessonBoard."""
        is_visible = self._scene.toggle_circular_motion()
        self._status.showMessage(
            f"Chuyển động tròn đều: {'Hiển thị (Ctrl+Alt+C)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_ph_titration(self) -> bool:
        """Toggle interactive pH Titration simulator on LessonBoard."""
        is_visible = self._scene.toggle_ph_titration()
        self._status.showMessage(
            f"Chuẩn độ Axit - Bazơ & Đường cong pH: {'Hiển thị (Ctrl+Alt+I)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_electromagnetic_induction(self) -> bool:
        """Toggle interactive Electromagnetic Induction simulator on LessonBoard."""
        is_visible = self._scene.toggle_electromagnetic_induction()
        self._status.showMessage(
            f"Cảm ứng điện từ & Định luật Faraday: {'Hiển thị (Ctrl+Alt+M)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_gcms(self) -> bool:
        """Toggle interactive GC-MS simulator on LessonBoard."""
        is_visible = self._scene.toggle_gcms()
        status_msg = "Sắc ký khí & Khối phổ GC-MS: " + ("Hiển thị (Ctrl+Alt+Y)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_photosynthesis(self) -> bool:
        """Toggle interactive Photosynthesis simulator on LessonBoard."""
        is_visible = self._scene.toggle_photosynthesis()
        status_msg = "Quang hợp & Chu trình Calvin: " + ("Hiển thị (Ctrl+Alt+Q)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_projectile_motion(self) -> bool:
        """Toggle interactive Projectile Motion simulator on LessonBoard."""
        is_visible = self._scene.toggle_projectile_motion()
        status_msg = "Chuyển động ném & Động học 2D: " + ("Hiển thị (Ctrl+Alt+K)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_periodic_table(self) -> bool:
        """Toggle interactive Periodic Table simulator on LessonBoard."""
        is_visible = self._scene.toggle_periodic_table()
        status_msg = "Bảng tuần hoàn nguyên tố: " + ("Hiển thị (Ctrl+Alt+J)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_cellular_respiration(self) -> bool:
        """Toggle interactive Cellular Respiration & ETC simulator on LessonBoard."""
        is_visible = self._scene.toggle_cellular_respiration()
        status_msg = "Hô hấp tế bào & ETC: " + ("Hiển thị (Ctrl+Alt+Z)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_protein_folding(self) -> bool:
        """Toggle interactive Protein Folding & Ramachandran simulator on LessonBoard."""
        is_visible = self._scene.toggle_protein_folding()
        status_msg = "Cấu trúc Protein & Ramachandran: " + ("Hiển thị (Ctrl+Alt+X)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_mlp_backprop(self) -> bool:
        """Toggle interactive MLP Backprop simulator on LessonBoard."""
        is_visible = self._scene.toggle_mlp_backprop()
        status_msg = "Mạng nơ-ron MLP & Backpropagation: " + ("Hiển thị (Ctrl+Alt+L)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_max_flow(self) -> bool:
        """Toggle interactive Max Flow Network simulator on LessonBoard."""
        is_visible = self._scene.toggle_max_flow()
        status_msg = "Luồng cực đại & Min-Cut: " + ("Hiển thị (Ctrl+Alt+1)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_oxyz_geometry(self) -> bool:
        """Toggle interactive 3D Oxyz Geometry simulator on LessonBoard."""
        is_visible = self._scene.toggle_oxyz_geometry()
        status_msg = "Hình học không gian Oxyz: " + ("Hiển thị (Ctrl+Alt+2)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_semiconductor_diode(self) -> bool:
        """Toggle interactive Semiconductor P-N Diode simulator on LessonBoard."""
        is_visible = self._scene.toggle_semiconductor_diode()
        status_msg = "Bán dẫn P-N & Điốt Shockley: " + ("Hiển thị (Ctrl+Alt+3)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_nuclear_reactions(self) -> bool:
        """Toggle interactive Nuclear Physics & Reactions simulator on LessonBoard."""
        is_visible = self._scene.toggle_nuclear_reactions()
        status_msg = "Vật lý Hạt nhân & Năng lượng liên kết: " + ("Hiển thị (Ctrl+Alt+4)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_doppler_effect(self) -> bool:
        """Toggle interactive Doppler Effect & Mach Cone simulator on LessonBoard."""
        is_visible = self._scene.toggle_doppler_effect()
        status_msg = "Hiệu ứng Doppler & Nón sóng Mach: " + ("Hiển thị (Ctrl+Alt+5)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_red_black_tree(self) -> bool:
        """Toggle interactive Red-Black Tree simulator on LessonBoard."""
        is_visible = self._scene.toggle_red_black_tree()
        status_msg = "Cây tự cân bằng Đỏ-Đen: " + ("Hiển thị (Ctrl+Alt+6)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_millikan_oil_drop(self) -> bool:
        """Toggle interactive Millikan Oil Drop simulator on LessonBoard."""
        is_visible = self._scene.toggle_millikan_oil_drop()
        status_msg = "Giọt dầu Millikan & Điện tích nguyên tố: " + ("Hiển thị (Ctrl+Alt+7)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_quantum_well(self) -> bool:
        """Toggle interactive Quantum Well & Particle in a Box simulator on LessonBoard."""
        is_visible = self._scene.toggle_quantum_well()
        status_msg = "Giếng lượng tử & Sóng Schrödinger: " + ("Hiển thị (Ctrl+Alt+8)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_hohmann_transfer(self) -> bool:
        """Toggle interactive Hohmann Transfer Orbit simulator on LessonBoard."""
        is_visible = self._scene.toggle_hohmann_transfer()
        status_msg = "Cơ học Thiên thể & Chuyển quỹ đạo Hohmann: " + ("Hiển thị (Ctrl+Alt+9)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_photoelectric_effect(self) -> bool:
        """Toggle interactive Photoelectric Effect simulator on LessonBoard."""
        is_visible = self._scene.toggle_photoelectric_effect()
        status_msg = "Hiện tượng Quang điện Ngoài: " + ("Hiển thị (Ctrl+Alt+0)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_punnett_genetics(self) -> bool:
        """Toggle interactive Punnett Genetics simulator on LessonBoard."""
        is_visible = self._scene.toggle_punnett_genetics()
        status_msg = "Di truyền Mendel & Khung Punnett: " + ("Hiển thị (Ctrl+Shift+G)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_reaction_kinetics(self) -> bool:
        """Toggle interactive Reaction Kinetics simulator on LessonBoard."""
        is_visible = self._scene.toggle_reaction_kinetics()
        status_msg = "Động Hóa học & Định luật Arrhenius: " + ("Hiển thị (Ctrl+Shift+K)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_calculus_taylor(self) -> bool:
        """Toggle interactive Riemann Calculus & Taylor simulator on LessonBoard."""
        is_visible = self._scene.toggle_calculus_taylor()
        status_msg = "Tích phân Riemann & Chuỗi Taylor: " + ("Hiển thị (Ctrl+Shift+R)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_michelson_interferometer(self) -> bool:
        """Toggle interactive Michelson Interferometer & LIGO simulator on LessonBoard."""
        is_visible = self._scene.toggle_michelson_interferometer()
        status_msg = "Giao thoa kế Michelson & LIGO: " + ("Hiển thị (Ctrl+Shift+M)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_huffman_coding(self) -> bool:
        """Toggle interactive Huffman Coding & Shannon Entropy simulator on LessonBoard."""
        is_visible = self._scene.toggle_huffman_coding()
        status_msg = "Mã hóa Huffman & Entropy Shannon: " + ("Hiển thị (Ctrl+Shift+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_action_potential(self) -> bool:
        """Toggle interactive Hodgkin-Huxley Action Potential simulator on LessonBoard."""
        is_visible = self._scene.toggle_action_potential()
        status_msg = "Điện thế Hoạt động Hodgkin-Huxley: " + ("Hiển thị (Ctrl+Shift+P)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_cyclotron(self) -> bool:
        """Toggle interactive Cyclotron particle accelerator simulator on LessonBoard."""
        is_visible = self._scene.toggle_cyclotron()
        status_msg = "Máy gia tốc Cyclotron & Synchro: " + ("Hiển thị (Ctrl+Alt+C)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_aho_corasick(self) -> bool:
        """Toggle interactive Aho-Corasick multi-pattern automaton simulator on LessonBoard."""
        is_visible = self._scene.toggle_aho_corasick()
        status_msg = "Mô phỏng Otomat Aho-Corasick: " + ("Hiển thị (Ctrl+Alt+A)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_phase_diagram(self) -> bool:
        """Toggle interactive Thermodynamics P-T phase diagram simulator on LessonBoard."""
        is_visible = self._scene.toggle_phase_diagram()
        status_msg = "Giản đồ Pha P-T & Clausius-Clapeyron: " + ("Hiển thị (Ctrl+Alt+P)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_em_wave(self) -> bool:
        """Toggle interactive Maxwell's electromagnetic wave simulator on LessonBoard."""
        is_visible = self._scene.toggle_em_wave()
        status_msg = "Sóng Điện từ Maxwell & Vectơ Poynting: " + ("Hiển thị (Ctrl+Alt+E)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_hardy_weinberg(self) -> bool:
        """Toggle interactive Hardy-Weinberg population genetics simulator on LessonBoard."""
        is_visible = self._scene.toggle_hardy_weinberg()
        status_msg = "Cân bằng Hardy-Weinberg: " + ("Hiển thị (Ctrl+Alt+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_fft(self) -> bool:
        """Toggle interactive Cooley-Tukey Fast Fourier Transform (FFT) simulator on LessonBoard."""
        is_visible = self._scene.toggle_fft()
        status_msg = "Phân tích Phổ FFT: " + ("Hiển thị (Ctrl+Alt+F)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_fabry_perot(self) -> bool:
        """Toggle interactive Fabry-Perot Interferometer & Cavity Resonator simulator on LessonBoard."""
        is_visible = self._scene.toggle_fabry_perot()
        status_msg = "Giao thoa kế Fabry-Pérot: " + ("Hiển thị (Ctrl+Alt+O)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_hplc(self) -> bool:
        """Toggle interactive HPLC & Van Deemter simulator on LessonBoard."""
        is_visible = self._scene.toggle_hplc()
        status_msg = "Sắc ký Lỏng HPLC: " + ("Hiển thị (Ctrl+Alt+L)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_photoelectric(self) -> bool:
        """Toggle interactive Photoelectric Effect simulator on LessonBoard."""
        is_visible = self._scene.toggle_photoelectric()
        status_msg = "Hiệu ứng Quang điện Einstein: " + ("Hiển thị (Ctrl+Alt+Q)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_pcr(self) -> bool:
        """Toggle interactive PCR & Gel Electrophoresis simulator on LessonBoard."""
        is_visible = self._scene.toggle_pcr()
        status_msg = "Phản ứng Chuỗi PCR & Điện di: " + ("Hiển thị (Ctrl+Alt+R)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_thin_lens(self) -> bool:
        """Toggle interactive Thin Lens Systems & Geometrical Optics simulator on LessonBoard."""
        is_visible = self._scene.toggle_thin_lens()
        status_msg = "Hệ Thấu kính Mỏng: " + ("Hiển thị (Ctrl+Alt+T)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_bragg_diffraction(self) -> bool:
        """Toggle interactive Bragg's Law & XRD Crystallography simulator on LessonBoard."""
        is_visible = self._scene.toggle_bragg_diffraction()
        status_msg = "Nhiễu xạ Tia X Bragg XRD: " + ("Hiển thị (Ctrl+Alt+Shift+B)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_buffer_system(self) -> bool:
        """Toggle interactive Acid-Base Buffer Systems & Van Slyke Capacity simulator on LessonBoard."""
        is_visible = self._scene.toggle_buffer_system()
        status_msg = "Hệ Đệm Axit-Bazơ & Van Slyke: " + ("Hiển thị (Ctrl+Alt+Shift+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_stern_gerlach(self) -> bool:
        """Toggle interactive Stern-Gerlach Experiment & Spin 1/2 simulator on LessonBoard."""
        is_visible = self._scene.toggle_stern_gerlach()
        status_msg = "Thí nghiệm Stern-Gerlach & Spin 1/2: " + ("Hiển thị (Ctrl+Alt+Shift+S)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_enzyme_kinetics(self) -> bool:
        """Toggle interactive Michaelis-Menten Enzyme Kinetics & Lineweaver-Burk simulator on LessonBoard."""
        is_visible = self._scene.toggle_enzyme_kinetics()
        status_msg = "Động học Enzyme Michaelis-Menten: " + ("Hiển thị (Ctrl+Alt+Shift+E)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_poiseuille_flow(self) -> bool:
        """Toggle interactive Hagen-Poiseuille Viscous Tube Flow & Reynolds simulator on LessonBoard."""
        is_visible = self._scene.toggle_poiseuille_flow()
        status_msg = "Dòng Chảy Nhớt Hagen-Poiseuille: " + ("Hiển thị (Ctrl+Alt+Shift+P)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_cosmic_expansion(self) -> bool:
        """Toggle interactive Hubble-Lemaître Law & Cosmic Expansion simulator on LessonBoard."""
        is_visible = self._scene.toggle_cosmic_expansion()
        status_msg = "Định Luật Hubble & Vũ Trụ Giãn Nở: " + ("Hiển thị (Ctrl+Alt+Shift+U)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_seismic_waves(self) -> bool:
        """Toggle interactive Seismic Waves & Epicenter Triangulation simulator on LessonBoard."""
        is_visible = self._scene.toggle_seismic_waves()
        status_msg = "Sóng Địa Chấn & Tâm Chấn: " + ("Hiển thị (Ctrl+Alt+Shift+Q)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_mass_spectrometer(self) -> bool:
        """Toggle interactive Mass Spectrometer & Wien Filter simulator on LessonBoard."""
        is_visible = self._scene.toggle_mass_spectrometer()
        status_msg = "Máy Khối Phổ & Bộ Lọc Wien: " + ("Hiển thị (Ctrl+Alt+Shift+M)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_spectrophotometry(self) -> bool:
        """Toggle interactive Spectrophotometry & Beer-Lambert Law simulator on LessonBoard."""
        is_visible = self._scene.toggle_spectrophotometry()
        status_msg = "Quang Phổ Hấp Thụ Beer-Lambert: " + ("Hiển thị (Ctrl+Alt+Shift+L)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_nuclear_binding_energy(self) -> bool:
        """Toggle interactive Nuclear Binding Energy & SEMF simulator on LessonBoard."""
        is_visible = self._scene.toggle_nuclear_binding_energy()
        status_msg = "Năng Lượng Liên Kết Hạt Nhân: " + ("Hiển thị (Ctrl+Alt+Shift+B)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_compton_scattering(self) -> bool:
        """Toggle interactive Compton Scattering & Relativistic Collision simulator on LessonBoard."""
        is_visible = self._scene.toggle_compton_scattering()
        status_msg = "Tán Xạ Compton: " + ("Hiển thị (Ctrl+Alt+Shift+Q)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_gravitational_lensing(self) -> bool:
        """Toggle interactive Gravitational Lensing & General Relativity simulator on LessonBoard."""
        is_visible = self._scene.toggle_gravitational_lensing()
        status_msg = "Thấu Kính Hấp Dẫn & Nhẫn Einstein: " + ("Hiển thị (Ctrl+Alt+Shift+G)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_chemical_equilibrium(self) -> bool:
        """Toggle interactive Chemical Equilibrium & Le Chatelier simulator on LessonBoard."""
        is_visible = self._scene.toggle_chemical_equilibrium()
        status_msg = "Cân Bằng Hóa Học & Le Chatelier: " + ("Hiển thị (Ctrl+Alt+Shift+E)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_hodgkin_huxley(self) -> bool:
        """Toggle interactive Hodgkin-Huxley Neuron ODE simulator on LessonBoard."""
        is_visible = self._scene.toggle_hodgkin_huxley()
        status_msg = "Điện Sinh Lý Nơ-ron Hodgkin-Huxley: " + ("Hiển thị (Ctrl+Alt+Shift+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_mst_visualizer(self) -> bool:
        """Toggle interactive Minimum Spanning Tree simulator on LessonBoard."""
        is_visible = self._scene.toggle_mst_visualizer()
        status_msg = "Cây Khung Nhỏ Nhất (MST): " + ("Hiển thị (Ctrl+Alt+Shift+M)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_hall_effect(self) -> bool:
        """Toggle interactive Hall Effect & Semiconductor Lorentz Drift simulator on LessonBoard."""
        is_visible = self._scene.toggle_hall_effect()
        status_msg = "Hiệu Ứng Hall & Bán Dẫn: " + ("Hiển thị (Ctrl+Alt+Shift+F)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_radiation_shielding(self) -> bool:
        """Toggle interactive Nuclear Radiation Shielding simulator on LessonBoard."""
        is_visible = self._scene.toggle_radiation_shielding()
        status_msg = "Che Chắn Phóng Xạ & HVL: " + ("Hiển thị (Ctrl+Alt+Shift+R)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_maxwell_boltzmann(self) -> bool:
        """Toggle interactive Maxwell-Boltzmann Distribution simulator on LessonBoard."""
        is_visible = self._scene.toggle_maxwell_boltzmann()
        status_msg = "Phân Bố Vận Tốc Maxwell-Boltzmann: " + ("Hiển thị (Ctrl+Alt+Shift+X)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_molecular_orbital(self) -> bool:
        """Toggle interactive Molecular Orbital & Hybridization simulator on LessonBoard."""
        is_visible = self._scene.toggle_molecular_orbital()
        status_msg = "Obitan Phân Tử MO-LCAO: " + ("Hiển thị (Ctrl+Alt+Shift+O)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_rlc_resonance(self) -> bool:
        """Toggle interactive RLC AC Resonance & Phasor simulator on LessonBoard."""
        is_visible = self._scene.toggle_rlc_resonance()
        status_msg = "Cộng Hưởng Điện RLC & Phasor: " + ("Hiển thị (Ctrl+Alt+Shift+C)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_stress_strain(self) -> bool:
        """Toggle interactive Stress-Strain Curve & Tensile Test simulator on LessonBoard."""
        is_visible = self._scene.toggle_stress_strain()
        status_msg = "Đường Cong Ứng Suất - Biến Dạng: " + ("Hiển thị (Ctrl+Alt+Shift+T)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_kepler_orbits(self) -> bool:
        """Toggle interactive Kepler Gravitational Orbits simulator on LessonBoard."""
        is_visible = self._scene.toggle_kepler_orbits()
        status_msg = "Quỹ Đạo Hấp Dẫn & Định Luật Kepler: " + ("Hiển thị (Ctrl+Alt+Shift+K)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_logic_circuits(self) -> bool:
        """Toggle interactive Digital Logic Circuits simulator on LessonBoard."""
        is_visible = self._scene.toggle_logic_circuits()
        status_msg = "Điện Tử Số & Cổng Logic: " + ("Hiển thị (Ctrl+Alt+Shift+G)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible







    @Slot()
    def _on_toggle_airfoil_aerodynamics(self) -> bool:
        """Toggle interactive Airfoil Aerodynamics simulator on LessonBoard."""
        is_visible = self._scene.toggle_airfoil_aerodynamics()
        status_msg = "Khí động học cánh NACA: " + ("Hiển thị (Ctrl+Alt+V)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_cnn_visualizer(self) -> bool:
        """Toggle interactive CNN 2D Convolution simulator on LessonBoard."""
        is_visible = self._scene.toggle_cnn_visualizer()
        status_msg = "Mạng nơ-ron CNN & Tích chập 2D: " + ("Hiển thị (Ctrl+Alt+N)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_dynamic_programming(self) -> bool:
        """Toggle interactive Dynamic Programming simulator on LessonBoard."""
        is_visible = self._scene.toggle_dynamic_programming()
        self._status.showMessage(
            f"Quy hoạch động & Bảng trạng thái: {'Hiển thị (Ctrl+Alt+D)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_cycle_laser_color(self) -> None:
        """Cycle laser pointer hue presets (Ruby -> Emerald -> Cyan -> Amber -> Violet)."""
        hex_c, name, emoji = self._scene.cycle_laser_color()
        self._status.showMessage(f"Tia Laser: {emoji} {name} (Phím Ctrl+L)", 2000)

    @Slot(float)
    def _on_zoom_changed(self, zoom: float) -> None:
        """Handle zoom scale update."""
        pct = int(round(zoom * 100))
        self._status.showMessage(f"Độ thu phóng bảng vẽ: {pct}%", 1500)

    @Slot()
    def _on_toggle_timer(self) -> None:
        """Toggle presentation countdown timer overlay on LessonBoard."""
        is_visible = self._scene.toggle_timer()
        self._status.showMessage(f"Đồng hồ đếm giờ: {'Hiển thị' if is_visible else 'Đã ẩn'}", 2000)

    @Slot()
    def _on_toggle_metronome(self) -> None:
        """Toggle classroom metronome overlay on LessonBoard."""
        is_visible = self._scene.toggle_metronome()
        self._status.showMessage(
            f"Máy đếm nhịp (Metronome): {'Hiển thị (Ctrl+M)' if is_visible else 'Đã ẩn'}", 2000
        )

    @Slot()
    def _on_toggle_loupe(self) -> None:
        """Toggle virtual loupe magnifying glass mode."""
        if self._teaching_mode == "LOUPE":
            self._set_teaching_mode("PEN")
            self._status.showMessage("Đã tắt kính lúp", 2000)
        else:
            self._set_teaching_mode("LOUPE")
            self._status.showMessage("Đã bật kính lúp phóng đại 2.0x (Phím M)", 2000)

    @Slot()
    def _on_toggle_spotlight(self) -> None:
        """Toggle presentation spotlight mode."""
        if self._teaching_mode == "SPOTLIGHT":
            self._set_teaching_mode("PEN")
            self._status.showMessage("Đã tắt đèn rọi Spotlight", 2000)
        else:
            self._set_teaching_mode("SPOTLIGHT")
            self._status.showMessage("Đã bật đèn rọi Spotlight (Phím S)", 2000)

    @Slot()
    def _on_toggle_ruler(self) -> None:
        """Toggle virtual measurement ruler on LessonBoard."""
        is_visible = self._scene.toggle_ruler()
        self._status.showMessage(f"Thước kẻ centimet: {'Hiển thị' if is_visible else 'Đã ẩn'}", 2000)

    @Slot()
    def _on_toggle_protractor(self) -> None:
        """Toggle virtual angle protractor on LessonBoard."""
        is_visible = self._scene.toggle_protractor()
        self._status.showMessage(f"Thước đo độ: {'Hiển thị' if is_visible else 'Đã ẩn'}", 2000)

    @Slot()
    def _on_toggle_compass(self) -> None:
        """Toggle virtual geometric compass on LessonBoard."""
        is_visible = self._scene.toggle_compass()
        self._status.showMessage(
            f"Compa vẽ đường tròn: {'Hiển thị (Ctrl+Shift+C)' if is_visible else 'Đã ẩn'}", 2000
        )

    @Slot()
    def _on_toggle_grid(self) -> None:
        """Cycle through whiteboard grid modes: none -> grid -> cartesian -> dots -> none."""
        new_mode = self._scene.toggle_grid()
        labels = {
            "none": "Tắt lưới",
            "grid": "Lưới ô vuông",
            "cartesian": "Hệ trục tọa độ Oxy",
            "dots": "Lưới chấm điểm",
        }
        self._status.showMessage(f"Lưới bảng: {labels.get(new_mode, new_mode)}", 3000)

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
        """Open, reconstruct, and display an .airteach lesson package with all strokes and recognized items."""
        try:
            from src.services.document_manager import PageInfo  # noqa: PLC0415
            from src.services.smart_ink_ocr import LaTeXRenderer  # noqa: PLC0415
            manifest, page_paths, strokes, recognized = self._doc_manager.open_lesson(Path(pkg_path))
            self._pages = [
                PageInfo(page_index=i, image_path=str(p), width_px=0, height_px=0, text="")
                for i, p in enumerate(page_paths)
            ]
            if hasattr(self, "_slide_drawer") and self._slide_drawer is not None:
                self._slide_drawer.set_pages(self._pages, self._current_page_index)
            self._page_strokes.clear()
            self._page_pixmap_cache.clear()
            self._scene.clear_all_strokes()
            for stem_list in self._page_stem_items.values():
                for itm in stem_list:
                    if itm.scene() is self._scene:
                        self._scene.removeItem(itm)
            self._page_stem_items.clear()

            self._load_strokes_from_records(strokes)

            # Restore recognized text & LaTeX formula overlays
            for rec in recognized:
                p_idx = rec.page_index
                bx = rec.bounding_box[0] if rec.bounding_box else 120.0
                by = rec.bounding_box[1] if rec.bounding_box else 120.0
                ov_item = None
                if rec.is_formula and rec.latex:
                    try:
                        rgba = LaTeXRenderer.render_to_rgba(rec.latex)
                        if rgba is not None:
                            ov_item = self._scene.add_latex_overlay(rgba, bx, by)
                    except Exception as err:
                        logger.warning("Failed to render LaTeX overlay for '%s': %s", rec.latex, err)
                if ov_item is None and (rec.text or rec.latex):
                    ov_item = self._scene.add_text_overlay(rec.latex or rec.text, bx, by)
                if ov_item is not None:
                    ov_item.setData(1, {
                        "text": rec.text,
                        "latex": rec.latex,
                        "is_formula": rec.is_formula,
                        "conf": rec.confidence,
                    })
                    ov_item.setVisible(p_idx == 0)
                    self._page_stem_items.setdefault(p_idx, []).append(ov_item)

            self._current_page_index = 0
            self._show_current_page()
            self._status.showMessage(
                f"Đã mở {Path(pkg_path).name} — {len(strokes)} nét mực, {len(recognized)} đối tượng OCR.", 4000
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi mở gói bài giảng", str(exc))

    # ── Settings ──────────────────────────────────────────────────────────────

    def _on_settings(self) -> None:
        teacher_name = self._teacher.get("full_name", "Giáo viên") if self._teacher else "Giáo viên Demo"
        dlg = SettingsDialog(
            teacher_id=self._teacher_id,
            teacher_name=teacher_name,
            credential_service=self._cred_service,
            parent=self,
        )
        dlg.api_key_changed.connect(self._on_api_key_changed)
        dlg.camera_index_changed.connect(self._on_camera_index_changed)
        dlg.calibration_requested.connect(self._on_calibration_requested)
        dlg.settings_updated.connect(self._on_ai_settings_updated)
        dlg.set_grid_mode(self._board.grid_mode())
        dlg.set_grid_spacing(self._board.grid_spacing())
        dlg.set_laser_color(self._scene.laser_color())
        dlg.set_laser_radius(self._scene.laser_radius())
        dlg.set_laser_trail_enabled(self._scene.is_laser_trail_enabled())
        if dlg.exec():
            self._ink_color       = QColor(dlg.ink_color)
            self._ink_min_width   = dlg.min_stroke_width
            self._ink_max_width   = dlg.max_stroke_width
            self._rdp_epsilon     = dlg.rdp_epsilon
            self._one_euro_beta   = getattr(dlg, "one_euro_beta", 0.35)
            self._filter          = OneEuroFilter2D(beta=self._one_euro_beta)
            self._snapping_on     = dlg.shape_snapping_enabled
            self._multi_hand_mode = dlg.multi_hand_mode
            self._dominant_hand   = dlg.dominant_hand
            self._air_click_type  = dlg.air_click_type
            self._eraser_radius   = dlg.eraser_radius
            self._menu_layout_mode = dlg.menu_layout_mode
            self._floating_menu.set_layout_mode(dlg.menu_layout_mode)
            self._board.set_grid_mode(dlg.grid_mode)
            self._board.set_grid_spacing(dlg.grid_spacing)
            self._scene.set_laser_color(dlg.laser_color)
            self._scene.set_laser_radius(dlg.laser_radius)
            self._scene.set_laser_trail_enabled(dlg.laser_trail_enabled)
            if self._cam_worker is not None:
                self._cam_worker.set_multi_hand_mode(self._multi_hand_mode)
                self._cam_worker.set_dominant_hand(self._dominant_hand)
                self._cam_worker.set_air_click_type(self._air_click_type)

    @Slot()
    def _on_ai_settings_updated(self) -> None:
        self._ai_settings = self._cred_service.get_settings(self._teacher_id)
        if self._ai_settings.draw_mode:
            self._draw_mode = self._ai_settings.draw_mode
            self._apply_draw_mode()
        if self._ai_settings.camera_lighting_mode:
            self._on_lighting_mode_changed(self._ai_settings.camera_lighting_mode)
        if hasattr(self._ai_settings, "multi_hand_mode") and self._ai_settings.multi_hand_mode:
            self._multi_hand_mode = self._ai_settings.multi_hand_mode
            if self._cam_worker is not None:
                self._cam_worker.set_multi_hand_mode(self._multi_hand_mode)
        if hasattr(self._ai_settings, "dominant_hand") and self._ai_settings.dominant_hand:
            self._dominant_hand = self._ai_settings.dominant_hand
            if self._cam_worker is not None:
                self._cam_worker.set_dominant_hand(self._dominant_hand)
        if hasattr(self._ai_settings, "air_click_type") and self._ai_settings.air_click_type:
            self._air_click_type = self._ai_settings.air_click_type
            if self._cam_worker is not None:
                self._cam_worker.set_air_click_type(self._air_click_type)
        if hasattr(self._ai_settings, "eraser_radius") and self._ai_settings.eraser_radius:
            self._eraser_radius = float(self._ai_settings.eraser_radius)
        if hasattr(self._ai_settings, "menu_layout_mode") and self._ai_settings.menu_layout_mode:
            self._menu_layout_mode = self._ai_settings.menu_layout_mode
            self._floating_menu.set_layout_mode(self._menu_layout_mode)
        if hasattr(self, "_chat_panel"):
            self._chat_panel.update_online_status(self._assistant.is_online)
        self._status.showMessage("Cài đặt AI & Cử chỉ đã được lưu an toàn", 3000)

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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle teacher presentation remote, clicker, and keyboard shortcuts."""
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_items:
                self._delete_selected_items()
                event.accept()
                return

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    self._on_redo()
                else:
                    self._on_undo()
                event.accept()
                return
            elif key == Qt.Key.Key_Y:
                self._on_redo()
                event.accept()
                return

        if key in (Qt.Key.Key_Right, Qt.Key.Key_PageDown):
            self._on_next_page()
            event.accept()
            return
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_PageUp):
            self._on_prev_page()
            event.accept()
            return
        elif key == Qt.Key.Key_Escape:
            if self._selected_items or (self._scene._select_rect_item and self._scene._select_rect_item.isVisible()):
                self._scene.hide_selection_rect()
                self._selected_items.clear()
                self._selected_strokes.clear()
                self._is_dragging_selection = False
                self._status.showMessage("Đã hủy vùng chọn", 1500)
                event.accept()
                return

        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._disconnect_camera()
        self._telemetry.stop()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "_zoom_ctrl") and self._zoom_ctrl is not None:
            if abs(self._zoom_ctrl.zoom_factor() - 1.0) < 0.01:
                self._board_view.fitInView(
                    self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
                )
            self._zoom_ctrl.update_position()
        else:
            self._board_view.fitInView(
                self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
