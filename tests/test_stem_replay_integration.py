"""
AirTeach AI - STEM Tools, Multi-Hand Progressive Replay & Camera Overlay Tests
=============================================================================
Validates:
  1. STEM Tools submenu integration in Floating Action Menu (radial & panel)
  2. Main Window routing of in-air STEM tool commands to specialized dialogs
  3. Progressive multi-hand stroke replay synchronization (both hands drawing in parallel)
  4. Multi-hand skeleton annotation and HUD telemetry in camera preview frame
  5. Multi-hand pipeline latency and performance verification (<= 15 ms budget)
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath
from PySide6.QtWidgets import QApplication

from src.core.multi_hand_coordinator import (
    HandLandmarkData,
    HandRole,
    MultiHandCoordinator,
    MultiHandTracker,
)
from src.core.hand_gesture_engine import GestureState, LM
from src.core.stroke_beautifier import BezierSegment
from src.ui.floating_action_menu import (
    FloatingActionMenu,
    MenuLayoutMode,
    MenuItem,
    STEM_MENU_ITEMS,
    ROOT_MENU_ITEMS,
)
from src.ui.lesson_board_canvas import LessonBoardCanvas, InkStroke
from src.ui.main_window import MainWindow, _QtCameraWorker
from src.ui.replay_controller import ReplayController


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ─── 1. STEM Menu Navigation & Execution Tests ─────────────────────────────────

def test_stem_submenu_in_floating_menu(qapp):
    """Verify STEM submenu exists, contains required tools, and emits actions."""
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    menu.show_at(960.0, 540.0)

    # Verify STEM item exists in root menu
    stem_item = next((item for item in menu._items if item.item_id == "stem"), None)
    assert stem_item is not None
    assert stem_item.command == "submenu"
    assert stem_item.submenu_id == "stem_menu"

    # Navigate to stem_menu
    stem_idx = menu._items.index(stem_item)
    menu._hovered_index = stem_idx
    nav_result = menu.trigger_click()
    assert nav_result == ("nav", "stem_menu")
    assert menu._current_menu_id == "stem_menu"
    assert len(menu._items) == len(STEM_MENU_ITEMS)

    # Check STEM tools present
    ids = [item.item_id for item in menu._items]
    assert "stem_oxy" in ids
    assert "stem_func" in ids
    assert "stem_force" in ids
    assert "stem_chem" in ids
    assert "stem_flow" in ids
    assert "stem_code" in ids
    assert "back" in ids

    # Activate stem_oxy item
    oxy_item = next(item for item in menu._items if item.item_id == "stem_oxy")
    oxy_idx = menu._items.index(oxy_item)
    menu._hovered_index = oxy_idx

    triggered_signals = []
    menu.action_triggered.connect(lambda cmd, val: triggered_signals.append((cmd, val)))

    click_result = menu.trigger_click()
    assert click_result == ("stem", "oxy")
    assert len(triggered_signals) == 1
    assert triggered_signals[0] == ("stem", "oxy")

    # Menu auto-closes after selecting a STEM tool
    assert not menu.isVisible()


def test_main_window_stem_menu_action_handling(qapp):
    """Verify MainWindow routes STEM commands from floating menu to appropriate dialog triggers."""
    win = MainWindow(camera_index=-1)
    win.show()

    with patch.object(win, "_on_insert_stem_oxy") as mock_oxy, \
         patch.object(win, "_on_insert_stem_function") as mock_func, \
         patch.object(win, "_on_insert_stem_force") as mock_force, \
         patch.object(win, "_on_insert_stem_flowchart") as mock_flow:

        win._on_menu_action_triggered("stem", "oxy")
        mock_oxy.assert_called_once()

        win._on_menu_action_triggered("stem", "function")
        mock_func.assert_called_once()

        win._on_menu_action_triggered("stem", "force")
        mock_force.assert_called_once()

        win._on_menu_action_triggered("stem", "flowchart")
        mock_flow.assert_called_once()

    win.close()


# ─── 2. Multi-Hand Progressive Replay Tests ───────────────────────────────────

def test_multi_hand_progressive_replay(qapp):
    """
    Verify strokes drawn by both hands concurrently animate progressively
    in parallel during playback.
    """
    canvas = LessonBoardCanvas()
    ctrl = ReplayController(canvas=canvas)

    # Create Stroke 1 (Hand 0: starts at 1000ms, ends at 2000ms, 4 segments)
    segs1 = [
        BezierSegment(np.array([100.0, 100.0]), np.array([120.0, 110.0]), np.array([140.0, 110.0]), np.array([160.0, 100.0])),
        BezierSegment(np.array([160.0, 100.0]), np.array([180.0, 110.0]), np.array([200.0, 110.0]), np.array([220.0, 100.0])),
        BezierSegment(np.array([220.0, 100.0]), np.array([240.0, 110.0]), np.array([260.0, 110.0]), np.array([280.0, 100.0])),
        BezierSegment(np.array([280.0, 100.0]), np.array([300.0, 110.0]), np.array([320.0, 110.0]), np.array([340.0, 100.0])),
    ]
    s1 = InkStroke(
        stroke_id="hand_0_stroke",
        segments=segs1,
        widths=[3.0],
        color=QColor("#1a73e8"),
        timestamps_ms=[1000.0, 1250.0, 1500.0, 1750.0, 2000.0],
    )
    canvas._render_and_commit(s1)

    # Create Stroke 2 (Hand 1: starts at 1200ms, ends at 2200ms, 4 segments)
    segs2 = [
        BezierSegment(np.array([500.0, 500.0]), np.array([520.0, 510.0]), np.array([540.0, 510.0]), np.array([560.0, 500.0])),
        BezierSegment(np.array([560.0, 500.0]), np.array([580.0, 510.0]), np.array([600.0, 510.0]), np.array([620.0, 500.0])),
        BezierSegment(np.array([620.0, 500.0]), np.array([640.0, 510.0]), np.array([660.0, 510.0]), np.array([680.0, 500.0])),
        BezierSegment(np.array([680.0, 500.0]), np.array([700.0, 510.0]), np.array([720.0, 510.0]), np.array([740.0, 500.0])),
    ]
    s2 = InkStroke(
        stroke_id="hand_1_stroke",
        segments=segs2,
        widths=[4.0],
        color=QColor("#ea4335"),
        timestamps_ms=[1200.0, 1450.0, 1700.0, 1950.0, 2200.0],
    )
    canvas._render_and_commit(s2)

    ctrl.load_strokes([s1, s2])
    assert ctrl._start_time_ms == 1000.0
    assert ctrl._total_duration_ms == 1200.0  # 2200 - 1000

    # 1. At t = 0 (abs_time = 1000 ms)
    # Stroke 1 just started (1st segment), Stroke 2 has not started yet
    ctrl._apply_timestamp_to_canvas(0.0)
    assert s1._path_item.isVisible()
    assert not s2._path_item.isVisible()

    # 2. At t = 500 (abs_time = 1500 ms)
    # Stroke 1 elapsed 500ms / 1000ms = 50% (2 segments)
    # Stroke 2 elapsed 300ms / 1000ms = 30% (2 segments)
    # BOTH strokes must be visible and drawing simultaneously
    ctrl._apply_timestamp_to_canvas(500.0)
    assert s1._path_item.isVisible()
    assert s2._path_item.isVisible()

    # Verify that partial path has been generated and set
    path1 = s1._path_item.path()
    assert not path1.isEmpty()
    path2 = s2._path_item.path()
    assert not path2.isEmpty()

    # 3. At t = 1300 (abs_time = 2300 ms, past both strokes' end times)
    ctrl._apply_timestamp_to_canvas(1300.0)
    assert s1._path_item.isVisible()
    assert s2._path_item.isVisible()

    # 4. Restore full strokes
    ctrl.restore_full_strokes()
    assert s1._path_item.isVisible()
    assert s2._path_item.isVisible()


# ─── 3. Multi-Hand Camera Preview Skeleton & HUD Telemetry Tests ───────────────

def test_camera_worker_multi_hand_skeleton_annotation(qapp):
    """
    Verify _QtCameraWorker annotates skeletons for multiple hands simultaneously
    with distinctive colors, index fingertip rings, and status badge.
    """
    worker = _QtCameraWorker(camera_index=0)

    # Construct two synthetic hands with 21 landmarks
    class MockPt:
        def __init__(self, x: float, y: float, z: float = 0.0):
            self.x = x
            self.y = y
            self.z = z

    lms0 = [MockPt(0.2 + 0.01 * i, 0.3 + 0.01 * (i % 5)) for i in range(21)]
    h0 = HandLandmarkData(
        landmarks=lms0,
        handedness="Right",
        confidence=0.95,
        track_id="hand_0",
        gesture_state=GestureState.DRAW,
        role=HandRole.PRIMARY,
    )

    lms1 = [MockPt(0.7 + 0.01 * i, 0.4 + 0.01 * (i % 5)) for i in range(21)]
    h1 = HandLandmarkData(
        landmarks=lms1,
        handedness="Left",
        confidence=0.92,
        track_id="hand_1",
        gesture_state=GestureState.POINT,
        is_pinched=True,
        role=HandRole.SECONDARY,
    )

    # Deliver multi_hands callback
    worker._on_multi_hands_cb([h0, h1])
    assert len(worker._last_multi_hands_data) == 2

    # Receive annotated frame signal
    received_frames = []
    worker.frame_ready.connect(lambda f: received_frames.append(f))

    # Send a blank black frame
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    worker._on_frame_cb(black_frame, lms0, GestureState.DRAW, 0.25, 0.35)

    assert len(received_frames) == 1
    ann = received_frames[0]
    assert ann.shape == (480, 640, 3)

    # The annotated frame must have non-zero pixels from both hands' skeletons and badges
    mean_val = float(np.mean(ann))
    assert mean_val > 0.5, "Frame should contain non-zero skeleton and badge overlays"


# ─── 4. Performance & Latency Budget Test ─────────────────────────────────────

def test_multi_hand_pipeline_latency_budget():
    """
    Verify multi-hand tracking and coordination executes well within the 15ms
    per-frame latency budget over 60 simulated dual-hand frames.
    """
    coordinator = MultiHandCoordinator(
        canvas_width=1920,
        canvas_height=1080,
        mode="dual_draw",
    )

    class MockPt:
        def __init__(self, x: float, y: float, z: float = 0.0):
            self.x = x
            self.y = y
            self.z = z

    frame_times_ms = []

    # Warmup frame with spatially separated hands to initialize lazy imports and JIT
    lms_warm0 = [MockPt(0.3, 0.4) for _ in range(21)]
    lms_warm1 = [MockPt(0.7, 0.5) for _ in range(21)]
    coordinator.process_frame_hands([(lms_warm0, 'Right', 0.95), (lms_warm1, 'Left', 0.92)], 900.0)

    for frame_idx in range(60):
        t_ms = 1000.0 + frame_idx * 16.6
        # Simulate slight motion
        offset = 0.001 * frame_idx
        lms0 = [MockPt(0.3 + offset, 0.4 + offset) for _ in range(21)]
        lms1 = [MockPt(0.7 - offset, 0.5 + offset) for _ in range(21)]

        raw_detections = [
            (lms0, "Right", 0.95),
            (lms1, "Left", 0.92),
        ]

        t0 = time.perf_counter()
        result = coordinator.process_frame_hands(raw_detections, t_ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        frame_times_ms.append(elapsed_ms)

        assert len(result) == 2

    avg_ms = float(np.mean(frame_times_ms))
    max_ms = float(np.max(frame_times_ms))
    # Budget: avg <= 15.0 ms (target 60fps = 16.6ms total)
    assert avg_ms < 15.0, f"Average latency too high: {avg_ms:.2f} ms"
    assert max_ms < 60.0, f"Max spike too high: {max_ms:.2f} ms"


# ─── 5. Main Window STEM Lifecycle & Unified Undo/Redo Tests ───────────────────

def test_main_window_stem_lifecycle_and_page_switching(qapp):
    """
    Verify STEM items are isolated per page, correctly hidden when stashed,
    and restored when navigating back to the original page.
    """
    from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsLineItem  # noqa: PLC0415
    win = MainWindow(camera_index=-1)
    win.show()

    # Configure 2 mock pages
    win._pages = [
        MagicMock(image_path="page0.png", text="Page 0 Content"),
        MagicMock(image_path="page1.png", text="Page 1 Content"),
    ]
    win._current_page_index = 0

    # Add a STEM diagram (e.g., coordinate axes box) on page 0
    rect_item = QGraphicsRectItem(100, 100, 300, 200)
    win._scene.addItem(rect_item)
    win._add_stem_item_to_current_page(rect_item)

    assert 0 in win._page_stem_items
    assert rect_item in win._page_stem_items[0]
    assert rect_item.isVisible()
    assert len(win._page_undo_stack[0]) == 1
    assert win._page_undo_stack[0][0][0] == "stem"

    # Stash page 0 and switch to page 1
    win._stash_current_page_strokes()
    assert not rect_item.isVisible(), "STEM item from page 0 must be hidden when page 0 is stashed"

    win._current_page_index = 1
    win._restore_current_page_strokes()

    # Add a different STEM item on page 1
    line_item = QGraphicsLineItem(50, 50, 400, 400)
    win._scene.addItem(line_item)
    win._add_stem_item_to_current_page(line_item)

    assert line_item in win._page_stem_items[1]
    assert line_item.isVisible()
    assert not rect_item.isVisible()

    # Stash page 1 and switch back to page 0
    win._stash_current_page_strokes()
    assert not line_item.isVisible()

    win._current_page_index = 0
    win._restore_current_page_strokes()

    assert rect_item.isVisible(), "Page 0 STEM item must be restored and visible"
    assert not line_item.isVisible(), "Page 1 STEM item must remain hidden on Page 0"

    win.close()


def test_main_window_unified_undo_redo_stem_and_ink(qapp):
    """
    Verify unified undo/redo stack correctly interleaves vector ink strokes
    and STEM diagrams in chronological LIFO order.
    """
    from PySide6.QtWidgets import QGraphicsEllipseItem  # noqa: PLC0415
    win = MainWindow(camera_index=-1)
    win.show()
    win._current_page_index = 0

    # 1. Draw an ink stroke
    win._start_new_hand_stroke("primary", 50.0, 50.0, 1000.0)
    win._extend_hand_stroke("primary", 100.0, 100.0, 1020.0)
    win._finalize_hand_stroke("primary")

    assert len(win._scene._strokes) == 1
    assert len(win._page_undo_stack[0]) == 1
    assert win._page_undo_stack[0][-1][0] == "stroke"

    # 2. Insert a STEM item (e.g., circle / particle)
    circle_item = QGraphicsEllipseItem(200, 200, 80, 80)
    win._scene.addItem(circle_item)
    win._add_stem_item_to_current_page(circle_item)

    assert len(win._page_undo_stack[0]) == 2
    assert win._page_undo_stack[0][-1][0] == "stem"
    assert circle_item in win._page_stem_items[0]
    assert circle_item.isVisible()

    # 3. Undo 1: Should undo the most recent action (STEM circle)
    win._on_undo()
    assert not circle_item.isVisible(), "STEM item should be hidden after undo"
    assert circle_item not in win._page_stem_items[0]
    assert len(win._scene._strokes) == 1, "Ink stroke must still remain untouched"
    assert len(win._page_redo_stack[0]) == 1
    assert win._page_redo_stack[0][-1][0] == "stem"

    # 4. Undo 2: Should undo the preceding action (Ink stroke)
    win._on_undo()
    assert len(win._scene._strokes) == 0, "Ink stroke should be removed from scene after undo"
    assert len(win._page_redo_stack[0]) == 2
    assert win._page_redo_stack[0][-1][0] == "stroke"

    # 5. Redo 1: Should restore the ink stroke first
    win._on_redo()
    assert len(win._scene._strokes) == 1, "Ink stroke must be restored first"
    assert not circle_item.isVisible(), "STEM item must still remain undone"

    # 6. Redo 2: Should restore the STEM item
    win._on_redo()
    assert circle_item.isVisible(), "STEM item must be restored and visible after second redo"
    assert circle_item in win._page_stem_items[0]

    win.close()


def test_main_window_export_pdf_with_stem_and_ink(qapp, tmp_path):
    """
    Verify MainWindow._on_export_pdf exports whiteboard and document pages
    including both vector ink strokes and STEM graphics items into valid PDFs.
    """
    from PySide6.QtWidgets import QGraphicsRectItem  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    win = MainWindow(camera_index=-1)
    win.show()

    # 1. Whiteboard export test (no pre-loaded document pages)
    # Draw ink stroke
    win._start_new_hand_stroke("primary", 100.0, 100.0, 1000.0)
    win._extend_hand_stroke("primary", 200.0, 200.0, 1020.0)
    win._finalize_hand_stroke("primary")

    # Add a STEM rectangle
    stem_box = QGraphicsRectItem(300, 300, 250, 150)
    win._scene.addItem(stem_box)
    win._add_stem_item_to_current_page(stem_box)

    wb_pdf = tmp_path / "whiteboard_export.pdf"
    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(wb_pdf), "PDF (*.pdf)")):
        win._on_export_pdf()

    assert wb_pdf.exists(), "Whiteboard PDF must be created"
    assert wb_pdf.stat().st_size > 1000, "PDF file must have non-trivial size"

    # 2. Multi-page document export test
    p0_img = tmp_path / "page_0.png"
    p1_img = tmp_path / "page_1.png"
    dummy_bg0 = np.full((1080, 1920, 3), 245, dtype=np.uint8)
    dummy_bg1 = np.full((1080, 1920, 3), 235, dtype=np.uint8)
    cv2.imwrite(str(p0_img), dummy_bg0)
    cv2.imwrite(str(p1_img), dummy_bg1)

    win._pages = [
        MagicMock(image_path=str(p0_img), text="Document Page 1"),
        MagicMock(image_path=str(p1_img), text="Document Page 2"),
    ]
    win._current_page_index = 0

    doc_pdf = tmp_path / "document_export.pdf"
    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(doc_pdf), "PDF (*.pdf)")):
        win._on_export_pdf()

    assert doc_pdf.exists(), "Document PDF must be created"
    assert doc_pdf.stat().st_size > 2000, "Multi-page PDF must have valid content"

    # Verify multi-page structure using pypdf
    import pypdf  # noqa: PLC0415
    reader = pypdf.PdfReader(str(doc_pdf))
    assert len(reader.pages) == 2, f"Document PDF should contain 2 pages, got {len(reader.pages)}"

    win.close()


def test_main_window_ocr_overlay_lifecycle_and_undo(qapp):
    """
    Verify OCR text and LaTeX overlays added to the scene are tracked per page,
    properly hidden when switching pages, and undoable via unified Undo/Redo stack.
    """
    win = MainWindow(camera_index=-1)
    win.show()

    win._pages = [
        MagicMock(image_path="page0.png", text=""),
        MagicMock(image_path="page1.png", text=""),
    ]
    win._current_page_index = 0

    # Add OCR text overlay on page 0
    txt_item = win._scene.add_text_overlay("E = mc^2", 150.0, 150.0)
    win._add_stem_item_to_current_page(txt_item)

    assert txt_item in win._page_stem_items[0]
    assert txt_item.isVisible()
    assert len(win._page_undo_stack[0]) == 1

    # Stash page 0 and navigate to page 1
    win._stash_current_page_strokes()
    assert not txt_item.isVisible(), "OCR text overlay must be hidden on other pages"

    win._current_page_index = 1
    win._restore_current_page_strokes()
    assert not txt_item.isVisible()

    # Navigate back to page 0
    win._stash_current_page_strokes()
    win._current_page_index = 0
    win._restore_current_page_strokes()
    assert txt_item.isVisible(), "OCR text overlay must be restored on original page"

    # Undo OCR overlay
    win._on_undo()
    assert not txt_item.isVisible(), "OCR overlay must be hidden after undo"
    assert txt_item not in win._page_stem_items[0]

    # Redo OCR overlay
    win._on_redo()
    assert txt_item.isVisible(), "OCR overlay must be restored after redo"
    assert txt_item in win._page_stem_items[0]

    win.close()




def test_main_window_edited_formula_ocr_rendering(qapp):
    """
    Verify that when a teacher edits the recognized LaTeX formula in SmartInkReviewDialog,
    MainWindow._on_ocr_done re-renders fresh RGBA vector math and adds the updated item.
    """
    win = MainWindow(camera_index=-1)
    win.show()

    from src.services.smart_ink_ocr import OcrResult
    init_rgba = np.full((30, 80, 4), 255, dtype=np.uint8)
    ocr_res = OcrResult(
        text="x + y = 0",
        latex="x + y = 0",
        confidence=0.85,
        is_formula=True,
        render_image_rgba=init_rgba,
    )
    setattr(ocr_res, "_board_x", 200.0)
    setattr(ocr_res, "_board_y", 300.0)

    with patch("src.ui.smart_ink_review_dialog.SmartInkReviewDialog") as MockDlgClass:
        instance = MagicMock()
        MockDlgClass.return_value = instance

        def trigger_confirm():
            for call_arg in instance.result_confirmed.connect.call_args_list:
                cb = call_arg[0][0]
                cb("x^2 + y^2 = r^2", False)
        instance.exec.side_effect = trigger_confirm

        win._on_ocr_done(ocr_res)

    items = win._page_stem_items.get(win._current_page_index, [])
    assert len(items) >= 1
    last_item = items[-1]
    assert last_item.isVisible()
    item_data = last_item.data(1)
    assert item_data is not None
    assert item_data["latex"] == "x^2 + y^2 = r^2"
    assert item_data["is_formula"] is True

    win.close()








def test_informatics_logic_gates(qapp):
    """Verify InformaticsTools.create_logic_gate creates accurate vector items for all gate types."""
    from src.ui.stem_tools import InformaticsTools

    for gate_type in ("AND", "OR", "NOT", "NAND", "NOR", "XOR"):
        items = InformaticsTools.create_logic_gate(
            gate_type=gate_type,
            center=(400.0, 300.0),
            width=120.0,
            height=70.0,
            input_labels=("In1", "In2") if gate_type != "NOT" else ("A",),
            output_label="Out",
        )
        assert len(items) >= 4, f"Gate {gate_type} must produce body, pins, and labels"
        # At least one QGraphicsPathItem (body)
        from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsLineItem, QGraphicsTextItem
        path_items = [it for it in items if isinstance(it, QGraphicsPathItem)]
        line_items = [it for it in items if isinstance(it, QGraphicsLineItem)]
        text_items = [it for it in items if isinstance(it, QGraphicsTextItem)]

        assert len(path_items) >= 1, f"Gate {gate_type} must contain body path"
        assert len(line_items) >= (2 if gate_type == "NOT" else 3), f"Gate {gate_type} must have pins"
        assert len(text_items) >= 2, f"Gate {gate_type} must have input/output labels"


def test_informatics_flow_arrow(qapp):
    """Verify InformaticsTools.create_flow_arrow creates straight and elbow arrows with labels."""
    from src.ui.stem_tools import InformaticsTools
    from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsTextItem

    # 1. Straight arrow
    items_straight = InformaticsTools.create_flow_arrow(
        start=(100.0, 100.0),
        end=(300.0, 100.0),
        arrow_type="straight",
        label="Đúng",
    )
    assert len(items_straight) >= 3
    paths = [it for it in items_straight if isinstance(it, QGraphicsPathItem)]
    heads = [it for it in items_straight if isinstance(it, QGraphicsPolygonItem)]
    labels = [it for it in items_straight if isinstance(it, QGraphicsTextItem)]
    assert len(paths) == 1
    assert len(heads) == 1
    assert len(labels) == 1
    assert labels[0].toPlainText() == "Đúng"

    # 2. Elbow arrow
    items_elbow = InformaticsTools.create_flow_arrow(
        start=(100.0, 100.0),
        end=(200.0, 300.0),
        arrow_type="elbow",
        label="Sai",
    )
    assert len(items_elbow) >= 3
    labels_elbow = [it for it in items_elbow if isinstance(it, QGraphicsTextItem)]
    assert len(labels_elbow) == 1
    assert labels_elbow[0].toPlainText() == "Sai"


def test_main_window_insert_logic_gate_and_flow_arrow(qapp):
    """Verify MainWindow slots for inserting logic gate and flow arrow with undo/redo."""
    win = MainWindow(camera_index=-1)
    win.show()

    initial_count = len(win._page_stem_items.get(win._current_page_index, []))

    # Mock dialog for logic gate
    with patch("src.ui.main_window.StemToolDialog") as MockDlg:
        instance = MagicMock()
        MockDlg.return_value = instance
        instance.exec.return_value = True
        instance.combo_gate_type.currentText.return_value = "AND (Cổng Và: Y = A · B)"
        instance.inp_inputs.text.return_value = "A, B"
        instance.inp_output.text.return_value = "Y"

        win._on_insert_stem_logic_gate()

    current_items = win._page_stem_items.get(win._current_page_index, [])
    assert len(current_items) > initial_count, "Logic gate items must be added to page"

    # Mock dialog for flow arrow
    gate_count = len(current_items)
    with patch("src.ui.main_window.StemToolDialog") as MockDlg:
        instance = MagicMock()
        MockDlg.return_value = instance
        instance.exec.return_value = True
        instance.combo_arrow_type.currentIndex.return_value = 1  # elbow
        instance.combo_condition.currentText.return_value = "Đúng (True)"

        win._on_insert_stem_flow_arrow()

    arrow_items = win._page_stem_items.get(win._current_page_index, [])
    assert len(arrow_items) > gate_count, "Flow arrow items must be added to page"

    # Test Undo flow arrow
    win._on_undo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == gate_count

    # Test Redo flow arrow
    win._on_redo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == len(arrow_items)

    win.close()








def test_biology_tools_dna_double_helix():
    """Verify BiologyTools.create_dna_double_helix produces strands, rungs, and terminal labels."""
    from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsLineItem, QGraphicsTextItem
    from src.ui.stem_tools import BiologyTools

    items = BiologyTools.create_dna_double_helix(
        start=(100.0, 300.0),
        end=(500.0, 300.0),
        turns=2.5,
        amplitude=30.0,
        rungs=12,
        show_labels=True,
    )
    assert len(items) >= 20, "Must create backbones, rungs, and terminal labels"

    strands = [it for it in items if isinstance(it, QGraphicsPathItem)]
    assert len(strands) == 2, "Must create 2 continuous sinusoidal backbone strands"

    rungs = [it for it in items if isinstance(it, QGraphicsLineItem)]
    assert len(rungs) == 24, "12 base-pair rungs split into 2 halves with hydrogen bond gap = 24 segments"

    labels = [it for it in items if isinstance(it, QGraphicsTextItem)]
    label_texts = [it.toPlainText() for it in labels]
    assert "5'" in label_texts
    assert "3'" in label_texts


def test_biology_tools_punnett_square():
    """Verify BiologyTools.create_punnett_square creates grid, gametes, and offspring genotypes."""
    from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem
    from src.ui.stem_tools import BiologyTools

    items = BiologyTools.create_punnett_square(
        center=(500.0, 400.0),
        cell_size=60.0,
        male_gametes=("B", "b"),
        female_gametes=("B", "b"),
        title="Phép lai: Bb × Bb",
    )
    assert len(items) >= 15, "Must create outer card, banner, headers, cells, and genotypes"

    texts = [it.toPlainText() for it in items if isinstance(it, QGraphicsTextItem)]
    assert "Phép lai: Bb × Bb" in texts
    assert "♂" in texts
    assert "♀" in texts
    assert "B" in texts
    assert "b" in texts
    assert "BB" in texts
    assert "Bb" in texts
    assert "bb" in texts


def test_main_window_insert_dna_and_punnett(qapp):
    """Verify MainWindow slots for inserting DNA double helix and Punnett square with undo/redo."""
    win = MainWindow(camera_index=-1)
    win.show()

    initial_count = len(win._page_stem_items.get(win._current_page_index, []))

    # Mock dialog for DNA double helix
    with patch("src.ui.main_window.StemToolDialog") as MockDlg:
        instance = MagicMock()
        MockDlg.return_value = instance
        instance.exec.return_value = True
        instance.inp_dna_len.text.return_value = "360"
        instance.inp_turns.text.return_value = "3.0"
        instance.inp_amp.text.return_value = "25"

        win._on_insert_stem_dna()

    current_items = win._page_stem_items.get(win._current_page_index, [])
    assert len(current_items) > initial_count, "DNA items must be added to page"
    dna_count = len(current_items)

    # Mock dialog for Punnett square
    with patch("src.ui.main_window.StemToolDialog") as MockDlg:
        instance = MagicMock()
        MockDlg.return_value = instance
        instance.exec.return_value = True
        instance.inp_male.text.return_value = "A, a"
        instance.inp_female.text.return_value = "A, a"
        instance.inp_cross_title.text.return_value = "Phép lai Aa × Aa"

        win._on_insert_stem_punnett()

    punnett_items = win._page_stem_items.get(win._current_page_index, [])
    assert len(punnett_items) > dna_count, "Punnett items must be added to page"

    # Test Undo Punnett square
    win._on_undo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == dna_count

    # Test Redo Punnett square
    win._on_redo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == len(punnett_items)

    win.close()




def test_physics_tools_optics_ray_diagram_converging():
    """Verify PhysicsTools.create_optics_ray_diagram generates thin converging lens ray optics."""
    from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsTextItem, QGraphicsPathItem
    from src.ui.stem_tools import PhysicsTools

    items = PhysicsTools.create_optics_ray_diagram(
        center=(500.0, 300.0),
        lens_type="converging",
        focal_length=120.0,
        object_dist=200.0,
        object_height=60.0,
    )
    assert len(items) >= 15, "Must produce axis, lens, arrows, dots, object, image, and rays"

    texts = [it.toPlainText() for it in items if isinstance(it, QGraphicsTextItem)]
    assert "O" in texts
    assert "F" in texts
    assert "F'" in texts
    assert "A" in texts
    assert "B" in texts
    assert "A'" in texts
    assert "B'" in texts

    lines = [it for it in items if isinstance(it, QGraphicsLineItem)]
    assert len(lines) >= 6, "Must include axis, lens line, object line, image line, and rays"


def test_physics_tools_optics_ray_diagram_diverging():
    """Verify PhysicsTools.create_optics_ray_diagram generates thin diverging lens with virtual image."""
    from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsTextItem
    from src.ui.stem_tools import PhysicsTools

    items = PhysicsTools.create_optics_ray_diagram(
        center=(500.0, 300.0),
        lens_type="diverging",
        focal_length=100.0,
        object_dist=150.0,
        object_height=50.0,
    )
    assert len(items) >= 15

    texts = [it.toPlainText() for it in items if isinstance(it, QGraphicsTextItem)]
    assert "O" in texts
    assert "F" in texts
    assert "F'" in texts
    assert "A" in texts
    assert "B" in texts
    assert "A'" in texts
    assert "B'" in texts


def test_main_window_insert_optics_ray_diagram(qapp):
    """Verify MainWindow slot for inserting optics ray tracing diagram with undo/redo."""
    win = MainWindow(camera_index=-1)
    win.show()

    initial_count = len(win._page_stem_items.get(win._current_page_index, []))

    with patch("src.ui.main_window.StemToolDialog") as MockDlg:
        instance = MagicMock()
        MockDlg.return_value = instance
        instance.exec.return_value = True
        instance.combo_lens_type.currentText.return_value = "Thấu kính hội tụ (Converging Lens)"
        instance.inp_focal.text.return_value = "120"
        instance.inp_obj_dist.text.return_value = "200"
        instance.inp_obj_height.text.return_value = "60"

        win._on_insert_stem_optics()

    current_items = win._page_stem_items.get(win._current_page_index, [])
    assert len(current_items) > initial_count, "Optics items must be added to page"

    # Test Undo
    win._on_undo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == initial_count

    # Test Redo
    win._on_redo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == len(current_items)

    win.close()




def test_chemistry_tools_molecular_geometry_tetrahedral():
    """Verify ChemistryTools.create_molecular_geometry creates tetrahedral CH4 model with 3D spheres."""
    from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
    from src.ui.stem_tools import ChemistryTools

    items = ChemistryTools.create_molecular_geometry(
        center=(400.0, 300.0),
        geometry_type="tetrahedral",
        scale=80.0,
    )
    assert len(items) >= 10

    # 5 atoms (1 Carbon + 4 Hydrogens)
    atoms = [it for it in items if isinstance(it, QGraphicsEllipseItem)]
    assert len(atoms) == 5

    # 4 covalent bonds
    bonds = [it for it in items if isinstance(it, QGraphicsLineItem)]
    assert len(bonds) == 4

    # Text symbols and title
    texts = [it.toPlainText() for it in items if isinstance(it, QGraphicsTextItem)]
    assert "C" in texts
    assert "H" in texts
    assert any("CH₄" in t for t in texts)


def test_chemistry_tools_molecular_geometry_varieties():
    """Verify Bent, Linear, Trigonal Planar, Pyramidal, and Octahedral molecular geometries."""
    from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
    from src.ui.stem_tools import ChemistryTools

    # 1. Bent (H2O)
    h2o_items = ChemistryTools.create_molecular_geometry(
        center=(300.0, 200.0),
        geometry_type="bent",
        scale=70.0,
    )
    h2o_atoms = [it for it in h2o_items if isinstance(it, QGraphicsEllipseItem)]
    assert len(h2o_atoms) == 3
    h2o_texts = [it.toPlainText() for it in h2o_items if isinstance(it, QGraphicsTextItem)]
    assert "O" in h2o_texts and "H" in h2o_texts
    assert any("H₂O" in t for t in h2o_texts)

    # 2. Linear (CO2)
    co2_items = ChemistryTools.create_molecular_geometry(
        center=(300.0, 200.0),
        geometry_type="linear",
        scale=70.0,
    )
    co2_atoms = [it for it in co2_items if isinstance(it, QGraphicsEllipseItem)]
    assert len(co2_atoms) == 3
    co2_texts = [it.toPlainText() for it in co2_items if isinstance(it, QGraphicsTextItem)]
    assert "C" in co2_texts and "O" in co2_texts
    assert any("CO₂" in t for t in co2_texts)

    # 3. Trigonal Planar (BF3)
    bf3_items = ChemistryTools.create_molecular_geometry(
        center=(300.0, 200.0),
        geometry_type="trigonal_planar",
        scale=70.0,
    )
    bf3_atoms = [it for it in bf3_items if isinstance(it, QGraphicsEllipseItem)]
    assert len(bf3_atoms) == 4
    bf3_texts = [it.toPlainText() for it in bf3_items if isinstance(it, QGraphicsTextItem)]
    assert "B" in bf3_texts and "F" in bf3_texts

    # 4. Trigonal Pyramidal (NH3)
    nh3_items = ChemistryTools.create_molecular_geometry(
        center=(300.0, 200.0),
        geometry_type="trigonal_pyramidal",
        scale=70.0,
    )
    nh3_atoms = [it for it in nh3_items if isinstance(it, QGraphicsEllipseItem)]
    assert len(nh3_atoms) == 4
    nh3_texts = [it.toPlainText() for it in nh3_items if isinstance(it, QGraphicsTextItem)]
    assert "N" in nh3_texts and "H" in nh3_texts

    # 5. Octahedral (SF6)
    sf6_items = ChemistryTools.create_molecular_geometry(
        center=(300.0, 200.0),
        geometry_type="octahedral",
        scale=70.0,
    )
    sf6_atoms = [it for it in sf6_items if isinstance(it, QGraphicsEllipseItem)]
    assert len(sf6_atoms) == 7
    sf6_texts = [it.toPlainText() for it in sf6_items if isinstance(it, QGraphicsTextItem)]
    assert "S" in sf6_texts and "F" in sf6_texts


def test_stem_tool_dialog_chem_molecule(qapp):
    """Verify StemToolDialog with chem_molecule type initializes and switches default titles."""
    from src.ui.stem_tools import StemToolDialog

    dlg = StemToolDialog("Mô hình Phân tử", "chem_molecule")
    assert hasattr(dlg, "combo_mol_type")
    assert hasattr(dlg, "inp_scale")
    assert hasattr(dlg, "inp_title")
    assert dlg.combo_mol_type.count() == 6

    # Change to Bent H2O
    dlg.combo_mol_type.setCurrentIndex(1)
    assert "H₂O" in dlg.inp_title.text()

    # Change to Linear CO2
    dlg.combo_mol_type.setCurrentIndex(2)
    assert "CO₂" in dlg.inp_title.text()


def test_main_window_insert_chem_molecule(qapp):
    """Verify MainWindow slot for inserting molecular geometry with undo/redo."""
    win = MainWindow(camera_index=-1)
    win.show()

    initial_count = len(win._page_stem_items.get(win._current_page_index, []))

    with patch("src.ui.main_window.StemToolDialog") as MockDlg:
        instance = MagicMock()
        MockDlg.return_value = instance
        instance.exec.return_value = True
        instance.combo_mol_type.currentIndex.return_value = 0  # tetrahedral
        instance.inp_scale.text.return_value = "80"
        instance.inp_title.text.return_value = "Metan (CH₄)"

        win._on_insert_stem_molecule()

    current_items = win._page_stem_items.get(win._current_page_index, [])
    assert len(current_items) > initial_count, "Molecular items must be added to page"

    # Test Undo
    win._on_undo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == initial_count

    # Test Redo
    win._on_redo()
    assert len(win._page_stem_items.get(win._current_page_index, [])) == len(current_items)

    win.close()
