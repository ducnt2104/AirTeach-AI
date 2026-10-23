"""
Unit Test Suite for Interactive DNA Double Helix Replication & Base Pairing Simulator
======================================================================================
Tests:
  - Sequence initialization and complementary base pairing (Watson-Crick rules: A=T, G≡C).
  - Hydrogen bond counting (2 for A-T, 3 for G-C).
  - Replication fork unzipping progress [0.0 to 1.0].
  - Semi-conservative replication daughter strand synthesis.
  - Automatic unzipping animation timer progression.
  - Point mutation simulation (Đột biến điểm) and wild-type sequence reset.
  - Mouse dragging along fork and touchless air gesture interaction (Air Drag & Air Tap).
  - Stamping permanent vector diagrams to LessonBoard and MainWindow integration.
"""
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_dna_replication import (
    InteractiveDnaReplicationItem,
    COMPLEMENT_MAP,
    DEFAULT_PARENT_SEQUENCE,
)
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication is running for Qt GUI objects."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_dna_initialization_and_properties(qapp):
    """Test default sequence, complementary strand, and bounding box."""
    item = InteractiveDnaReplicationItem(pos=(220.0, 160.0))
    assert item.pos() == QPointF(220.0, 160.0)
    assert item.boundingRect().width() >= 660.0
    assert item.boundingRect().height() >= 430.0
    assert item.sequence_length() == len(DEFAULT_PARENT_SEQUENCE)
    assert not item.has_mutation()
    assert not item.is_auto_replicating()
    assert item.unzip_progress() == pytest.approx(0.40, abs=1e-3)

    # Check complementary pairs
    for i in range(item.sequence_length()):
        b1, b2 = item.base_pair_at(i)
        assert COMPLEMENT_MAP[b1] == b2

    item.close()


def test_watson_crick_base_pairing_and_hydrogen_bonds(qapp):
    """Verify Watson-Crick hydrogen bonds: A-T has 2, G-C has 3."""
    custom_seq = ["A", "T", "G", "C"]
    item = InteractiveDnaReplicationItem(parent_sequence=custom_seq)
    assert item.sequence_length() == 4

    # A-T (2) + T-A (2) + G-C (3) + C-G (3) = 10 bonds
    assert item.count_hydrogen_bonds() == 10

    # Default sequence: 6 A-T pairs (12 bonds) + 6 G-C pairs (18 bonds) = 30 bonds
    item_default = InteractiveDnaReplicationItem()
    assert item_default.count_hydrogen_bonds() == 30

    item.close()
    item_default.close()


def test_unzip_progress_and_fork_clamping(qapp):
    """Test unzipping progress updates and boundary clamping [0.0, 1.0]."""
    item = InteractiveDnaReplicationItem()

    item.set_unzip_progress(0.75)
    assert item.unzip_progress() == pytest.approx(0.75, abs=1e-3)

    # Clamping
    item.set_unzip_progress(1.8)
    assert item.unzip_progress() == pytest.approx(1.0, abs=1e-3)

    item.set_unzip_progress(-0.4)
    assert item.unzip_progress() == pytest.approx(0.0, abs=1e-3)

    item.close()


def test_auto_replication_animation(qapp):
    """Test auto-replication animation toggle and timer callback."""
    item = InteractiveDnaReplicationItem()
    assert not item.is_auto_replicating()

    # Toggle play
    started = item.toggle_auto_replication()
    assert started is True
    assert item.is_auto_replicating() is True

    # Advance tick
    old_p = item.unzip_progress()
    item._on_replicate_tick()
    assert item.unzip_progress() > old_p

    # Toggle stop
    stopped = item.toggle_auto_replication()
    assert stopped is False
    assert item.is_auto_replicating() is False

    item.close()


def test_point_mutation_and_reset(qapp):
    """Test point mutation trigger and sequence restoration."""
    item = InteractiveDnaReplicationItem()
    assert not item.has_mutation()

    # Trigger mutation
    item.trigger_point_mutation()
    assert item.has_mutation() is True
    assert item._mutation_idx == 5
    b1, b2 = item.base_pair_at(5)
    assert b1 == "C" and b2 == "G"

    # Reset
    item.reset_sequence()
    assert item.has_mutation() is False
    assert item.sequence_length() == len(DEFAULT_PARENT_SEQUENCE)
    assert item.base_pair_at(5) == (DEFAULT_PARENT_SEQUENCE[5], COMPLEMENT_MAP[DEFAULT_PARENT_SEQUENCE[5]])

    item.close()


def test_touchless_air_gestures(qapp):
    """Test touchless air drag (unzipping fork) and air tap (controls)."""
    scene = QGraphicsScene()
    item = InteractiveDnaReplicationItem(pos=(100.0, 100.0))
    scene.addItem(item)

    # Hover
    item.set_air_hover_pos(QPointF(150.0, 150.0))
    assert item._air_hover_pos is not None
    item.set_air_hover_pos(None)
    assert item._air_hover_pos is None

    # Air tap on mutate button
    btn_mut_scene = item.mapToScene(item._rect_btn_mutate.center())
    tapped = item.handle_air_tap(btn_mut_scene)
    assert tapped is True
    assert item.has_mutation() is True

    # Air tap on reset button
    btn_res_scene = item.mapToScene(item._rect_btn_reset.center())
    tapped_res = item.handle_air_tap(btn_res_scene)
    assert tapped_res is True
    assert item.has_mutation() is False

    # Air tap on play/pause auto-replication
    btn_play_scene = item.mapToScene(item._rect_btn_play.center())
    tapped_play = item.handle_air_tap(btn_play_scene)
    assert tapped_play is True
    assert item.is_auto_replicating() is True

    # Air drag to move replication fork
    fork_drag_scene = item.mapToScene(QPointF(250.0, item._height * 0.5))
    dragged = item.handle_air_drag(fork_drag_scene)
    assert dragged is True

    item.close()


def test_stamp_to_canvas_and_lessonboard(qapp):
    """Test stamping vector DNA double helix diagram onto LessonBoard."""
    board = LessonBoard()
    item = InteractiveDnaReplicationItem(pos=(300.0, 300.0))
    board.addItem(item)

    stamped_items = item.stamp_to_canvas(board)
    assert len(stamped_items) >= 2  # helical backbones, base pairs, labels
    for itm in stamped_items:
        assert itm.scene() == board

    item.close()


def test_lessonboard_and_main_window_integration(qapp):
    """Test LessonBoard lifecycle methods and MainWindow action toggles."""
    win = MainWindow()
    board = win._scene
    assert not board.is_dna_replication_visible()

    # Show item
    board.show_dna_replication(True)
    assert board.is_dna_replication_visible()
    item = board.get_dna_replication_item()
    assert item is not None
    assert item.isVisible()

    # Hide and Toggle
    board.show_dna_replication(False)
    assert not board.is_dna_replication_visible()

    is_vis = board.toggle_dna_replication()
    assert is_vis is True
    assert board.is_dna_replication_visible()

    # MainWindow slot
    win._on_toggle_dna_replication()
    assert not board.is_dna_replication_visible()

    item.close()
    win.close()
