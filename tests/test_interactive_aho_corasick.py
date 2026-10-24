"""
Unit tests for InteractiveAhoCorasickItem and InformaticsTools.create_aho_corasick_diagram.
Verifies multi-pattern string matching automaton, BFS failure transitions, dictionary suffix propagation,
step scanner, UI gestures, presets, and lesson board integration.
"""

import sys
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_aho_corasick import InteractiveAhoCorasickItem, PRESETS
from src.ui.stem_tools import InformaticsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_aho_corasick_initialization(qapp):
    item = InteractiveAhoCorasickItem()
    assert item.active_preset.code == "CLASSIC_BELL_LABS_HE_SHE_HIS_HERS"
    assert "he" in item.patterns
    assert "she" in item.patterns
    assert item.sample_text == "ushers"
    assert item.nodes_count >= 9
    assert item.boundingRect().width() == InteractiveAhoCorasickItem.CARD_W
    assert item.boundingRect().height() == InteractiveAhoCorasickItem.CARD_H


def test_aho_corasick_automaton_structure(qapp):
    item = InteractiveAhoCorasickItem()
    # Bell Labs patterns: he, she, his, hers
    # Root is node 0
    assert 0 in item._nodes
    root = item._nodes[0]
    assert root.fail_state == 0
    assert len(root.output_patterns) == 0

    # 'h' and 's' are transitions from root
    assert 'h' in root.children
    assert 's' in root.children

    # Check failure links exist for all nodes
    for nid, node in item._nodes.items():
        assert node.fail_state is not None
        assert node.fail_state in item._nodes
        # Depth 1 nodes must fail to root (0)
        if node.depth == 1:
            assert node.fail_state == 0


def test_aho_corasick_suffix_output_propagation(qapp):
    item = InteractiveAhoCorasickItem()
    # In "ushers", at the end of "she", both "she" and "he" should be matched because failure of "she" leads to "he"
    # Find the node representing "she"
    curr = 0
    for ch in "she":
        curr = item._nodes[curr].children[ch]
    she_node = item._nodes[curr]
    assert "she" in she_node.output_patterns
    assert "he" in she_node.output_patterns, "Suffix failure link must propagate 'he' into outputs of 'she'"


def test_aho_corasick_presets(qapp):
    item = InteractiveAhoCorasickItem()
    expected_presets = [
        "CLASSIC_BELL_LABS_HE_SHE_HIS_HERS",
        "DNA_BIOINFORMATICS_RESTRICTION_SITES",
        "NATURAL_LANGUAGE_KEYWORD_FILTER",
        "OVERLAPPING_PREFIX_SUFFIX_STRESS",
        "CYBERSECURITY_SNORT_IDS_SIGNATURES",
    ]
    for p_key in expected_presets:
        item.load_preset(p_key)
        assert item.active_preset.code == p_key
        assert len(item.patterns) > 0
        assert len(item.sample_text) > 0
        assert item.nodes_count > 0


def test_aho_corasick_custom_patterns_and_text(qapp):
    item = InteractiveAhoCorasickItem()
    custom_patterns = ["nano", "robot", "tech"]
    custom_text = "nanorobottechnology"
    item.set_custom_patterns_and_text(custom_patterns, custom_text)

    assert item.patterns == custom_patterns
    assert item.sample_text == custom_text
    assert item.active_preset.code == "CUSTOM"
    assert item.nodes_count > len(custom_patterns)


def test_aho_corasick_step_scanning_and_signal(qapp):
    item = InteractiveAhoCorasickItem()
    item.load_preset("CLASSIC_BELL_LABS_HE_SHE_HIS_HERS")
    item.reset_scan()

    matches_detected = []
    item.match_found.connect(lambda pat, idx: matches_detected.append((pat, idx)))

    # Step through all characters of "ushers"
    for _ in range(len(item.sample_text)):
        item.step_text_scan()

    assert len(matches_detected) >= 2
    matched_words = [m[0] for m in matches_detected]
    assert "she" in matched_words
    assert "he" in matched_words
    assert "hers" in matched_words


def test_aho_corasick_reset_scan(qapp):
    item = InteractiveAhoCorasickItem()
    for _ in range(3):
        item.step_text_scan()
    assert item._text_scan_idx == 3
    assert len(item._matches_history) >= 0

    item.reset_scan()
    assert item._text_scan_idx == 0
    assert item._current_state_id == 0
    assert len(item._matches_history) == 0


def test_aho_corasick_animation_toggle(qapp):
    item = InteractiveAhoCorasickItem()
    # Initially running is True
    assert item._is_running
    item.toggle_animation()
    assert not item._is_running
    item.toggle_animation()
    assert item._is_running


def test_aho_corasick_ui_gestures(qapp):
    item = InteractiveAhoCorasickItem(pos=(100.0, 150.0))
    initial_pos = item.pos()
    item.handle_air_drag(QPointF(25.0, -15.0))
    assert item.pos().x() == initial_pos.x() + 25.0
    assert item.pos().y() == initial_pos.y() - 15.0

    # Test air pinch on button
    btn_step_rect = item._buttons.get("btn_step")
    assert btn_step_rect is not None
    initial_idx = item._text_scan_idx
    scene_target = item.mapToScene(btn_step_rect.center())
    handled = item.handle_air_pinch(scene_target)
    assert handled
    assert item._text_scan_idx == initial_idx + 1


def test_aho_corasick_button_interactions(qapp):
    item = InteractiveAhoCorasickItem()

    # Step button
    initial_idx = item._text_scan_idx
    item._handle_button_click("btn_step")
    assert item._text_scan_idx == initial_idx + 1

    # Play button
    item._handle_button_click("btn_play")
    assert not item._is_running
    item._handle_button_click("btn_play")
    assert item._is_running

    # Fail links toggle
    initial_fail_vis = item._show_failure_links
    item._handle_button_click("btn_toggle_fail")
    assert item._show_failure_links != initial_fail_vis


def test_aho_corasick_diagram_generation(qapp):
    items = InformaticsTools.create_aho_corasick_diagram(
        center=(300.0, 200.0),
        width=660.0,
        height=360.0,
        preset_code="CLASSIC_BELL_LABS_HE_SHE_HIS_HERS",
    )
    assert len(items) >= 10
    # First item is background
    bg = items[0]
    assert bg.rect().width() == 660.0
    assert bg.rect().height() == 360.0


def test_aho_corasick_lesson_board_integration(qapp):
    board = LessonBoard()
    assert not board.is_aho_corasick_visible()
    assert board.get_aho_corasick() is None

    # Show item
    item = board.show_aho_corasick(True, pos=(120.0, 80.0))
    assert item is not None
    assert board.is_aho_corasick_visible()
    assert board.get_aho_corasick() == item
    assert board.get_aho_corasick_item() == item

    # Toggle off
    res = board.toggle_aho_corasick()
    assert not res
    assert not board.is_aho_corasick_visible()
    assert board.get_aho_corasick() is None
    assert board.get_aho_corasick_item() is not None

    # Toggle on
    res2 = board.toggle_aho_corasick()
    assert res2
    assert board.is_aho_corasick_visible()
