"""
Tests for Interactive Huffman Coding & Shannon Entropy Simulator.
Verifies authentic Huffman prefix-free tree generation, Shannon Source Coding Theorem,
entropy calculations, prefix-free invariants, UI gestures, rendering, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_huffman_coding import (
    PRESETS,
    HuffmanNode,
    HuffmanPreset,
    InteractiveHuffmanCodingItem,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import InformaticsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestHuffmanInformationTheory:
    """Authentic mathematical and information theory verification."""

    def test_shannon_entropy_uniform_distribution(self, qapp):
        item = InteractiveHuffmanCodingItem()
        item.load_preset("UNIFORM_DISTRIBUTION_WORST_CASE")

        # 8 distinct characters: A, B, C, D, E, F, G, H with equal freq 1
        assert len(item.frequencies) == 8
        for p in item._probabilities.values():
            assert math.isclose(p, 1.0 / 8.0, rel_tol=1e-5)

        # Theoretical entropy H(X) = log2(8) = 3.0 bits/char
        entropy = item.compute_shannon_entropy()
        assert math.isclose(entropy, 3.0, rel_tol=1e-5)

        # Average code length should equal 3.0
        avg_len = item.compute_average_code_length()
        assert math.isclose(avg_len, 3.0, rel_tol=1e-5)

        # Efficiency should be 100%
        eta, saved_pct, orig_b, comp_b = item.compute_compression_metrics()
        assert math.isclose(eta, 1.0, rel_tol=1e-5)

    def test_shannon_source_coding_theorem_bound(self, qapp):
        """Shannon's theorem: H(X) <= L̄ < H(X) + 1."""
        item = InteractiveHuffmanCodingItem()
        for code in PRESETS.keys():
            item.load_preset(code)
            h = item.compute_shannon_entropy()
            l_bar = item.compute_average_code_length()

            # Upper and lower bounds
            assert l_bar >= h - 1e-9, f"Failed lower bound for {code}: L̄={l_bar}, H={h}"
            assert l_bar < h + 1.0, f"Failed upper bound for {code}: L̄={l_bar}, H={h}"

    def test_prefix_free_property_invariant(self, qapp):
        """Verifies that no Huffman code is a prefix of any other code."""
        item = InteractiveHuffmanCodingItem()
        for code in PRESETS.keys():
            item.load_preset(code)
            codes = list(item.huffman_codes.values())

            for i in range(len(codes)):
                for j in range(len(codes)):
                    if i != j:
                        c_i = codes[i]
                        c_j = codes[j]
                        assert not c_j.startswith(c_i), f"Code '{c_i}' is a prefix of '{c_j}' in preset {code}!"

    def test_abracadabra_skewed_frequencies(self, qapp):
        item = InteractiveHuffmanCodingItem()
        item.load_preset("CLASSIC_ABRACADABRA")

        # In "ABRACADABRA", 'A' has frequency 5 out of 11 chars
        assert item.frequencies['A'] == 5
        code_a = item.huffman_codes['A']
        # Most frequent symbol must have the shortest code
        min_len = min(len(c) for c in item.huffman_codes.values())
        assert len(code_a) == min_len

        # Total bits vs ASCII
        _, saved_pct, orig_b, comp_b = item.compute_compression_metrics()
        assert orig_b == 11 * 8  # 88 bits
        assert comp_b < orig_b
        assert saved_pct > 65.0  # Big space saving


class TestHuffmanPresetsAndControls:
    """Verifies presets, custom text, stepping, and gestures."""

    def test_all_presets_load(self, qapp):
        item = InteractiveHuffmanCodingItem()
        for code in PRESETS.keys():
            item.load_preset(code)
            assert item.active_preset.code == code
            assert len(item.huffman_codes) == len(item.frequencies)

    def test_custom_text_input(self, qapp):
        item = InteractiveHuffmanCodingItem()
        item.set_custom_text("MISSISSIPPI")
        assert len(item.text_data) == 11
        assert item.frequencies['I'] == 4
        assert item.frequencies['S'] == 4
        assert item.frequencies['P'] == 2
        assert item.frequencies['M'] == 1

        # Prefix free test on custom text
        codes = list(item.huffman_codes.values())
        for i, c1 in enumerate(codes):
            for j, c2 in enumerate(codes):
                if i != j:
                    assert not c2.startswith(c1)

    def test_tree_build_stepping(self, qapp):
        item = InteractiveHuffmanCodingItem()
        item.load_preset("CLASSIC_ABRACADABRA")

        assert len(item._build_steps) > 0
        item.reset_build_steps()
        assert item._current_step_idx == 0

        item.step_next_build()
        assert item._current_step_idx == 1

        item.toggle_animation()
        item.toggle_animation()

    def test_air_pinch_button_interactions(self, qapp):
        item = InteractiveHuffmanCodingItem()
        item.load_preset("CLASSIC_ABRACADABRA")

        # Render once to populate buttons dict
        img = QImage(960, 650, QImage.Format.Format_ARGB32)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()

        assert "btn_preset" in item._buttons
        assert "btn_stamp" in item._buttons

        # Air pinch on DNA text shortcut
        dna_btn_rect = item._buttons["btn_txt_dna"]
        handled = item.handle_air_pinch(item.mapToScene(dna_btn_rect.center()))
        assert handled is True
        assert item.active_preset.code == "DNA_GENOMICS_SEQUENCE"

        # Air pinch on stamp
        stamp_handled = item.handle_air_pinch(item.mapToScene(item._buttons["btn_stamp"].center()))
        assert stamp_handled is True

    def test_paint_rendering_all_presets(self, qapp):
        item = InteractiveHuffmanCodingItem()
        img = QImage(960, 650, QImage.Format.Format_ARGB32)

        for code in PRESETS.keys():
            item.load_preset(code)
            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle."""

    def test_informatics_tools_huffman_diagram(self, qapp):
        items = InformaticsTools.create_huffman_coding_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="CLASSIC_ABRACADABRA",
            text_sample="ABRACADABRA",
            entropy=2.04,
            avg_code_len=2.09,
            efficiency_pct=97.6,
            saved_storage_pct=73.9,
            codes={'A': '0', 'B': '10', 'R': '11', 'C': '100', 'D': '101'},
        )
        assert len(items) > 10
        assert any("HUFFMAN" in getattr(item, "toPlainText", lambda: "")() for item in items)

    def test_lesson_board_huffman_lifecycle(self, qapp):
        board = LessonBoard()

        # Initially not visible
        assert board.is_huffman_coding_visible() is False
        assert board.get_huffman_coding() is None

        # Show
        item = board.show_huffman_coding(True)
        assert item is not None
        assert board.is_huffman_coding_visible() is True
        assert board.get_huffman_coding() is item

        # Toggle hide
        new_state = board.toggle_huffman_coding()
        assert new_state is False
        assert board.is_huffman_coding_visible() is False

        # Toggle show
        new_state = board.toggle_huffman_coding()
        assert new_state is True
        assert board.is_huffman_coding_visible() is True

        # Ensure item is excluded from selection box in get_items_in_rect
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        selected_items = board.get_items_in_rect(selection_rect)
        assert item not in selected_items
