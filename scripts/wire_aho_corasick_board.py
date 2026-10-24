import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _aho_corasick_item in __init__
init_target = '        self._cyclotron_item: Optional[QGraphicsItem] = None'
init_replace = '        self._cyclotron_item: Optional[QGraphicsItem] = None\n        self._aho_corasick_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_cyclotron_item", None),'
rect_replace = '            getattr(self, "_cyclotron_item", None),\n            getattr(self, "_aho_corasick_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)
else:
    # Look for another item in excluded
    rect_target2 = '            getattr(self, "_action_potential_item", None),'
    if rect_target2 in text:
        text = text.replace(rect_target2, rect_target2 + '\n            getattr(self, "_aho_corasick_item", None),', 1)
    else:
        print('Warning: could not find cyclotron or action_potential in get_items_in_rect, will check other')

# 3. Add lifecycle methods
methods = '''

    def show_aho_corasick(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Aho-Corasick Multi-Pattern Automaton simulator."""
        if self._aho_corasick_item is None:
            from src.ui.interactive_aho_corasick import InteractiveAhoCorasickItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._aho_corasick_item = InteractiveAhoCorasickItem(pos=spawn_pos)
            self.addItem(self._aho_corasick_item)
        else:
            self._aho_corasick_item.setVisible(visible)
            if pos is not None:
                self._aho_corasick_item.setPos(pos[0], pos[1])
        return self._aho_corasick_item if visible else None

    def toggle_aho_corasick(self) -> bool:
        """Toggle interactive Aho-Corasick Multi-Pattern Automaton simulator visibility."""
        if self._aho_corasick_item is None or not self._aho_corasick_item.isVisible():
            self.show_aho_corasick(True)
            return True
        else:
            self.show_aho_corasick(False)
            return False

    def is_aho_corasick_visible(self) -> bool:
        """Check if interactive Aho-Corasick Multi-Pattern Automaton simulator is currently visible."""
        return self._aho_corasick_item is not None and self._aho_corasick_item.isVisible()

    def get_aho_corasick(self) -> Optional[QGraphicsItem]:
        """Get interactive Aho-Corasick Multi-Pattern Automaton simulator item if visible."""
        if self._aho_corasick_item is not None and self._aho_corasick_item.isVisible():
            return self._aho_corasick_item
        return None

    def get_aho_corasick_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Aho-Corasick Multi-Pattern Automaton simulator item regardless of visibility."""
        return self._aho_corasick_item
'''

method_target = '    def get_cyclotron_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive Cyclotron Particle Accelerator simulator item regardless of visibility."""\n        return self._cyclotron_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired Aho-Corasick into lesson_board.py!')
