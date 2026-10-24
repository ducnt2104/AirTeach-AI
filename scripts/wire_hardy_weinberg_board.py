import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _hardy_weinberg_item in __init__
init_target = '        self._em_wave_item: Optional[QGraphicsItem] = None'
init_replace = '        self._em_wave_item: Optional[QGraphicsItem] = None\n        self._hardy_weinberg_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_em_wave_item", None),'
rect_replace = '            getattr(self, "_em_wave_item", None),\n            getattr(self, "_hardy_weinberg_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_hardy_weinberg(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Hardy-Weinberg Population Genetics simulator."""
        if self._hardy_weinberg_item is None:
            from src.ui.interactive_hardy_weinberg import InteractiveHardyWeinbergItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._hardy_weinberg_item = InteractiveHardyWeinbergItem(pos=spawn_pos)
            self.addItem(self._hardy_weinberg_item)
        else:
            self._hardy_weinberg_item.setVisible(visible)
            if pos is not None:
                self._hardy_weinberg_item.setPos(pos[0], pos[1])
        return self._hardy_weinberg_item if visible else None

    def toggle_hardy_weinberg(self) -> bool:
        """Toggle interactive Hardy-Weinberg Population Genetics simulator visibility."""
        if self._hardy_weinberg_item is None or not self._hardy_weinberg_item.isVisible():
            self.show_hardy_weinberg(True)
            return True
        else:
            self.show_hardy_weinberg(False)
            return False

    def is_hardy_weinberg_visible(self) -> bool:
        """Check if interactive Hardy-Weinberg Population Genetics simulator is currently visible."""
        return self._hardy_weinberg_item is not None and self._hardy_weinberg_item.isVisible()

    def get_hardy_weinberg(self) -> Optional[QGraphicsItem]:
        """Get interactive Hardy-Weinberg Population Genetics simulator item if visible."""
        if self._hardy_weinberg_item is not None and self._hardy_weinberg_item.isVisible():
            return self._hardy_weinberg_item
        return None

    def get_hardy_weinberg_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Hardy-Weinberg Population Genetics simulator item regardless of visibility."""
        return self._hardy_weinberg_item
'''

method_target = '    def get_em_wave_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive Maxwell\'s Electromagnetic Waves simulator item regardless of visibility."""\n        return self._em_wave_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired Hardy-Weinberg into lesson_board.py!')
