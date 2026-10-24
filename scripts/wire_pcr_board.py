import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _pcr_item in __init__
init_target = '        self._photoelectric_item: Optional[QGraphicsItem] = None'
init_replace = '        self._photoelectric_item: Optional[QGraphicsItem] = None\n        self._pcr_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_photoelectric_item", None),'
rect_replace = '            getattr(self, "_photoelectric_item", None),\n            getattr(self, "_pcr_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_pcr(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Polymerase Chain Reaction (PCR) simulator."""
        if self._pcr_item is None:
            from src.ui.interactive_pcr import InteractivePCRItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._pcr_item = InteractivePCRItem(pos=spawn_pos)
            self.addItem(self._pcr_item)
        else:
            self._pcr_item.setVisible(visible)
            if pos is not None:
                self._pcr_item.setPos(pos[0], pos[1])
        return self._pcr_item if visible else None

    def toggle_pcr(self) -> bool:
        """Toggle interactive Polymerase Chain Reaction (PCR) simulator visibility."""
        if self._pcr_item is None or not self._pcr_item.isVisible():
            self.show_pcr(True)
            return True
        else:
            self.show_pcr(False)
            return False

    def is_pcr_visible(self) -> bool:
        """Check if interactive PCR simulator is currently visible."""
        return self._pcr_item is not None and self._pcr_item.isVisible()

    def get_pcr(self) -> Optional[QGraphicsItem]:
        """Get interactive PCR simulator item if visible."""
        if self._pcr_item is not None and self._pcr_item.isVisible():
            return self._pcr_item
        return None

    def get_pcr_item(self) -> Optional[QGraphicsItem]:
        """Get interactive PCR simulator item regardless of visibility."""
        return self._pcr_item
'''

method_target = '    def get_photoelectric_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive Photoelectric Effect simulator item regardless of visibility."""\n        return self._photoelectric_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired PCR into lesson_board.py!')
