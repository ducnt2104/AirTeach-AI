import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _hplc_item in __init__
init_target = '        self._fabry_perot_item: Optional[QGraphicsItem] = None'
init_replace = '        self._fabry_perot_item: Optional[QGraphicsItem] = None\n        self._hplc_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_fabry_perot_item", None),'
rect_replace = '            getattr(self, "_fabry_perot_item", None),\n            getattr(self, "_hplc_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_hplc(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive HPLC & Van Deemter Column Efficiency simulator."""
        if self._hplc_item is None:
            from src.ui.interactive_hplc import InteractiveHPLCItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._hplc_item = InteractiveHPLCItem(pos=spawn_pos)
            self.addItem(self._hplc_item)
        else:
            self._hplc_item.setVisible(visible)
            if pos is not None:
                self._hplc_item.setPos(pos[0], pos[1])
        return self._hplc_item if visible else None

    def toggle_hplc(self) -> bool:
        """Toggle interactive HPLC & Van Deemter Column Efficiency simulator visibility."""
        if self._hplc_item is None or not self._hplc_item.isVisible():
            self.show_hplc(True)
            return True
        else:
            self.show_hplc(False)
            return False

    def is_hplc_visible(self) -> bool:
        """Check if interactive HPLC simulator is currently visible."""
        return self._hplc_item is not None and self._hplc_item.isVisible()

    def get_hplc(self) -> Optional[QGraphicsItem]:
        """Get interactive HPLC simulator item if visible."""
        if self._hplc_item is not None and self._hplc_item.isVisible():
            return self._hplc_item
        return None

    def get_hplc_item(self) -> Optional[QGraphicsItem]:
        """Get interactive HPLC simulator item regardless of visibility."""
        return self._hplc_item
'''

method_target = '    def get_fabry_perot_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive Fabry-Pérot Interferometer simulator item regardless of visibility."""\n        return self._fabry_perot_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired HPLC into lesson_board.py!')
