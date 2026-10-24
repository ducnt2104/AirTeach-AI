import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _photoelectric_item in __init__
init_target = '        self._hplc_item: Optional[QGraphicsItem] = None'
init_replace = '        self._hplc_item: Optional[QGraphicsItem] = None\n        self._photoelectric_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_hplc_item", None),'
rect_replace = '            getattr(self, "_hplc_item", None),\n            getattr(self, "_photoelectric_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_photoelectric(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Einstein Photoelectric Effect simulator."""
        if self._photoelectric_item is None:
            from src.ui.interactive_photoelectric_effect import InteractivePhotoelectricEffectItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._photoelectric_item = InteractivePhotoelectricEffectItem(pos=spawn_pos)
            self.addItem(self._photoelectric_item)
        else:
            self._photoelectric_item.setVisible(visible)
            if pos is not None:
                self._photoelectric_item.setPos(pos[0], pos[1])
        return self._photoelectric_item if visible else None

    def toggle_photoelectric(self) -> bool:
        """Toggle interactive Einstein Photoelectric Effect simulator visibility."""
        if self._photoelectric_item is None or not self._photoelectric_item.isVisible():
            self.show_photoelectric(True)
            return True
        else:
            self.show_photoelectric(False)
            return False

    def is_photoelectric_visible(self) -> bool:
        """Check if interactive Photoelectric Effect simulator is currently visible."""
        return self._photoelectric_item is not None and self._photoelectric_item.isVisible()

    def get_photoelectric(self) -> Optional[QGraphicsItem]:
        """Get interactive Photoelectric Effect simulator item if visible."""
        if self._photoelectric_item is not None and self._photoelectric_item.isVisible():
            return self._photoelectric_item
        return None

    def get_photoelectric_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Photoelectric Effect simulator item regardless of visibility."""
        return self._photoelectric_item
'''

method_target = '    def get_hplc_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive HPLC simulator item regardless of visibility."""\n        return self._hplc_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired Photoelectric Effect into lesson_board.py!')
