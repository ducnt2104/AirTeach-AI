import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _thin_lens_item in __init__
init_target = '        self._pcr_item: Optional[QGraphicsItem] = None'
init_replace = '        self._pcr_item: Optional[QGraphicsItem] = None\n        self._thin_lens_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_pcr_item", None),'
rect_replace = '            getattr(self, "_pcr_item", None),\n            getattr(self, "_thin_lens_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_thin_lens(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Thin Lens Systems & Geometrical Optics simulator."""
        if self._thin_lens_item is None:
            from src.ui.interactive_thin_lens_system import InteractiveThinLensSystemItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._thin_lens_item = InteractiveThinLensSystemItem(pos=spawn_pos)
            self.addItem(self._thin_lens_item)
        else:
            self._thin_lens_item.setVisible(visible)
            if pos is not None:
                self._thin_lens_item.setPos(pos[0], pos[1])
        return self._thin_lens_item if visible else None

    def toggle_thin_lens(self) -> bool:
        """Toggle interactive Thin Lens Systems & Geometrical Optics simulator visibility."""
        if self._thin_lens_item is None or not self._thin_lens_item.isVisible():
            self.show_thin_lens(True)
            return True
        else:
            self.show_thin_lens(False)
            return False

    def is_thin_lens_visible(self) -> bool:
        """Check if interactive Thin Lens simulator is currently visible."""
        return self._thin_lens_item is not None and self._thin_lens_item.isVisible()

    def get_thin_lens(self) -> Optional[QGraphicsItem]:
        """Get interactive Thin Lens simulator item if visible."""
        if self._thin_lens_item is not None and self._thin_lens_item.isVisible():
            return self._thin_lens_item
        return None

    def get_thin_lens_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Thin Lens simulator item regardless of visibility."""
        return self._thin_lens_item
'''

method_target = '    def get_pcr_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive PCR simulator item regardless of visibility."""\n        return self._pcr_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired Thin Lens into lesson_board.py!')
