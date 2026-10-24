import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _fabry_perot_item in __init__
init_target = '        self._fft_item: Optional[QGraphicsItem] = None'
init_replace = '        self._fft_item: Optional[QGraphicsItem] = None\n        self._fabry_perot_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_fft_item", None),'
rect_replace = '            getattr(self, "_fft_item", None),\n            getattr(self, "_fabry_perot_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_fabry_perot(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Fabry-Pérot Interferometer & Cavity Resonator simulator."""
        if self._fabry_perot_item is None:
            from src.ui.interactive_fabry_perot import InteractiveFabryPerotItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._fabry_perot_item = InteractiveFabryPerotItem(pos=spawn_pos)
            self.addItem(self._fabry_perot_item)
        else:
            self._fabry_perot_item.setVisible(visible)
            if pos is not None:
                self._fabry_perot_item.setPos(pos[0], pos[1])
        return self._fabry_perot_item if visible else None

    def toggle_fabry_perot(self) -> bool:
        """Toggle interactive Fabry-Pérot Interferometer & Cavity Resonator simulator visibility."""
        if self._fabry_perot_item is None or not self._fabry_perot_item.isVisible():
            self.show_fabry_perot(True)
            return True
        else:
            self.show_fabry_perot(False)
            return False

    def is_fabry_perot_visible(self) -> bool:
        """Check if interactive Fabry-Pérot Interferometer simulator is currently visible."""
        return self._fabry_perot_item is not None and self._fabry_perot_item.isVisible()

    def get_fabry_perot(self) -> Optional[QGraphicsItem]:
        """Get interactive Fabry-Pérot Interferometer simulator item if visible."""
        if self._fabry_perot_item is not None and self._fabry_perot_item.isVisible():
            return self._fabry_perot_item
        return None

    def get_fabry_perot_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Fabry-Pérot Interferometer simulator item regardless of visibility."""
        return self._fabry_perot_item
'''

method_target = '    def get_fft_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive Cooley-Tukey FFT & Spectrum Analyzer simulator item regardless of visibility."""\n        return self._fft_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired Fabry-Pérot into lesson_board.py!')
