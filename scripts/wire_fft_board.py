import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _fft_item in __init__
init_target = '        self._hardy_weinberg_item: Optional[QGraphicsItem] = None'
init_replace = '        self._hardy_weinberg_item: Optional[QGraphicsItem] = None\n        self._fft_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_hardy_weinberg_item", None),'
rect_replace = '            getattr(self, "_hardy_weinberg_item", None),\n            getattr(self, "_fft_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_fft(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Cooley-Tukey FFT & Spectrum Analyzer simulator."""
        if self._fft_item is None:
            from src.ui.interactive_fft import InteractiveFFTItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._fft_item = InteractiveFFTItem(pos=spawn_pos)
            self.addItem(self._fft_item)
        else:
            self._fft_item.setVisible(visible)
            if pos is not None:
                self._fft_item.setPos(pos[0], pos[1])
        return self._fft_item if visible else None

    def toggle_fft(self) -> bool:
        """Toggle interactive Cooley-Tukey FFT & Spectrum Analyzer simulator visibility."""
        if self._fft_item is None or not self._fft_item.isVisible():
            self.show_fft(True)
            return True
        else:
            self.show_fft(False)
            return False

    def is_fft_visible(self) -> bool:
        """Check if interactive Cooley-Tukey FFT & Spectrum Analyzer simulator is currently visible."""
        return self._fft_item is not None and self._fft_item.isVisible()

    def get_fft(self) -> Optional[QGraphicsItem]:
        """Get interactive Cooley-Tukey FFT & Spectrum Analyzer simulator item if visible."""
        if self._fft_item is not None and self._fft_item.isVisible():
            return self._fft_item
        return None

    def get_fft_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Cooley-Tukey FFT & Spectrum Analyzer simulator item regardless of visibility."""
        return self._fft_item
'''

method_target = '    def get_hardy_weinberg_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive Hardy-Weinberg Population Genetics simulator item regardless of visibility."""\n        return self._hardy_weinberg_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired FFT into lesson_board.py!')
