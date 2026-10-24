import sys

with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _em_wave_item in __init__
init_target = '        self._phase_diagram_item: Optional[QGraphicsItem] = None'
init_replace = '        self._phase_diagram_item: Optional[QGraphicsItem] = None\n        self._em_wave_item: Optional[QGraphicsItem] = None'

if init_target not in text:
    print('Failed to find init_target in lesson_board.py')
    sys.exit(1)

text = text.replace(init_target, init_replace, 1)

# 2. Add getattr in get_items_in_rect
rect_target = '            getattr(self, "_phase_diagram_item", None),'
rect_replace = '            getattr(self, "_phase_diagram_item", None),\n            getattr(self, "_em_wave_item", None),'

if rect_target in text:
    text = text.replace(rect_target, rect_replace, 1)

# 3. Add lifecycle methods
methods = '''

    def show_em_wave(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Maxwell's Electromagnetic Waves simulator."""
        if self._em_wave_item is None:
            from src.ui.interactive_em_wave import InteractiveElectromagneticWaveItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._em_wave_item = InteractiveElectromagneticWaveItem(pos=spawn_pos)
            self.addItem(self._em_wave_item)
        else:
            self._em_wave_item.setVisible(visible)
            if pos is not None:
                self._em_wave_item.setPos(pos[0], pos[1])
        return self._em_wave_item if visible else None

    def toggle_em_wave(self) -> bool:
        """Toggle interactive Maxwell's Electromagnetic Waves simulator visibility."""
        if self._em_wave_item is None or not self._em_wave_item.isVisible():
            self.show_em_wave(True)
            return True
        else:
            self.show_em_wave(False)
            return False

    def is_em_wave_visible(self) -> bool:
        """Check if interactive Maxwell's Electromagnetic Waves simulator is currently visible."""
        return self._em_wave_item is not None and self._em_wave_item.isVisible()

    def get_em_wave(self) -> Optional[QGraphicsItem]:
        """Get interactive Maxwell's Electromagnetic Waves simulator item if visible."""
        if self._em_wave_item is not None and self._em_wave_item.isVisible():
            return self._em_wave_item
        return None

    def get_em_wave_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Maxwell's Electromagnetic Waves simulator item regardless of visibility."""
        return self._em_wave_item
'''

method_target = '    def get_phase_diagram_item(self) -> Optional[QGraphicsItem]:\n        """Get interactive Thermodynamics P-T Phase Diagram simulator item regardless of visibility."""\n        return self._phase_diagram_item'
if method_target not in text:
    print('Failed to find method_target in lesson_board.py')
    sys.exit(1)

text = text.replace(method_target, method_target + methods, 1)

with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired EM Wave into lesson_board.py!')
