"""
Update src/ui/lesson_board.py to add InteractiveThinLensBenchItem support.
"""

def main():
    with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Attribute in __init__
    target_init = '        # Interactive AC Series RLC Circuit & Fresnel Phasor Simulator\n        self._rlc_circuit_item: Optional[QGraphicsItem] = None'
    rep_init = target_init + '\n\n        # Interactive Geometric Optics Thin Lens & Optical Bench Simulator\n        self._thin_lens_item: Optional[QGraphicsItem] = None'

    assert target_init in content, "target_init not found"
    content = content.replace(target_init, rep_init, 1)

    # 2. Methods
    target_methods = '''    def get_rlc_circuit_item(self) -> Optional[QGraphicsItem]:
        """Return interactive RLC circuit instance if created."""
        return self._rlc_circuit_item'''

    lens_methods = '''    def get_rlc_circuit_item(self) -> Optional[QGraphicsItem]:
        """Return interactive RLC circuit instance if created."""
        return self._rlc_circuit_item

    def show_thin_lens(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Geometric Optics Thin Lens & Optical Bench simulator."""
        if self._thin_lens_item is None:
            from src.ui.interactive_thin_lens import InteractiveThinLensBenchItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._thin_lens_item = InteractiveThinLensBenchItem(pos=spawn_pos)
            self.addItem(self._thin_lens_item)

        self._thin_lens_item.setVisible(visible)
        if visible and pos is not None:
            self._thin_lens_item.setPos(pos[0], pos[1])
        self.update()
        return self._thin_lens_item if visible else None

    def toggle_thin_lens(self) -> bool:
        """Toggle interactive thin lens simulator visibility; returns new state."""
        if self._thin_lens_item is None or not self._thin_lens_item.isVisible():
            self.show_thin_lens(True)
            return True
        else:
            self.show_thin_lens(False)
            return False

    def is_thin_lens_visible(self) -> bool:
        """Return True if interactive thin lens simulator is active on board."""
        return self._thin_lens_item is not None and self._thin_lens_item.isVisible()

    def get_thin_lens(self) -> Optional[QGraphicsItem]:
        """Return interactive thin lens instance if active and visible."""
        if self._thin_lens_item is not None and self._thin_lens_item.isVisible():
            return self._thin_lens_item
        return None

    def get_thin_lens_item(self) -> Optional[QGraphicsItem]:
        """Return interactive thin lens instance if created."""
        return self._thin_lens_item'''

    assert target_methods in content, "target_methods not found"
    content = content.replace(target_methods, lens_methods, 1)

    # 3. Excluded set
    target_excl = 'getattr(self, "_rlc_circuit_item", None),'
    rep_excl = target_excl + '\n            getattr(self, "_thin_lens_item", None),'

    assert target_excl in content, "target_excl not found"
    content = content.replace(target_excl, rep_excl, 1)

    with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated src/ui/lesson_board.py!")

if __name__ == '__main__':
    main()
