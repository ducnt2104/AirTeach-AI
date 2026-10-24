"""
Update src/ui/lesson_board.py to add InteractiveYoungInterferenceItem support.
"""

def main():
    with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Attribute in __init__
    target_init = '        # Interactive Special Relativity Minkowski Spacetime & Lorentz Boost Simulator\n        self._spacetime_relativity_item: Optional[QGraphicsItem] = None'
    rep_init = target_init + '\n\n        # Interactive Wave Optics & Young\'s Double-Slit Interference Simulator\n        self._young_interference_item: Optional[QGraphicsItem] = None'

    assert target_init in content, "target_init not found"
    content = content.replace(target_init, rep_init, 1)

    # 2. Methods
    target_methods = '''    def get_spacetime_relativity_item(self) -> Optional[QGraphicsItem]:
        """Return interactive spacetime relativity instance if created."""
        return self._spacetime_relativity_item'''

    young_methods = '''    def get_spacetime_relativity_item(self) -> Optional[QGraphicsItem]:
        """Return interactive spacetime relativity instance if created."""
        return self._spacetime_relativity_item

    def show_young_interference(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Wave Optics & Young\'s Double-Slit simulator."""
        if self._young_interference_item is None:
            from src.ui.interactive_young_interference import InteractiveYoungInterferenceItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._young_interference_item = InteractiveYoungInterferenceItem(pos=spawn_pos)
            self.addItem(self._young_interference_item)

        self._young_interference_item.setVisible(visible)
        if visible and pos is not None:
            self._young_interference_item.setPos(pos[0], pos[1])
        self.update()
        return self._young_interference_item if visible else None

    def toggle_young_interference(self) -> bool:
        """Toggle interactive wave optics interference simulator visibility; returns new state."""
        if self._young_interference_item is None or not self._young_interference_item.isVisible():
            self.show_young_interference(True)
            return True
        else:
            self.show_young_interference(False)
            return False

    def is_young_interference_visible(self) -> bool:
        """Return True if interactive wave optics interference simulator is active on board."""
        return self._young_interference_item is not None and self._young_interference_item.isVisible()

    def get_young_interference(self) -> Optional[QGraphicsItem]:
        """Return interactive wave optics interference instance if active and visible."""
        if self._young_interference_item is not None and self._young_interference_item.isVisible():
            return self._young_interference_item
        return None

    def get_young_interference_item(self) -> Optional[QGraphicsItem]:
        """Return interactive wave optics interference instance if created."""
        return self._young_interference_item'''

    assert target_methods in content, "target_methods not found"
    content = content.replace(target_methods, young_methods, 1)

    # 3. Excluded set
    target_excl = 'getattr(self, "_spacetime_relativity_item", None),'
    rep_excl = target_excl + '\n            getattr(self, "_young_interference_item", None),'

    assert target_excl in content, "target_excl not found"
    content = content.replace(target_excl, rep_excl, 1)

    with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated src/ui/lesson_board.py!")

if __name__ == '__main__':
    main()
