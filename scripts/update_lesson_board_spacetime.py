"""
Update src/ui/lesson_board.py to add InteractiveSpacetimeRelativityItem support.
"""

def main():
    with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Attribute in __init__
    target_init = '        # Interactive Physics & Astronomy Gravitational N-Body Planetary Orbit & Lagrange Points Simulator\n        self._gravitational_orbits_item: Optional[QGraphicsItem] = None'
    rep_init = target_init + '\n\n        # Interactive Special Relativity Minkowski Spacetime & Lorentz Boost Simulator\n        self._spacetime_relativity_item: Optional[QGraphicsItem] = None'

    assert target_init in content, "target_init not found"
    content = content.replace(target_init, rep_init, 1)

    # 2. Methods
    target_methods = '''    def get_gravitational_orbits_item(self) -> Optional[QGraphicsItem]:
        """Return interactive gravitational orbits instance if created."""
        return self._gravitational_orbits_item'''

    relativity_methods = '''    def get_gravitational_orbits_item(self) -> Optional[QGraphicsItem]:
        """Return interactive gravitational orbits instance if created."""
        return self._gravitational_orbits_item

    def show_spacetime_relativity(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Special Relativity Minkowski Spacetime simulator."""
        if self._spacetime_relativity_item is None:
            from src.ui.interactive_spacetime_relativity import InteractiveSpacetimeRelativityItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._spacetime_relativity_item = InteractiveSpacetimeRelativityItem(pos=spawn_pos)
            self.addItem(self._spacetime_relativity_item)

        self._spacetime_relativity_item.setVisible(visible)
        if visible and pos is not None:
            self._spacetime_relativity_item.setPos(pos[0], pos[1])
        self.update()
        return self._spacetime_relativity_item if visible else None

    def toggle_spacetime_relativity(self) -> bool:
        """Toggle interactive spacetime relativity simulator visibility; returns new state."""
        if self._spacetime_relativity_item is None or not self._spacetime_relativity_item.isVisible():
            self.show_spacetime_relativity(True)
            return True
        else:
            self.show_spacetime_relativity(False)
            return False

    def is_spacetime_relativity_visible(self) -> bool:
        """Return True if interactive spacetime relativity simulator is active on board."""
        return self._spacetime_relativity_item is not None and self._spacetime_relativity_item.isVisible()

    def get_spacetime_relativity(self) -> Optional[QGraphicsItem]:
        """Return interactive spacetime relativity instance if active and visible."""
        if self._spacetime_relativity_item is not None and self._spacetime_relativity_item.isVisible():
            return self._spacetime_relativity_item
        return None

    def get_spacetime_relativity_item(self) -> Optional[QGraphicsItem]:
        """Return interactive spacetime relativity instance if created."""
        return self._spacetime_relativity_item'''

    assert target_methods in content, "target_methods not found"
    content = content.replace(target_methods, relativity_methods, 1)

    # 3. Excluded items set
    target_excl = 'getattr(self, "_gravitational_orbits_item", None),'
    rep_excl = target_excl + '\n            getattr(self, "_spacetime_relativity_item", None),'

    assert target_excl in content, "target_excl not found"
    content = content.replace(target_excl, rep_excl, 1)

    with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated src/ui/lesson_board.py!")

if __name__ == '__main__':
    main()
