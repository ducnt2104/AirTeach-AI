"""
Update src/ui/lesson_board.py to add InteractiveRlcCircuitItem support.
"""

def main():
    with open('src/ui/lesson_board.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Attribute in __init__
    target_init = '        # Interactive Wave Optics & Young\'s Double-Slit Interference Simulator\n        self._young_interference_item: Optional[QGraphicsItem] = None'
    rep_init = target_init + '\n\n        # Interactive AC Series RLC Circuit & Fresnel Phasor Simulator\n        self._rlc_circuit_item: Optional[QGraphicsItem] = None'

    assert target_init in content, "target_init not found"
    content = content.replace(target_init, rep_init, 1)

    # 2. Methods
    target_methods = '''    def get_young_interference_item(self) -> Optional[QGraphicsItem]:
        """Return interactive wave optics interference instance if created."""
        return self._young_interference_item'''

    rlc_methods = '''    def get_young_interference_item(self) -> Optional[QGraphicsItem]:
        """Return interactive wave optics interference instance if created."""
        return self._young_interference_item

    def show_rlc_circuit(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive AC Series RLC Circuit & Phasor simulator."""
        if self._rlc_circuit_item is None:
            from src.ui.interactive_rlc_circuit import InteractiveRlcCircuitItem
            spawn_pos = pos if pos is not None else (450.0, 240.0)
            self._rlc_circuit_item = InteractiveRlcCircuitItem(pos=spawn_pos)
            self.addItem(self._rlc_circuit_item)

        self._rlc_circuit_item.setVisible(visible)
        if visible and pos is not None:
            self._rlc_circuit_item.setPos(pos[0], pos[1])
        self.update()
        return self._rlc_circuit_item if visible else None

    def toggle_rlc_circuit(self) -> bool:
        """Toggle interactive RLC circuit simulator visibility; returns new state."""
        if self._rlc_circuit_item is None or not self._rlc_circuit_item.isVisible():
            self.show_rlc_circuit(True)
            return True
        else:
            self.show_rlc_circuit(False)
            return False

    def is_rlc_circuit_visible(self) -> bool:
        """Return True if interactive RLC circuit simulator is active on board."""
        return self._rlc_circuit_item is not None and self._rlc_circuit_item.isVisible()

    def get_rlc_circuit(self) -> Optional[QGraphicsItem]:
        """Return interactive RLC circuit instance if active and visible."""
        if self._rlc_circuit_item is not None and self._rlc_circuit_item.isVisible():
            return self._rlc_circuit_item
        return None

    def get_rlc_circuit_item(self) -> Optional[QGraphicsItem]:
        """Return interactive RLC circuit instance if created."""
        return self._rlc_circuit_item'''

    assert target_methods in content, "target_methods not found"
    content = content.replace(target_methods, rlc_methods, 1)

    # 3. Excluded set
    target_excl = 'getattr(self, "_young_interference_item", None),'
    rep_excl = target_excl + '\n            getattr(self, "_rlc_circuit_item", None),'

    assert target_excl in content, "target_excl not found"
    content = content.replace(target_excl, rep_excl, 1)

    with open('src/ui/lesson_board.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated src/ui/lesson_board.py!")

if __name__ == '__main__':
    main()
