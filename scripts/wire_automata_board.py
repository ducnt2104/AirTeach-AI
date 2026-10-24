"""
Script to wire InteractiveAutomataItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute in __init__
    init_target = "        self._bohr_atom_item: Optional[QGraphicsItem] = None"
    init_add = "\n        self._automata_item: Optional[QGraphicsItem] = None"
    if init_target in content and "self._automata_item" not in content:
        content = content.replace(init_target, init_target + init_add, 1)
        print("Added _automata_item attribute to __init__")

    # 2. Lifecycle methods after get_bohr_atom_item
    methods_target = """    def get_bohr_atom_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Bohr atom instance if created.\"\"\"
        return self._bohr_atom_item"""

    methods_add = """

    def show_automata(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        \"\"\"Show or hide the interactive Computer Science Finite State Machine (Automata) simulator.\"\"\"
        if self._automata_item is None:
            from src.ui.interactive_automata import InteractiveAutomataItem
            spawn_pos = pos if pos is not None else (340.0, 160.0)
            self._automata_item = InteractiveAutomataItem(pos=spawn_pos)
            self.addItem(self._automata_item)

        self._automata_item.setVisible(visible)
        if visible and pos is not None:
            self._automata_item.setPos(pos[0], pos[1])
        self.update()
        return self._automata_item if visible else None

    def toggle_automata(self) -> bool:
        \"\"\"Toggle interactive Automata simulator visibility; returns new state.\"\"\"
        if self._automata_item is None or not self._automata_item.isVisible():
            self.show_automata(True)
            return True
        else:
            self.show_automata(False)
            return False

    def is_automata_visible(self) -> bool:
        \"\"\"Return True if interactive Automata simulator is active on board.\"\"\"
        return self._automata_item is not None and self._automata_item.isVisible()

    def get_automata(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Automata instance if active and visible.\"\"\"
        if self._automata_item is not None and self._automata_item.isVisible():
            return self._automata_item
        return None

    def get_automata_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Automata instance if created.\"\"\"
        return self._automata_item"""

    if methods_target in content and "def show_automata" not in content:
        content = content.replace(methods_target, methods_target + methods_add, 1)
        print("Added automata lifecycle methods")

    # 3. Selection exclusion in get_items_in_rect
    excl_target = "getattr(self, \"_bohr_atom_item\", None),"
    excl_add = "\n            getattr(self, \"_automata_item\", None),"
    if excl_target in content and "_automata_item" not in content.split("get_items_in_rect")[1]:
        content = content.replace(excl_target, excl_target + excl_add, 1)
        print("Added _automata_item to get_items_in_rect exclusion tuple")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done wiring automata into lesson_board.py")

if __name__ == "__main__":
    main()
