"""
Script to wire InteractiveBohrAtomItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute
    attr_target = "        self._carnot_engine_item: Optional[QGraphicsItem] = None"
    attr_addition = "\n        # Interactive Quantum Physics Bohr Atom Simulator\n        self._bohr_atom_item: Optional[QGraphicsItem] = None"
    if attr_target not in content:
        print("ERROR: attr_target not found!")
        sys.exit(1)
    if "self._bohr_atom_item" not in content:
        content = content.replace(attr_target, attr_target + attr_addition, 1)

    # 2. Methods
    method_target = """    def get_carnot_engine_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Carnot engine instance if created.\"\"\"
        return self._carnot_engine_item"""

    method_addition = """

    def show_bohr_atom(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        \"\"\"Show or hide the interactive Quantum Physics Bohr Atom simulator.\"\"\"
        if self._bohr_atom_item is None:
            from src.ui.interactive_bohr_atom import InteractiveBohrAtomItem
            spawn_pos = pos if pos is not None else (360.0, 180.0)
            self._bohr_atom_item = InteractiveBohrAtomItem(pos=spawn_pos)
            self.addItem(self._bohr_atom_item)

        self._bohr_atom_item.setVisible(visible)
        if visible and pos is not None:
            self._bohr_atom_item.setPos(pos[0], pos[1])
        self.update()
        return self._bohr_atom_item if visible else None

    def toggle_bohr_atom(self) -> bool:
        \"\"\"Toggle interactive Bohr atom simulator visibility; returns new state.\"\"\"
        if self._bohr_atom_item is None or not self._bohr_atom_item.isVisible():
            self.show_bohr_atom(True)
            return True
        else:
            self.show_bohr_atom(False)
            return False

    def is_bohr_atom_visible(self) -> bool:
        \"\"\"Return True if interactive Bohr atom simulator is active on board.\"\"\"
        return self._bohr_atom_item is not None and self._bohr_atom_item.isVisible()

    def get_bohr_atom(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Bohr atom instance if active and visible.\"\"\"
        if self._bohr_atom_item is not None and self._bohr_atom_item.isVisible():
            return self._bohr_atom_item
        return None

    def get_bohr_atom_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Bohr atom instance if created.\"\"\"
        return self._bohr_atom_item"""

    if method_target not in content:
        print("ERROR: method_target not found!")
        sys.exit(1)
    if "def show_bohr_atom" not in content:
        content = content.replace(method_target, method_target + method_addition, 1)

    # 3. Selection exclusion in get_items_in_rect
    excl_target = "getattr(self, \"_carnot_engine_item\", None),"
    excl_addition = "\n            getattr(self, \"_bohr_atom_item\", None),"
    if excl_target not in content:
        print("ERROR: excl_target not found!")
        sys.exit(1)
    if "getattr(self, \"_bohr_atom_item\", None)" not in content:
        content = content.replace(excl_target, excl_target + excl_addition, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired Bohr Atom into LessonBoard!")

if __name__ == "__main__":
    main()
