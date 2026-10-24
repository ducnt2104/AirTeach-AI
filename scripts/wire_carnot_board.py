"""
Script to wire InteractiveCarnotEngineItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute
    attr_target = "        self._photoelectric_effect_item: Optional[QGraphicsItem] = None"
    attr_addition = "\n        # Interactive Thermodynamics Carnot Heat Engine Simulator\n        self._carnot_engine_item: Optional[QGraphicsItem] = None"
    if attr_target not in content:
        print("ERROR: attr_target not found!")
        sys.exit(1)
    if "self._carnot_engine_item" not in content:
        content = content.replace(attr_target, attr_target + attr_addition, 1)

    # 2. Methods
    method_target = """    def get_photoelectric_effect_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive photoelectric effect instance if created.\"\"\"
        return self._photoelectric_effect_item"""

    method_addition = """

    def show_carnot_engine(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        \"\"\"Show or hide the interactive Thermodynamics Carnot Heat Engine simulator.\"\"\"
        if self._carnot_engine_item is None:
            from src.ui.interactive_carnot_engine import InteractiveCarnotEngineItem
            spawn_pos = pos if pos is not None else (380.0, 180.0)
            self._carnot_engine_item = InteractiveCarnotEngineItem(pos=spawn_pos)
            self.addItem(self._carnot_engine_item)

        self._carnot_engine_item.setVisible(visible)
        if visible and pos is not None:
            self._carnot_engine_item.setPos(pos[0], pos[1])
        self.update()
        return self._carnot_engine_item if visible else None

    def toggle_carnot_engine(self) -> bool:
        \"\"\"Toggle interactive Carnot heat engine simulator visibility; returns new state.\"\"\"
        if self._carnot_engine_item is None or not self._carnot_engine_item.isVisible():
            self.show_carnot_engine(True)
            return True
        else:
            self.show_carnot_engine(False)
            return False

    def is_carnot_engine_visible(self) -> bool:
        \"\"\"Return True if interactive Carnot engine simulator is active on board.\"\"\"
        return self._carnot_engine_item is not None and self._carnot_engine_item.isVisible()

    def get_carnot_engine(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Carnot engine instance if active and visible.\"\"\"
        if self._carnot_engine_item is not None and self._carnot_engine_item.isVisible():
            return self._carnot_engine_item
        return None

    def get_carnot_engine_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Carnot engine instance if created.\"\"\"
        return self._carnot_engine_item"""

    if method_target not in content:
        print("ERROR: method_target not found!")
        sys.exit(1)
    if "def show_carnot_engine" not in content:
        content = content.replace(method_target, method_target + method_addition, 1)

    # 3. Selection exclusion in get_items_in_rect
    excl_target = "getattr(self, \"_photoelectric_effect_item\", None),"
    excl_addition = "\n            getattr(self, \"_carnot_engine_item\", None),"
    if excl_target not in content:
        print("ERROR: excl_target not found!")
        sys.exit(1)
    if "getattr(self, \"_carnot_engine_item\", None)" not in content:
        content = content.replace(excl_target, excl_target + excl_addition, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired Carnot Engine into LessonBoard!")

if __name__ == "__main__":
    main()
