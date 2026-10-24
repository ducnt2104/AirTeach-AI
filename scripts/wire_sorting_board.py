"""
Script to wire InteractiveSortingItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute in __init__
    init_target = "        self._automata_item: Optional[QGraphicsItem] = None"
    init_add = "\n        self._sorting_item: Optional[QGraphicsItem] = None"
    if init_target in content and "self._sorting_item" not in content:
        content = content.replace(init_target, init_target + init_add, 1)
        print("Added _sorting_item attribute to __init__")

    # 2. Lifecycle methods after get_automata_item
    methods_target = """    def get_automata_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Automata instance if created.\"\"\"
        return self._automata_item"""

    methods_add = """

    def show_sorting(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        \"\"\"Show or hide the interactive Computer Science Sorting Algorithm visualizer.\"\"\"
        if self._sorting_item is None:
            from src.ui.interactive_sorting import InteractiveSortingItem
            spawn_pos = pos if pos is not None else (320.0, 150.0)
            self._sorting_item = InteractiveSortingItem(pos=spawn_pos)
            self.addItem(self._sorting_item)

        self._sorting_item.setVisible(visible)
        if visible and pos is not None:
            self._sorting_item.setPos(pos[0], pos[1])
        self.update()
        return self._sorting_item if visible else None

    def toggle_sorting(self) -> bool:
        \"\"\"Toggle interactive Sorting visualizer visibility; returns new state.\"\"\"
        if self._sorting_item is None or not self._sorting_item.isVisible():
            self.show_sorting(True)
            return True
        else:
            self.show_sorting(False)
            return False

    def is_sorting_visible(self) -> bool:
        \"\"\"Return True if interactive Sorting visualizer is active on board.\"\"\"
        return self._sorting_item is not None and self._sorting_item.isVisible()

    def get_sorting(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Sorting instance if active and visible.\"\"\"
        if self._sorting_item is not None and self._sorting_item.isVisible():
            return self._sorting_item
        return None

    def get_sorting_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Sorting instance if created.\"\"\"
        return self._sorting_item"""

    if methods_target in content and "def show_sorting" not in content:
        content = content.replace(methods_target, methods_target + methods_add, 1)
        print("Added sorting lifecycle methods")

    # 3. Selection exclusion in get_items_in_rect
    excl_target = "getattr(self, \"_automata_item\", None),"
    excl_add = "\n            getattr(self, \"_sorting_item\", None),"
    if excl_target in content and "_sorting_item" not in content.split("get_items_in_rect")[1]:
        content = content.replace(excl_target, excl_target + excl_add, 1)
        print("Added _sorting_item to get_items_in_rect exclusion tuple")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done wiring sorting into lesson_board.py")

if __name__ == "__main__":
    main()
