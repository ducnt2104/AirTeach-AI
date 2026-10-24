"""
Script to wire InteractiveGeneticsPunnettItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute in __init__
    init_target = "        self._sorting_item: Optional[QGraphicsItem] = None"
    init_add = "\n        self._punnett_genetics_item: Optional[QGraphicsItem] = None"
    if init_target in content and "self._punnett_genetics_item" not in content:
        content = content.replace(init_target, init_target + init_add, 1)
        print("Added _punnett_genetics_item attribute to __init__")

    # 2. Lifecycle methods after get_sorting_item
    methods_target = """    def get_sorting_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Sorting instance if created.\"\"\"
        return self._sorting_item"""

    methods_add = """

    def show_punnett_genetics(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        \"\"\"Show or hide the interactive Biology Mendel Genetics & Punnett Square visualizer.\"\"\"
        if self._punnett_genetics_item is None:
            from src.ui.interactive_genetics_punnett import InteractiveGeneticsPunnettItem
            spawn_pos = pos if pos is not None else (330.0, 140.0)
            self._punnett_genetics_item = InteractiveGeneticsPunnettItem(pos=spawn_pos)
            self.addItem(self._punnett_genetics_item)

        self._punnett_genetics_item.setVisible(visible)
        if visible and pos is not None:
            self._punnett_genetics_item.setPos(pos[0], pos[1])
        self.update()
        return self._punnett_genetics_item if visible else None

    def toggle_punnett_genetics(self) -> bool:
        \"\"\"Toggle interactive Punnett genetics visualizer visibility; returns new state.\"\"\"
        if self._punnett_genetics_item is None or not self._punnett_genetics_item.isVisible():
            self.show_punnett_genetics(True)
            return True
        else:
            self.show_punnett_genetics(False)
            return False

    def is_punnett_genetics_visible(self) -> bool:
        \"\"\"Return True if interactive Punnett genetics visualizer is active on board.\"\"\"
        return self._punnett_genetics_item is not None and self._punnett_genetics_item.isVisible()

    def get_punnett_genetics(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Punnett genetics instance if active and visible.\"\"\"
        if self._punnett_genetics_item is not None and self._punnett_genetics_item.isVisible():
            return self._punnett_genetics_item
        return None

    def get_punnett_genetics_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Punnett genetics instance if created.\"\"\"
        return self._punnett_genetics_item"""

    if methods_target in content and "def show_punnett_genetics" not in content:
        content = content.replace(methods_target, methods_target + methods_add, 1)
        print("Added punnett genetics lifecycle methods")

    # 3. Selection exclusion in get_items_in_rect
    excl_target = "getattr(self, \"_sorting_item\", None),"
    excl_add = "\n            getattr(self, \"_punnett_genetics_item\", None),"
    if excl_target in content and "_punnett_genetics_item" not in content.split("get_items_in_rect")[1]:
        content = content.replace(excl_target, excl_target + excl_add, 1)
        print("Added _punnett_genetics_item to get_items_in_rect exclusion tuple")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done wiring punnett genetics into lesson_board.py")

if __name__ == "__main__":
    main()
