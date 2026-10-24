"""
Script to cleanly wire InteractiveFluidVenturiItem into LessonBoard in src/ui/lesson_board.py.
"""
import py_compile
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute declaration in __init__
    init_target = "        self._organic_chemistry_item: Optional[QGraphicsItem] = None"
    init_repl = """        self._organic_chemistry_item: Optional[QGraphicsItem] = None
        self._fluid_venturi_item: Optional[QGraphicsItem] = None"""

    if init_target in content and "_fluid_venturi_item" not in content:
        content = content.replace(init_target, init_repl, 1)
        print("Added _fluid_venturi_item to __init__")

    # 2. Lifecycle methods
    methods_target = """    def get_organic_chemistry_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive organic chemistry instance if created.\"\"\"
        return self._organic_chemistry_item"""

    methods_repl = """    def get_organic_chemistry_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive organic chemistry instance if created.\"\"\"
        return self._organic_chemistry_item

    def show_fluid_venturi(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        \"\"\"Show or hide the interactive Physics Fluid Dynamics Venturi Tube Simulator.\"\"\"
        if self._fluid_venturi_item is None:
            from src.ui.interactive_fluid_venturi import InteractiveFluidVenturiItem
            spawn_pos = pos if pos is not None else (310.0, 130.0)
            self._fluid_venturi_item = InteractiveFluidVenturiItem(pos=spawn_pos)
            self.addItem(self._fluid_venturi_item)

        self._fluid_venturi_item.setVisible(visible)
        if visible and pos is not None:
            self._fluid_venturi_item.setPos(pos[0], pos[1])
        self.update()
        return self._fluid_venturi_item if visible else None

    def toggle_fluid_venturi(self) -> bool:
        \"\"\"Toggle interactive fluid venturi simulator visibility; returns new state.\"\"\"
        if self._fluid_venturi_item is None or not self._fluid_venturi_item.isVisible():
            self.show_fluid_venturi(True)
            return True
        else:
            self.show_fluid_venturi(False)
            return False

    def is_fluid_venturi_visible(self) -> bool:
        \"\"\"Return True if interactive fluid venturi simulator is active on board.\"\"\"
        return self._fluid_venturi_item is not None and self._fluid_venturi_item.isVisible()

    def get_fluid_venturi(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive fluid venturi instance if active and visible.\"\"\"
        if self._fluid_venturi_item is not None and self._fluid_venturi_item.isVisible():
            return self._fluid_venturi_item
        return None

    def get_fluid_venturi_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive fluid venturi instance if created.\"\"\"
        return self._fluid_venturi_item"""

    if methods_target in content and "def show_fluid_venturi(" not in content:
        content = content.replace(methods_target, methods_repl, 1)
        print("Added show_fluid_venturi lifecycle methods")

    # 3. Exclude from get_items_in_rect
    excl_target = "            getattr(self, \"_organic_chemistry_item\", None),"
    excl_repl = """            getattr(self, \"_organic_chemistry_item\", None),
            getattr(self, \"_fluid_venturi_item\", None),"""

    if excl_target in content and "\"_fluid_venturi_item\"" not in content:
        content = content.replace(excl_target, excl_repl, 1)
        print("Added _fluid_venturi_item to get_items_in_rect exclusions")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    py_compile.compile(path, doraise=True)
    print("Successfully updated lesson_board.py and compiled!")

if __name__ == "__main__":
    main()
