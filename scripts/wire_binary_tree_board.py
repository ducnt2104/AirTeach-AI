"""
Script to cleanly wire InteractiveBinaryTreeItem into LessonBoard in src/ui/lesson_board.py.
"""
import py_compile
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute declaration in __init__
    init_target = "        self._fluid_venturi_item: Optional[QGraphicsItem] = None"
    init_repl = """        self._fluid_venturi_item: Optional[QGraphicsItem] = None
        self._binary_tree_item: Optional[QGraphicsItem] = None"""

    if init_target in content and "_binary_tree_item" not in content:
        content = content.replace(init_target, init_repl, 1)
        print("Added _binary_tree_item to __init__")

    # 2. Lifecycle methods
    methods_target = """    def get_fluid_venturi_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive fluid venturi instance if created.\"\"\"
        return self._fluid_venturi_item"""

    methods_repl = """    def get_fluid_venturi_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive fluid venturi instance if created.\"\"\"
        return self._fluid_venturi_item

    def show_binary_tree(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        \"\"\"Show or hide the interactive Informatics Binary Tree & Huffman Simulator.\"\"\"
        if self._binary_tree_item is None:
            from src.ui.interactive_binary_tree import InteractiveBinaryTreeItem
            spawn_pos = pos if pos is not None else (310.0, 130.0)
            self._binary_tree_item = InteractiveBinaryTreeItem(pos=spawn_pos)
            self.addItem(self._binary_tree_item)

        self._binary_tree_item.setVisible(visible)
        if visible and pos is not None:
            self._binary_tree_item.setPos(pos[0], pos[1])
        self.update()
        return self._binary_tree_item if visible else None

    def toggle_binary_tree(self) -> bool:
        \"\"\"Toggle interactive binary tree simulator visibility; returns new state.\"\"\"
        if self._binary_tree_item is None or not self._binary_tree_item.isVisible():
            self.show_binary_tree(True)
            return True
        else:
            self.show_binary_tree(False)
            return False

    def is_binary_tree_visible(self) -> bool:
        \"\"\"Return True if interactive binary tree simulator is active on board.\"\"\"
        return self._binary_tree_item is not None and self._binary_tree_item.isVisible()

    def get_binary_tree(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive binary tree instance if active and visible.\"\"\"
        if self._binary_tree_item is not None and self._binary_tree_item.isVisible():
            return self._binary_tree_item
        return None

    def get_binary_tree_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive binary tree instance if created.\"\"\"
        return self._binary_tree_item"""

    if methods_target in content and "def show_binary_tree(" not in content:
        content = content.replace(methods_target, methods_repl, 1)
        print("Added show_binary_tree lifecycle methods")

    # 3. Exclude from get_items_in_rect
    excl_target = "            getattr(self, \"_fluid_venturi_item\", None),"
    excl_repl = """            getattr(self, \"_fluid_venturi_item\", None),
            getattr(self, \"_binary_tree_item\", None),"""

    if excl_target in content and "\"_binary_tree_item\"" not in content:
        content = content.replace(excl_target, excl_repl, 1)
        print("Added _binary_tree_item to get_items_in_rect exclusions")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    py_compile.compile(path, doraise=True)
    print("Successfully updated lesson_board.py and compiled!")

if __name__ == "__main__":
    main()
