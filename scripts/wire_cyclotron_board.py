"""
Script to wire InteractiveCyclotronItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add attribute in __init__
    init_target = "        self._action_potential_item: Optional[QGraphicsItem] = None"
    if init_target not in content:
        print("ERROR: init_target not found in lesson_board.py")
        sys.exit(1)

    attr_line = "        self._cyclotron_item: Optional[QGraphicsItem] = None"
    if attr_line not in content:
        content = content.replace(init_target, init_target + "\n" + attr_line, 1)

    # 2. Add methods
    methods_target = "    def get_action_potential_item(self) -> Optional[QGraphicsItem]:\n        \"\"\"Get interactive Hodgkin-Huxley Action Potential simulator item regardless of visibility.\"\"\"\n        return self._action_potential_item"
    if methods_target not in content:
        print("ERROR: methods_target not found in lesson_board.py")
        sys.exit(1)

    new_methods = '''

    def show_cyclotron(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Cyclotron Particle Accelerator simulator."""
        if self._cyclotron_item is None:
            from src.ui.interactive_cyclotron import InteractiveCyclotronItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._cyclotron_item = InteractiveCyclotronItem(pos=spawn_pos)
            self.addItem(self._cyclotron_item)
        else:
            self._cyclotron_item.setVisible(visible)
            if pos is not None:
                self._cyclotron_item.setPos(pos[0], pos[1])
        return self._cyclotron_item if visible else None

    def toggle_cyclotron(self) -> bool:
        """Toggle interactive Cyclotron Particle Accelerator simulator visibility."""
        if self._cyclotron_item is None or not self._cyclotron_item.isVisible():
            self.show_cyclotron(True)
            return True
        else:
            self.show_cyclotron(False)
            return False

    def is_cyclotron_visible(self) -> bool:
        """Check if interactive Cyclotron Particle Accelerator simulator is currently visible."""
        return self._cyclotron_item is not None and self._cyclotron_item.isVisible()

    def get_cyclotron(self) -> Optional[QGraphicsItem]:
        """Get interactive Cyclotron Particle Accelerator simulator item if visible."""
        if self._cyclotron_item is not None and self._cyclotron_item.isVisible():
            return self._cyclotron_item
        return None

    def get_cyclotron_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Cyclotron Particle Accelerator simulator item regardless of visibility."""
        return self._cyclotron_item'''

    if "def show_cyclotron" not in content:
        content = content.replace(methods_target, methods_target + new_methods, 1)

    # 3. Add to exclusion set in get_items_in_rect
    exclude_target = "            getattr(self, \"_action_potential_item\", None),"
    if exclude_target not in content:
        print("ERROR: exclude_target not found in lesson_board.py")
        sys.exit(1)

    exclude_line = "            getattr(self, \"_cyclotron_item\", None),"
    if exclude_line not in content:
        content = content.replace(exclude_target, exclude_target + "\n" + exclude_line, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired InteractiveCyclotronItem into LessonBoard!")

if __name__ == "__main__":
    main()
