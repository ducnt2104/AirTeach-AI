"""
Script to wire InteractiveActionPotentialItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add attribute in __init__
    init_target = "        self._huffman_coding_item: Optional[QGraphicsItem] = None"
    if init_target not in content:
        print("ERROR: init_target not found in lesson_board.py")
        sys.exit(1)

    attr_line = "        self._action_potential_item: Optional[QGraphicsItem] = None"
    if attr_line not in content:
        content = content.replace(init_target, init_target + "\n" + attr_line, 1)

    # 2. Add methods
    methods_target = "    def get_huffman_coding_item(self) -> Optional[QGraphicsItem]:\n        \"\"\"Get interactive Huffman Coding & Shannon Entropy simulator item regardless of visibility.\"\"\"\n        return self._huffman_coding_item"
    if methods_target not in content:
        print("ERROR: methods_target not found in lesson_board.py")
        sys.exit(1)

    new_methods = '''

    def show_action_potential(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Hodgkin-Huxley Action Potential simulator."""
        if self._action_potential_item is None:
            from src.ui.interactive_action_potential import InteractiveActionPotentialItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._action_potential_item = InteractiveActionPotentialItem(pos=spawn_pos)
            self.addItem(self._action_potential_item)
        else:
            self._action_potential_item.setVisible(visible)
            if pos is not None:
                self._action_potential_item.setPos(pos[0], pos[1])
        return self._action_potential_item if visible else None

    def toggle_action_potential(self) -> bool:
        """Toggle interactive Hodgkin-Huxley Action Potential simulator visibility."""
        if self._action_potential_item is None or not self._action_potential_item.isVisible():
            self.show_action_potential(True)
            return True
        else:
            self.show_action_potential(False)
            return False

    def is_action_potential_visible(self) -> bool:
        """Check if interactive Hodgkin-Huxley Action Potential simulator is currently visible."""
        return self._action_potential_item is not None and self._action_potential_item.isVisible()

    def get_action_potential(self) -> Optional[QGraphicsItem]:
        """Get interactive Hodgkin-Huxley Action Potential simulator item if visible."""
        if self._action_potential_item is not None and self._action_potential_item.isVisible():
            return self._action_potential_item
        return None

    def get_action_potential_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Hodgkin-Huxley Action Potential simulator item regardless of visibility."""
        return self._action_potential_item'''

    if "def show_action_potential" not in content:
        content = content.replace(methods_target, methods_target + new_methods, 1)

    # 3. Add to exclusion set in get_items_in_rect
    exclude_target = "            getattr(self, \"_huffman_coding_item\", None),"
    if exclude_target not in content:
        print("ERROR: exclude_target not found in lesson_board.py")
        sys.exit(1)

    exclude_line = "            getattr(self, \"_action_potential_item\", None),"
    if exclude_line not in content:
        content = content.replace(exclude_target, exclude_target + "\n" + exclude_line, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired InteractiveActionPotentialItem into LessonBoard!")

if __name__ == "__main__":
    main()
