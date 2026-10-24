"""
Script to wire InteractiveMichelsonInterferometerItem into LessonBoard in src/ui/lesson_board.py.
"""
import sys

def main():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add attribute in __init__
    init_target = "        self._calculus_taylor_item: Optional[QGraphicsItem] = None"
    if init_target not in content:
        print("ERROR: init_target not found in lesson_board.py")
        sys.exit(1)
    
    attr_line = "        self._michelson_interferometer_item: Optional[QGraphicsItem] = None"
    if attr_line not in content:
        content = content.replace(init_target, init_target + "\n" + attr_line, 1)

    # 2. Add methods
    methods_target = "    def get_calculus_taylor_item(self) -> Optional[QGraphicsItem]:\n        \"\"\"Get interactive Riemann Calculus & Taylor Polynomials simulator item regardless of visibility.\"\"\"\n        return self._calculus_taylor_item"
    if methods_target not in content:
        print("ERROR: methods_target not found in lesson_board.py")
        sys.exit(1)

    new_methods = '''

    def show_michelson_interferometer(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Michelson Interferometer & LIGO simulator."""
        if self._michelson_interferometer_item is None:
            from src.ui.interactive_michelson_interferometer import InteractiveMichelsonInterferometerItem
            spawn_pos = pos if pos is not None else (140.0, 50.0)
            self._michelson_interferometer_item = InteractiveMichelsonInterferometerItem(pos=spawn_pos)
            self.addItem(self._michelson_interferometer_item)
        else:
            self._michelson_interferometer_item.setVisible(visible)
            if pos is not None:
                self._michelson_interferometer_item.setPos(pos[0], pos[1])
        return self._michelson_interferometer_item if visible else None

    def toggle_michelson_interferometer(self) -> bool:
        """Toggle interactive Michelson Interferometer & LIGO simulator visibility."""
        if self._michelson_interferometer_item is None or not self._michelson_interferometer_item.isVisible():
            self.show_michelson_interferometer(True)
            return True
        else:
            self.show_michelson_interferometer(False)
            return False

    def is_michelson_interferometer_visible(self) -> bool:
        """Check if interactive Michelson Interferometer & LIGO simulator is currently visible."""
        return self._michelson_interferometer_item is not None and self._michelson_interferometer_item.isVisible()

    def get_michelson_interferometer(self) -> Optional[QGraphicsItem]:
        """Get interactive Michelson Interferometer & LIGO simulator item if visible."""
        if self._michelson_interferometer_item is not None and self._michelson_interferometer_item.isVisible():
            return self._michelson_interferometer_item
        return None

    def get_michelson_interferometer_item(self) -> Optional[QGraphicsItem]:
        """Get interactive Michelson Interferometer & LIGO simulator item regardless of visibility."""
        return self._michelson_interferometer_item'''

    if "def show_michelson_interferometer" not in content:
        content = content.replace(methods_target, methods_target + new_methods, 1)

    # 3. Add to exclusion set in get_items_in_rect
    exclude_target = "            getattr(self, \"_calculus_taylor_item\", None),"
    if exclude_target not in content:
        print("ERROR: exclude_target not found in lesson_board.py")
        sys.exit(1)

    exclude_line = "            getattr(self, \"_michelson_interferometer_item\", None),"
    if exclude_line not in content:
        content = content.replace(exclude_target, exclude_target + "\n" + exclude_line, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired InteractiveMichelsonInterferometerItem into LessonBoard!")

if __name__ == "__main__":
    main()
