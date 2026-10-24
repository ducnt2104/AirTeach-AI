path = "src/ui/main_window.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''    @Slot()
    def _on_toggle_sorting(self) -> bool:
        """Toggle interactive Computer Science Sorting Algorithm visualizer on LessonBoard."""
        is_visible = self._scene.toggle_sorting()
        self._status.showMessage(
            f"Thuật toán sắp xếp & Big-O: {'Hiển thị (Ctrl+Alt+S)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible'''

replacement = '''    @Slot()
    def _on_toggle_sorting(self) -> bool:
        """Toggle interactive Computer Science Sorting Algorithm visualizer on LessonBoard."""
        is_visible = self._scene.toggle_sorting()
        self._status.showMessage(
            f"Thuật toán sắp xếp & Big-O: {'Hiển thị (Ctrl+Alt+S)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_punnett_genetics(self) -> bool:
        """Toggle interactive Biology Mendel Genetics & Punnett Square visualizer on LessonBoard."""
        is_visible = self._scene.toggle_punnett_genetics()
        self._status.showMessage(
            f"Di truyền Mendel & Khung Punnett: {'Hiển thị (Ctrl+Alt+G)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible'''

if target in content and "def _on_toggle_punnett_genetics" not in content:
    content = content.replace(target, replacement, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully added _on_toggle_punnett_genetics slot!")
else:
    print("Target not found or already present")
