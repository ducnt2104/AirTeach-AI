path = "src/ui/main_window.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''    @Slot()
    def _on_cycle_laser_color(self) -> None:'''

addition = '''    @Slot()
    def _on_toggle_automata(self) -> bool:
        """Toggle interactive Computer Science Finite State Machine (Automata) simulator on LessonBoard."""
        is_visible = self._scene.toggle_automata()
        self._status.showMessage(
            f"Máy trạng thái hữu hạn & Automata: {'Hiển thị (Ctrl+Alt+A)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_cycle_laser_color(self) -> None:'''

if target in content and "def _on_toggle_automata" not in content:
    content = content.replace(target, addition, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully added _on_toggle_automata slot!")
else:
    print("Target not found or already added")
