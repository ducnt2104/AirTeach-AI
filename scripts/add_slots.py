path = "src/ui/main_window.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''    @Slot()
    def _on_cycle_laser_color(self) -> None:'''

addition = '''    @Slot()
    def _on_toggle_carnot_engine(self) -> bool:
        """Toggle interactive Thermodynamics Carnot Heat Engine simulator on LessonBoard."""
        is_visible = self._scene.toggle_carnot_engine()
        self._status.showMessage(
            f"Động cơ nhiệt Carnot (PV - TS): {'Hiển thị (Ctrl+Alt+T)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_bohr_atom(self) -> bool:
        """Toggle interactive Quantum Physics Bohr Atom simulator on LessonBoard."""
        is_visible = self._scene.toggle_bohr_atom()
        self._status.showMessage(
            f"Mô hình Nguyên tử Bohr: {'Hiển thị (Ctrl+Alt+B)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_cycle_laser_color(self) -> None:'''

if target in content and "def _on_toggle_carnot_engine" not in content:
    content = content.replace(target, addition, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully added carnot and bohr slots!")
else:
    print("Target not found or already added")
