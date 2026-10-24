path = "src/ui/main_window.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

target = '    @Slot()\n    def _on_cycle_laser_color'
new_slot = '''    @Slot()
    def _on_toggle_organic_chemistry(self) -> bool:
        """Toggle interactive Organic Chemistry Isomer simulator on LessonBoard."""
        is_visible = self._scene.toggle_organic_chemistry()
        self._status.showMessage(
            f"Đồng phân Hóa học Hữu cơ: {'Hiển thị (Ctrl+Alt+H)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_cycle_laser_color'''

if target in text:
    text = text.replace(target, new_slot, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully inserted _on_toggle_organic_chemistry!")
else:
    print("Target not found in main_window.py")
