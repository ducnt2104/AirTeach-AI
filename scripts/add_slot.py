import sys

with open("src/ui/main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

target = '    @Slot()\n    def _on_cycle_laser_color(self) -> None:'

method = '''    @Slot()
    def _on_toggle_photoelectric_effect(self) -> bool:
        """Toggle interactive Modern Physics Photoelectric Effect simulator on LessonBoard."""
        is_visible = self._scene.toggle_photoelectric_effect()
        self._status.showMessage(
            f"Hiệu ứng Quang điện & Thuyết Lượng tử: {'Hiển thị (Ctrl+Alt+P)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

'''

if target not in content:
    print("ERROR: Target not found!")
    sys.exit(1)

content = content.replace(target, method + target, 1)

with open("src/ui/main_window.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully added _on_toggle_photoelectric_effect!")
