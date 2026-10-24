import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. View menu action
menu_target = '        view_menu.addAction(em_wave_act)'
menu_add = '''        view_menu.addAction(em_wave_act)

        hardy_weinberg_act = QAction("🧬 Cân bằng Di truyền Quần thể Hardy-Weinberg (Hardy-Weinberg)", self)
        hardy_weinberg_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_act.triggered.connect(self._on_toggle_hardy_weinberg)
        view_menu.addAction(hardy_weinberg_act)'''

if menu_target not in text:
    print('Failed to find menu_target in main_window.py')
    sys.exit(1)

text = text.replace(menu_target, menu_add, 1)

# 2. Toolbar action
tb_target = '        nav_tb.addAction(em_wave_tb_act)'
tb_add = '''        nav_tb.addAction(em_wave_tb_act)

        hardy_weinberg_tb_act = QAction("🧬 Hardy-Weinberg", self)
        hardy_weinberg_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền học quần thể Hardy-Weinberg & phiêu bạt gen (Ctrl+Alt+H)")
        hardy_weinberg_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_tb_act.triggered.connect(self._on_toggle_hardy_weinberg)
        nav_tb.addAction(hardy_weinberg_tb_act)'''

if tb_target not in text:
    print('Failed to find tb_target in main_window.py')
    sys.exit(1)

text = text.replace(tb_target, tb_add, 1)

# 3. Slot method
slot_target = '''    @Slot()
    def _on_toggle_em_wave(self) -> bool:
        """Toggle interactive Maxwell's electromagnetic wave simulator on LessonBoard."""
        is_visible = self._scene.toggle_em_wave()
        status_msg = "Sóng Điện từ Maxwell & Vectơ Poynting: " + ("Hiển thị (Ctrl+Alt+E)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

slot_add = '''    @Slot()
    def _on_toggle_em_wave(self) -> bool:
        """Toggle interactive Maxwell's electromagnetic wave simulator on LessonBoard."""
        is_visible = self._scene.toggle_em_wave()
        status_msg = "Sóng Điện từ Maxwell & Vectơ Poynting: " + ("Hiển thị (Ctrl+Alt+E)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_hardy_weinberg(self) -> bool:
        """Toggle interactive Hardy-Weinberg population genetics simulator on LessonBoard."""
        is_visible = self._scene.toggle_hardy_weinberg()
        status_msg = "Cân bằng Hardy-Weinberg: " + ("Hiển thị (Ctrl+Alt+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

if slot_target not in text:
    print('Failed to find slot_target in main_window.py')
    sys.exit(1)

text = text.replace(slot_target, slot_add, 1)

with open('src/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired Hardy-Weinberg into main_window.py!')
