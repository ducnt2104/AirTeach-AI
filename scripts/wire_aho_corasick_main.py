import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. View menu action
menu_target = '        view_menu.addAction(cyclotron_act)'
menu_add = '''        view_menu.addAction(cyclotron_act)

        aho_corasick_act = QAction("🔤 Máy Tự động Tìm kiếm Chuỗi Aho-Corasick (Aho-Corasick Automaton)", self)
        aho_corasick_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        aho_corasick_act.triggered.connect(self._on_toggle_aho_corasick)
        view_menu.addAction(aho_corasick_act)'''

if menu_target not in text:
    print('Failed to find menu_target in main_window.py')
    sys.exit(1)

text = text.replace(menu_target, menu_add, 1)

# 2. Toolbar action
tb_target = '        nav_tb.addAction(cyclotron_tb_act)'
tb_add = '''        nav_tb.addAction(cyclotron_tb_act)

        aho_corasick_tb_act = QAction("🔤 Otomat Aho-Corasick", self)
        aho_corasick_tb_act.setToolTip("Bật/Tắt mô phỏng Máy tự động tìm kiếm chuỗi Aho-Corasick O(n+m+z) (Ctrl+Alt+A)")
        aho_corasick_tb_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        aho_corasick_tb_act.triggered.connect(self._on_toggle_aho_corasick)
        nav_tb.addAction(aho_corasick_tb_act)'''

if tb_target not in text:
    print('Failed to find tb_target in main_window.py')
    sys.exit(1)

text = text.replace(tb_target, tb_add, 1)

# 3. Slot method
slot_target = '''    @Slot()
    def _on_toggle_cyclotron(self) -> bool:
        """Toggle interactive Cyclotron particle accelerator simulator on LessonBoard."""
        is_visible = self._scene.toggle_cyclotron()
        status_msg = "Máy gia tốc Cyclotron & Synchro: " + ("Hiển thị (Ctrl+Alt+C)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

slot_add = '''    @Slot()
    def _on_toggle_cyclotron(self) -> bool:
        """Toggle interactive Cyclotron particle accelerator simulator on LessonBoard."""
        is_visible = self._scene.toggle_cyclotron()
        status_msg = "Máy gia tốc Cyclotron & Synchro: " + ("Hiển thị (Ctrl+Alt+C)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_aho_corasick(self) -> bool:
        """Toggle interactive Aho-Corasick multi-pattern automaton simulator on LessonBoard."""
        is_visible = self._scene.toggle_aho_corasick()
        status_msg = "Mô phỏng Otomat Aho-Corasick: " + ("Hiển thị (Ctrl+Alt+A)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

if slot_target not in text:
    print('Failed to find slot_target in main_window.py')
    sys.exit(1)

text = text.replace(slot_target, slot_add, 1)

with open('src/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired Aho-Corasick into main_window.py!')
