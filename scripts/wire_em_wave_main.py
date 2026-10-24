import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. View menu action
menu_target = '        view_menu.addAction(phase_diagram_act)'
menu_add = '''        view_menu.addAction(phase_diagram_act)

        em_wave_act = QAction("📡 Sóng Điện từ Maxwell & Vectơ Poynting (EM Waves)", self)
        em_wave_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        em_wave_act.triggered.connect(self._on_toggle_em_wave)
        view_menu.addAction(em_wave_act)'''

if menu_target not in text:
    print('Failed to find menu_target in main_window.py')
    sys.exit(1)

text = text.replace(menu_target, menu_add, 1)

# 2. Toolbar action
tb_target = '        nav_tb.addAction(phase_diagram_tb_act)'
tb_add = '''        nav_tb.addAction(phase_diagram_tb_act)

        em_wave_tb_act = QAction("📡 Sóng Điện từ", self)
        em_wave_tb_act.setToolTip("Bật/Tắt mô phỏng Sóng điện từ Maxwell, phân cực & vectơ Poynting (Ctrl+Alt+E)")
        em_wave_tb_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        em_wave_tb_act.triggered.connect(self._on_toggle_em_wave)
        nav_tb.addAction(em_wave_tb_act)'''

if tb_target not in text:
    print('Failed to find tb_target in main_window.py')
    sys.exit(1)

text = text.replace(tb_target, tb_add, 1)

# 3. Slot method
slot_target = '''    @Slot()
    def _on_toggle_phase_diagram(self) -> bool:
        """Toggle interactive Thermodynamics P-T phase diagram simulator on LessonBoard."""
        is_visible = self._scene.toggle_phase_diagram()
        status_msg = "Giản đồ Pha P-T & Clausius-Clapeyron: " + ("Hiển thị (Ctrl+Alt+P)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

slot_add = '''    @Slot()
    def _on_toggle_phase_diagram(self) -> bool:
        """Toggle interactive Thermodynamics P-T phase diagram simulator on LessonBoard."""
        is_visible = self._scene.toggle_phase_diagram()
        status_msg = "Giản đồ Pha P-T & Clausius-Clapeyron: " + ("Hiển thị (Ctrl+Alt+P)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_em_wave(self) -> bool:
        """Toggle interactive Maxwell's electromagnetic wave simulator on LessonBoard."""
        is_visible = self._scene.toggle_em_wave()
        status_msg = "Sóng Điện từ Maxwell & Vectơ Poynting: " + ("Hiển thị (Ctrl+Alt+E)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

if slot_target not in text:
    print('Failed to find slot_target in main_window.py')
    sys.exit(1)

text = text.replace(slot_target, slot_add, 1)

with open('src/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully wired EM Wave into main_window.py!')
