"""
Script to wire Cyclotron menu action, toolbar button, and slot in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View menu action
    view_target = '''        action_potential_act = QAction("⚡ Điện thế Hoạt động & Mô hình Hodgkin-Huxley (Action Potential)", self)
        action_potential_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
        action_potential_act.triggered.connect(self._on_toggle_action_potential)
        view_menu.addAction(action_potential_act)'''

    if view_target not in content:
        print("ERROR: view_target not found in main_window.py")
        sys.exit(1)

    act_code = '''        cyclotron_act = QAction("⚛️ Máy Gia tốc Hạt Cyclotron & Hiệu ứng Khối lượng γ (Cyclotron)", self)
        cyclotron_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        cyclotron_act.triggered.connect(self._on_toggle_cyclotron)
        view_menu.addAction(cyclotron_act)'''

    if "cyclotron_act = QAction" not in content:
        content = content.replace(view_target, view_target + "\n\n" + act_code, 1)

    # 2. Toolbar action
    tb_target = '''        action_potential_tb_act = QAction("⚡ Điện thế Hoạt động", self)
        action_potential_tb_act.setToolTip("Bật/Tắt mô phỏng Điện thế Hoạt động & Mô hình Hodgkin-Huxley (Ctrl+Shift+P)")
        action_potential_tb_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
        action_potential_tb_act.triggered.connect(self._on_toggle_action_potential)
        nav_tb.addAction(action_potential_tb_act)'''

    if tb_target not in content:
        print("ERROR: tb_target not found in main_window.py")
        sys.exit(1)

    tb_code = '''        cyclotron_tb_act = QAction("⚛️ Gia tốc Cyclotron", self)
        cyclotron_tb_act.setToolTip("Bật/Tắt mô phỏng Máy gia tốc hạt Cyclotron & Synchrocyclotron (Ctrl+Alt+C)")
        cyclotron_tb_act.setShortcut(QKeySequence("Ctrl+Alt+C"))
        cyclotron_tb_act.triggered.connect(self._on_toggle_cyclotron)
        nav_tb.addAction(cyclotron_tb_act)'''

    if "cyclotron_tb_act = QAction" not in content:
        content = content.replace(tb_target, tb_target + "\n\n" + tb_code, 1)

    # 3. Slot method
    slot_target = '''    @Slot()
    def _on_toggle_action_potential(self) -> bool:
        """Toggle interactive Hodgkin-Huxley Action Potential simulator on LessonBoard."""
        is_visible = self._scene.toggle_action_potential()
        status_msg = "Điện thế Hoạt động Hodgkin-Huxley: " + ("Hiển thị (Ctrl+Shift+P)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

    if slot_target not in content:
        print("ERROR: slot_target not found in main_window.py")
        sys.exit(1)

    slot_code = '''    @Slot()
    def _on_toggle_cyclotron(self) -> bool:
        """Toggle interactive Cyclotron particle accelerator simulator on LessonBoard."""
        is_visible = self._scene.toggle_cyclotron()
        status_msg = "Máy gia tốc Cyclotron & Synchro: " + ("Hiển thị (Ctrl+Alt+C)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

    if "def _on_toggle_cyclotron" not in content:
        content = content.replace(slot_target, slot_target + "\n\n" + slot_code, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired Cyclotron into MainWindow!")

if __name__ == "__main__":
    main()
