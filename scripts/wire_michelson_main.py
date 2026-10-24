"""
Script to wire Michelson Interferometer menu action, toolbar button, and slot in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View menu action
    view_target = '''        calculus_act = QAction("📐 Tích phân Riemann & Chuỗi Taylor (Riemann Calculus)", self)
        calculus_act.setShortcut(QKeySequence("Ctrl+Shift+R"))
        calculus_act.triggered.connect(self._on_toggle_calculus_taylor)
        view_menu.addAction(calculus_act)'''

    if view_target not in content:
        print("ERROR: view_target not found in main_window.py")
        sys.exit(1)

    michelson_act_code = '''        michelson_act = QAction("🔬 Giao thoa kế Michelson & Sóng Hấp dẫn LIGO (Michelson Interferometer)", self)
        michelson_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
        michelson_act.triggered.connect(self._on_toggle_michelson_interferometer)
        view_menu.addAction(michelson_act)'''

    if "michelson_act = QAction" not in content:
        content = content.replace(view_target, view_target + "\n\n" + michelson_act_code, 1)

    # 2. Toolbar action
    tb_target = '''        calculus_tb_act = QAction("📐 Tích phân Riemann", self)
        calculus_tb_act.setToolTip("Bật/Tắt mô phỏng Tích phân Riemann, Cầu phương Simpson & Chuỗi Taylor (Ctrl+Shift+R)")
        calculus_tb_act.setShortcut(QKeySequence("Ctrl+Shift+R"))
        calculus_tb_act.triggered.connect(self._on_toggle_calculus_taylor)
        nav_tb.addAction(calculus_tb_act)'''

    if tb_target not in content:
        print("ERROR: tb_target not found in main_window.py")
        sys.exit(1)

    michelson_tb_code = '''        michelson_tb_act = QAction("🔬 Giao thoa Michelson", self)
        michelson_tb_act.setToolTip("Bật/Tắt mô phỏng Giao thoa kế Michelson & Sóng Hấp dẫn LIGO (Ctrl+Shift+M)")
        michelson_tb_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
        michelson_tb_act.triggered.connect(self._on_toggle_michelson_interferometer)
        nav_tb.addAction(michelson_tb_act)'''

    if "michelson_tb_act = QAction" not in content:
        content = content.replace(tb_target, tb_target + "\n\n" + michelson_tb_code, 1)

    # 3. Slot method
    slot_target = '''    @Slot()
    def _on_toggle_calculus_taylor(self) -> bool:
        """Toggle interactive Riemann Calculus & Taylor simulator on LessonBoard."""
        is_visible = self._scene.toggle_calculus_taylor()
        status_msg = "Tích phân Riemann & Chuỗi Taylor: " + ("Hiển thị (Ctrl+Shift+R)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

    if slot_target not in content:
        print("ERROR: slot_target not found in main_window.py")
        sys.exit(1)

    michelson_slot_code = '''    @Slot()
    def _on_toggle_michelson_interferometer(self) -> bool:
        """Toggle interactive Michelson Interferometer & LIGO simulator on LessonBoard."""
        is_visible = self._scene.toggle_michelson_interferometer()
        status_msg = "Giao thoa kế Michelson & LIGO: " + ("Hiển thị (Ctrl+Shift+M)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

    if "def _on_toggle_michelson_interferometer" not in content:
        content = content.replace(slot_target, slot_target + "\n\n" + michelson_slot_code, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired Michelson Interferometer into MainWindow!")

if __name__ == "__main__":
    main()
