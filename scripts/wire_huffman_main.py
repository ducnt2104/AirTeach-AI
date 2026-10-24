"""
Script to wire Huffman Coding menu action, toolbar button, and slot in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View menu action
    view_target = '''        michelson_act = QAction("🔬 Giao thoa kế Michelson & Sóng Hấp dẫn LIGO (Michelson Interferometer)", self)
        michelson_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
        michelson_act.triggered.connect(self._on_toggle_michelson_interferometer)
        view_menu.addAction(michelson_act)'''

    if view_target not in content:
        print("ERROR: view_target not found in main_window.py")
        sys.exit(1)

    huffman_act_code = '''        huffman_act = QAction("🌲 Cây Mã hóa Huffman & Lý thuyết Thông tin Shannon (Huffman Coding)", self)
        huffman_act.setShortcut(QKeySequence("Ctrl+Shift+H"))
        huffman_act.triggered.connect(self._on_toggle_huffman_coding)
        view_menu.addAction(huffman_act)'''

    if "huffman_act = QAction" not in content:
        content = content.replace(view_target, view_target + "\n\n" + huffman_act_code, 1)

    # 2. Toolbar action
    tb_target = '''        michelson_tb_act = QAction("🔬 Giao thoa Michelson", self)
        michelson_tb_act.setToolTip("Bật/Tắt mô phỏng Giao thoa kế Michelson & Sóng Hấp dẫn LIGO (Ctrl+Shift+M)")
        michelson_tb_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
        michelson_tb_act.triggered.connect(self._on_toggle_michelson_interferometer)
        nav_tb.addAction(michelson_tb_act)'''

    if tb_target not in content:
        print("ERROR: tb_target not found in main_window.py")
        sys.exit(1)

    huffman_tb_code = '''        huffman_tb_act = QAction("🌲 Mã hóa Huffman", self)
        huffman_tb_act.setToolTip("Bật/Tắt mô phỏng Cây tiền tố tối ưu Huffman & Entropy Shannon (Ctrl+Shift+H)")
        huffman_tb_act.setShortcut(QKeySequence("Ctrl+Shift+H"))
        huffman_tb_act.triggered.connect(self._on_toggle_huffman_coding)
        nav_tb.addAction(huffman_tb_act)'''

    if "huffman_tb_act = QAction" not in content:
        content = content.replace(tb_target, tb_target + "\n\n" + huffman_tb_code, 1)

    # 3. Slot method
    slot_target = '''    @Slot()
    def _on_toggle_michelson_interferometer(self) -> bool:
        """Toggle interactive Michelson Interferometer & LIGO simulator on LessonBoard."""
        is_visible = self._scene.toggle_michelson_interferometer()
        status_msg = "Giao thoa kế Michelson & LIGO: " + ("Hiển thị (Ctrl+Shift+M)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

    if slot_target not in content:
        print("ERROR: slot_target not found in main_window.py")
        sys.exit(1)

    huffman_slot_code = '''    @Slot()
    def _on_toggle_huffman_coding(self) -> bool:
        """Toggle interactive Huffman Coding & Shannon Entropy simulator on LessonBoard."""
        is_visible = self._scene.toggle_huffman_coding()
        status_msg = "Mã hóa Huffman & Entropy Shannon: " + ("Hiển thị (Ctrl+Shift+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

    if "def _on_toggle_huffman_coding" not in content:
        content = content.replace(slot_target, slot_target + "\n\n" + huffman_slot_code, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired Huffman Coding into MainWindow!")

if __name__ == "__main__":
    main()
