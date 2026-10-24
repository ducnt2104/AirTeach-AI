"""
Script to cleanly wire InteractiveBinaryTreeItem into MainWindow in src/ui/main_window.py.
"""
import py_compile
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View menu action
    vm_target = """        fluid_act = QAction("🌊 Cơ học chất lưu & Ống Venturi Bernoulli (Fluid Dynamics)", self)
        fluid_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_act.triggered.connect(self._on_toggle_fluid_venturi)
        view_menu.addAction(fluid_act)"""

    vm_repl = """        fluid_act = QAction("🌊 Cơ học chất lưu & Ống Venturi Bernoulli (Fluid Dynamics)", self)
        fluid_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_act.triggered.connect(self._on_toggle_fluid_venturi)
        view_menu.addAction(fluid_act)

        tree_act = QAction("🌳 Cây nhị phân & Mã hóa Huffman (Binary Tree & Huffman)", self)
        tree_act.setShortcut(QKeySequence("Ctrl+Alt+U"))
        tree_act.triggered.connect(self._on_toggle_binary_tree)
        view_menu.addAction(tree_act)"""

    if vm_target in content and "tree_act" not in content:
        content = content.replace(vm_target, vm_repl, 1)
        print("Added tree_act to View menu")

    # 2. Toolbar action
    tb_target = """        fluid_tb_act = QAction("🌊 Cơ học chất lưu", self)
        fluid_tb_act.setToolTip("Bật/Tắt mô phỏng Cơ học chất lưu & Ống Venturi Bernoulli (Ctrl+Alt+F)")
        fluid_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_tb_act.triggered.connect(self._on_toggle_fluid_venturi)
        nav_tb.addAction(fluid_tb_act)"""

    tb_repl = """        fluid_tb_act = QAction("🌊 Cơ học chất lưu", self)
        fluid_tb_act.setToolTip("Bật/Tắt mô phỏng Cơ học chất lưu & Ống Venturi Bernoulli (Ctrl+Alt+F)")
        fluid_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_tb_act.triggered.connect(self._on_toggle_fluid_venturi)
        nav_tb.addAction(fluid_tb_act)

        tree_tb_act = QAction("🌳 Cây nhị phân", self)
        tree_tb_act.setToolTip("Bật/Tắt mô phỏng Cây nhị phân tìm kiếm & Mã hóa Huffman (Ctrl+Alt+U)")
        tree_tb_act.setShortcut(QKeySequence("Ctrl+Alt+U"))
        tree_tb_act.triggered.connect(self._on_toggle_binary_tree)
        nav_tb.addAction(tree_tb_act)"""

    if tb_target in content and "tree_tb_act" not in content:
        content = content.replace(tb_target, tb_repl, 1)
        print("Added tree_tb_act to Navigation toolbar")

    # 3. In-air Tool Dispatch
    air_target = '            elif val in ("fluid", "fluid_venturi", "venturi", "bernoulli", "chat_luu", "ap_suat_tinh", "manometer"):\n                self._on_toggle_fluid_venturi()'

    air_repl = '            elif val in ("fluid", "fluid_venturi", "venturi", "bernoulli", "chat_luu", "ap_suat_tinh", "manometer"):\n                self._on_toggle_fluid_venturi()\n            elif val in ("tree", "binary_tree", "bst", "huffman", "cay_nhi_phan", "duyet_cay"):\n                self._on_toggle_binary_tree()'

    if air_target in content and "cay_nhi_phan" not in content:
        content = content.replace(air_target, air_repl, 1)
        print("Added in-air dispatch for binary tree")

    # 4. Slot
    slot_target = """    @Slot()
    def _on_toggle_fluid_venturi(self) -> bool:
        \"\"\"Toggle interactive Fluid Dynamics Venturi simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_fluid_venturi()
        self._status.showMessage(
            f"Cơ học chất lưu Bernoulli: {'Hiển thị (Ctrl+Alt+F)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    slot_repl = """    @Slot()
    def _on_toggle_fluid_venturi(self) -> bool:
        \"\"\"Toggle interactive Fluid Dynamics Venturi simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_fluid_venturi()
        self._status.showMessage(
            f"Cơ học chất lưu Bernoulli: {'Hiển thị (Ctrl+Alt+F)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_binary_tree(self) -> bool:
        \"\"\"Toggle interactive Binary Tree & Huffman simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_binary_tree()
        self._status.showMessage(
            f"Cây nhị phân & Huffman: {'Hiển thị (Ctrl+Alt+U)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    if slot_target in content and "def _on_toggle_binary_tree(" not in content:
        content = content.replace(slot_target, slot_repl, 1)
        print("Added _on_toggle_binary_tree slot")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    py_compile.compile(path, doraise=True)
    print("Successfully updated main_window.py and compiled!")

if __name__ == "__main__":
    main()
