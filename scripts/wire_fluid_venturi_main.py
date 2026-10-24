"""
Script to cleanly wire InteractiveFluidVenturiItem into MainWindow in src/ui/main_window.py.
"""
import py_compile
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View menu action
    vm_target = """        organic_act = QAction("🧪 Đồng phân Hóa học Hữu cơ & Nhóm chức (Organic Isomerism)", self)
        organic_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_act.triggered.connect(self._on_toggle_organic_chemistry)
        view_menu.addAction(organic_act)"""

    vm_repl = """        organic_act = QAction("🧪 Đồng phân Hóa học Hữu cơ & Nhóm chức (Organic Isomerism)", self)
        organic_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_act.triggered.connect(self._on_toggle_organic_chemistry)
        view_menu.addAction(organic_act)

        fluid_act = QAction("🌊 Cơ học chất lưu & Ống Venturi Bernoulli (Fluid Dynamics)", self)
        fluid_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_act.triggered.connect(self._on_toggle_fluid_venturi)
        view_menu.addAction(fluid_act)"""

    if vm_target in content and "fluid_act" not in content:
        content = content.replace(vm_target, vm_repl, 1)
        print("Added fluid_act to View menu")

    # 2. Toolbar action
    tb_target = """        organic_tb_act = QAction("🧪 Đồng phân Hóa học", self)
        organic_tb_act.setToolTip("Bật/Tắt mô phỏng Đồng phân Hóa học Hữu cơ & Nhóm chức (Ctrl+Alt+H)")
        organic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_tb_act.triggered.connect(self._on_toggle_organic_chemistry)
        nav_tb.addAction(organic_tb_act)"""

    tb_repl = """        organic_tb_act = QAction("🧪 Đồng phân Hóa học", self)
        organic_tb_act.setToolTip("Bật/Tắt mô phỏng Đồng phân Hóa học Hữu cơ & Nhóm chức (Ctrl+Alt+H)")
        organic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_tb_act.triggered.connect(self._on_toggle_organic_chemistry)
        nav_tb.addAction(organic_tb_act)

        fluid_tb_act = QAction("🌊 Cơ học chất lưu", self)
        fluid_tb_act.setToolTip("Bật/Tắt mô phỏng Cơ học chất lưu & Ống Venturi Bernoulli (Ctrl+Alt+F)")
        fluid_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fluid_tb_act.triggered.connect(self._on_toggle_fluid_venturi)
        nav_tb.addAction(fluid_tb_act)"""

    if tb_target in content and "fluid_tb_act" not in content:
        content = content.replace(tb_target, tb_repl, 1)
        print("Added fluid_tb_act to Navigation toolbar")

    # 3. In-air Tool Dispatch
    air_target = """            elif val in ("organic", "organic_chem", "isomer", "dong_phan", "hoa_huu_co", "hydrocarbon"):
                self._on_toggle_organic_chemistry()"""

    air_repl = """            elif val in ("organic", "organic_chem", "isomer", "dong_phan", "hoa_huu_co", "hydrocarbon"):
                self._on_toggle_organic_chemistry()
            elif val in ("fluid", "fluid_venturi", "venturi", "bernoulli", "chat_luu", "ap_suat_tinh", "manometer"):
                self._on_toggle_fluid_venturi()"""

    if air_target in content and "fluid_venturi" not in content:
        content = content.replace(air_target, air_repl, 1)
        print("Added in-air dispatch for fluid venturi")

    # 4. Slot
    slot_target = """    @Slot()
    def _on_toggle_organic_chemistry(self) -> bool:
        \"\"\"Toggle interactive Organic Chemistry Isomer simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_organic_chemistry()
        self._status.showMessage(
            f"Đồng phân Hóa học Hữu cơ: {'Hiển thị (Ctrl+Alt+H)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    slot_repl = """    @Slot()
    def _on_toggle_organic_chemistry(self) -> bool:
        \"\"\"Toggle interactive Organic Chemistry Isomer simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_organic_chemistry()
        self._status.showMessage(
            f"Đồng phân Hóa học Hữu cơ: {'Hiển thị (Ctrl+Alt+H)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_fluid_venturi(self) -> bool:
        \"\"\"Toggle interactive Fluid Dynamics Venturi simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_fluid_venturi()
        self._status.showMessage(
            f"Cơ học chất lưu Bernoulli: {'Hiển thị (Ctrl+Alt+F)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    if slot_target in content and "def _on_toggle_fluid_venturi(" not in content:
        content = content.replace(slot_target, slot_repl, 1)
        print("Added _on_toggle_fluid_venturi slot")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    py_compile.compile(path, doraise=True)
    print("Successfully updated main_window.py and compiled!")

if __name__ == "__main__":
    main()
