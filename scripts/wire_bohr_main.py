"""
Script to wire InteractiveBohrAtomItem into MainWindow in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View Menu Action
    view_target = """        carnot_act = QAction("🔥 Động cơ nhiệt Carnot & Chu trình PV - TS (Carnot Heat Engine)", self)
        carnot_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_act.triggered.connect(self._on_toggle_carnot_engine)
        view_menu.addAction(carnot_act)"""

    view_addition = """

        bohr_act = QAction("⚛️ Mô hình Nguyên tử Bohr & Quang phổ Hydro (Bohr Atom & Hydrogen Spectrum)", self)
        bohr_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_act.triggered.connect(self._on_toggle_bohr_atom)
        view_menu.addAction(bohr_act)"""

    if view_target not in content:
        print("ERROR: view_target not found!")
        sys.exit(1)
    if "bohr_act" not in content:
        content = content.replace(view_target, view_target + view_addition, 1)

    # 2. Toolbar Action
    tb_target = """        carnot_tb_act = QAction("🔥 Chu trình Carnot", self)
        carnot_tb_act.setToolTip("Bật/Tắt mô phỏng Động cơ nhiệt lý tưởng Carnot & Đồ thị PV / TS (Ctrl+Alt+T)")
        carnot_tb_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_tb_act.triggered.connect(self._on_toggle_carnot_engine)
        nav_tb.addAction(carnot_tb_act)"""

    tb_addition = """

        bohr_tb_act = QAction("⚛️ Nguyên tử Bohr", self)
        bohr_tb_act.setToolTip("Bật/Tắt mô phỏng Mô hình Nguyên tử Bohr & Dãy vạch quang phổ Hydro (Ctrl+Alt+B)")
        bohr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_tb_act.triggered.connect(self._on_toggle_bohr_atom)
        nav_tb.addAction(bohr_tb_act)"""

    if tb_target not in content:
        print("ERROR: tb_target not found!")
        sys.exit(1)
    if "bohr_tb_act" not in content:
        content = content.replace(tb_target, tb_target + tb_addition, 1)

    # 3. In-air Tool Dispatch
    dispatch_target = """            elif val in ("carnot", "carnot_engine", "dong_co_nhiet", "chu_trinh_carnot", "nhiet_dong_luc"):
                self._on_toggle_carnot_engine()"""

    dispatch_addition = """
            elif val in ("bohr", "bohr_atom", "nguyen_tu_bohr", "quang_pho_hydro", "lyman", "balmer", "paschen"):
                self._on_toggle_bohr_atom()"""

    if dispatch_target not in content:
        print("ERROR: dispatch_target not found!")
        sys.exit(1)
    if "bohr_atom" not in content:
        content = content.replace(dispatch_target, dispatch_target + dispatch_addition, 1)

    # 4. Slot Handler
    slot_target = """    @Slot()
    def _on_toggle_carnot_engine(self) -> bool:
        \"\"\"Toggle interactive Thermodynamics Carnot Heat Engine simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_carnot_engine()
        self._status.showMessage(
            f"Động cơ nhiệt Carnot (PV - TS): {'Hiển thị (Ctrl+Alt+T)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    slot_addition = """

    @Slot()
    def _on_toggle_bohr_atom(self) -> bool:
        \"\"\"Toggle interactive Quantum Physics Bohr Atom simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_bohr_atom()
        self._status.showMessage(
            f"Mô hình Nguyên tử Bohr: {'Hiển thị (Ctrl+Alt+B)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    if slot_target not in content:
        print("ERROR: slot_target not found!")
        sys.exit(1)
    if "_on_toggle_bohr_atom" not in content:
        content = content.replace(slot_target, slot_target + slot_addition, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired Bohr Atom into MainWindow!")

if __name__ == "__main__":
    main()
