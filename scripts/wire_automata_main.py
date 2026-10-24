"""
Script to wire InteractiveAutomataItem into MainWindow in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View Menu
    view_target = """        bohr_act = QAction("⚛️ Mô hình Nguyên tử Bohr & Quang phổ Hydro (Bohr Atom & Hydrogen Spectrum)", self)
        bohr_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_act.triggered.connect(self._on_toggle_bohr_atom)
        view_menu.addAction(bohr_act)"""

    view_addition = """
        automata_act = QAction("🤖 Máy trạng thái hữu hạn & Automata DFA (Finite State Machine)", self)
        automata_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        automata_act.triggered.connect(self._on_toggle_automata)
        view_menu.addAction(automata_act)"""

    if view_target in content and "automata_act" not in content:
        content = content.replace(view_target, view_target + view_addition, 1)
        print("Added automata_act to view_menu")

    # 2. Toolbar
    tb_target = """        bohr_tb_act = QAction("⚛️ Nguyên tử Bohr", self)
        bohr_tb_act.setToolTip("Bật/Tắt mô phỏng Mô hình Nguyên tử Bohr & Dãy vạch quang phổ Hydro (Ctrl+Alt+B)")
        bohr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_tb_act.triggered.connect(self._on_toggle_bohr_atom)
        nav_tb.addAction(bohr_tb_act)"""

    tb_addition = """
        automata_tb_act = QAction("🤖 Máy trạng thái", self)
        automata_tb_act.setToolTip("Bật/Tắt mô phỏng Máy trạng thái hữu hạn & Automata DFA/NFA (Ctrl+Alt+A)")
        automata_tb_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        automata_tb_act.triggered.connect(self._on_toggle_automata)
        nav_tb.addAction(automata_tb_act)"""

    if tb_target in content and "automata_tb_act" not in content:
        content = content.replace(tb_target, tb_target + tb_addition, 1)
        print("Added automata_tb_act to nav_tb")

    # 3. In-air Tool Dispatch
    dispatch_target = """            elif val in ("bohr", "bohr_atom", "nguyen_tu_bohr", "quang_pho_hydro", "lyman", "balmer", "paschen"):
                self._on_toggle_bohr_atom()"""

    dispatch_addition = """            elif val in ("bohr", "bohr_atom", "nguyen_tu_bohr", "quang_pho_hydro", "lyman", "balmer", "paschen"):
                self._on_toggle_bohr_atom()
            elif val in ("automata", "fsm", "dfa", "nfa", "finite_state", "may_trang_thai", "automaton"):
                self._on_toggle_automata()"""

    if dispatch_target in content and "may_trang_thai" not in content:
        content = content.replace(dispatch_target, dispatch_addition, 1)
        print("Added automata to in-air dispatch")

    # 4. Slot
    slot_target = """    @Slot()
    def _on_toggle_bohr_atom(self) -> bool:
        \"\"\"Toggle interactive Quantum Physics Bohr Atom simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_bohr_atom()
        self._status.showMessage(
            f"Mô hình Nguyên tử Bohr: {'Hiển thị (Ctrl+Alt+B)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    slot_addition = """    @Slot()
    def _on_toggle_bohr_atom(self) -> bool:
        \"\"\"Toggle interactive Quantum Physics Bohr Atom simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_bohr_atom()
        self._status.showMessage(
            f"Mô hình Nguyên tử Bohr: {'Hiển thị (Ctrl+Alt+B)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_automata(self) -> bool:
        \"\"\"Toggle interactive Computer Science Finite State Machine (Automata) simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_automata()
        self._status.showMessage(
            f"Máy trạng thái hữu hạn & Automata: {'Hiển thị (Ctrl+Alt+A)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    if slot_target in content and "_on_toggle_automata" not in content:
        content = content.replace(slot_target, slot_addition, 1)
        print("Added _on_toggle_automata slot")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done wiring automata into MainWindow.")

if __name__ == "__main__":
    main()
