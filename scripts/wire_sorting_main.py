"""
Script to wire InteractiveSortingItem into MainWindow in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View Menu
    view_target = """        automata_act = QAction("🤖 Máy trạng thái hữu hạn & Automata DFA (Finite State Machine)", self)
        automata_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        automata_act.triggered.connect(self._on_toggle_automata)
        view_menu.addAction(automata_act)"""

    view_addition = """
        sorting_act = QAction("📊 Thuật toán sắp xếp & Độ phức tạp Big-O (Sorting Algorithms)", self)
        sorting_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        sorting_act.triggered.connect(self._on_toggle_sorting)
        view_menu.addAction(sorting_act)"""

    if view_target in content and "sorting_act" not in content:
        content = content.replace(view_target, view_target + view_addition, 1)
        print("Added sorting_act to view_menu")

    # 2. Toolbar
    tb_target = """        automata_tb_act = QAction("🤖 Máy trạng thái", self)
        automata_tb_act.setToolTip("Bật/Tắt mô phỏng Máy trạng thái hữu hạn & Automata DFA/NFA (Ctrl+Alt+A)")
        automata_tb_act.setShortcut(QKeySequence("Ctrl+Alt+A"))
        automata_tb_act.triggered.connect(self._on_toggle_automata)
        nav_tb.addAction(automata_tb_act)"""

    tb_addition = """
        sorting_tb_act = QAction("📊 Sắp xếp Big-O", self)
        sorting_tb_act.setToolTip("Bật/Tắt mô phỏng Thuật toán sắp xếp & Bảng độ phức tạp Big-O (Ctrl+Alt+S)")
        sorting_tb_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        sorting_tb_act.triggered.connect(self._on_toggle_sorting)
        nav_tb.addAction(sorting_tb_act)"""

    if tb_target in content and "sorting_tb_act" not in content:
        content = content.replace(tb_target, tb_target + tb_addition, 1)
        print("Added sorting_tb_act to nav_tb")

    # 3. In-air Tool Dispatch
    dispatch_target = """            elif val in ("automata", "fsm", "dfa", "nfa", "finite_state", "may_trang_thai", "automaton"):
                self._on_toggle_automata()"""

    dispatch_addition = """            elif val in ("automata", "fsm", "dfa", "nfa", "finite_state", "may_trang_thai", "automaton"):
                self._on_toggle_automata()
            elif val in ("sorting", "sort", "sap_xep", "big_o", "bubble_sort", "quick_sort", "merge_sort"):
                self._on_toggle_sorting()"""

    if dispatch_target in content and "sap_xep" not in content:
        content = content.replace(dispatch_target, dispatch_addition, 1)
        print("Added sorting to in-air dispatch")

    # 4. Slot (target right before _on_cycle_laser_color)
    slot_target = """    @Slot()
    def _on_cycle_laser_color(self) -> None:"""

    slot_addition = """    @Slot()
    def _on_toggle_sorting(self) -> bool:
        \"\"\"Toggle interactive Computer Science Sorting Algorithm visualizer on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_sorting()
        self._status.showMessage(
            f"Thuật toán sắp xếp & Big-O: {'Hiển thị (Ctrl+Alt+S)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_cycle_laser_color(self) -> None:"""

    if slot_target in content and "_on_toggle_sorting" not in content:
        content = content.replace(slot_target, slot_addition, 1)
        print("Added _on_toggle_sorting slot")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done wiring sorting into MainWindow.")

if __name__ == "__main__":
    main()
