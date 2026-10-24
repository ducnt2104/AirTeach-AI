"""
Script to wire InteractiveGeneticsPunnettItem into MainWindow in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View Menu
    view_target = """        sorting_act = QAction("📊 Thuật toán sắp xếp & Độ phức tạp Big-O (Sorting Algorithms)", self)
        sorting_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        sorting_act.triggered.connect(self._on_toggle_sorting)
        view_menu.addAction(sorting_act)"""

    view_addition = """
        punnett_act = QAction("🧬 Di truyền học Mendel & Khung Punnett (Genetics Punnett Square)", self)
        punnett_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        punnett_act.triggered.connect(self._on_toggle_punnett_genetics)
        view_menu.addAction(punnett_act)"""

    if view_target in content and "punnett_act" not in content:
        content = content.replace(view_target, view_target + view_addition, 1)
        print("Added punnett_act to view_menu")

    # 2. Toolbar
    tb_target = """        sorting_tb_act = QAction("📊 Sắp xếp Big-O", self)
        sorting_tb_act.setToolTip("Bật/Tắt mô phỏng Thuật toán sắp xếp & Bảng độ phức tạp Big-O (Ctrl+Alt+S)")
        sorting_tb_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        sorting_tb_act.triggered.connect(self._on_toggle_sorting)
        nav_tb.addAction(sorting_tb_act)"""

    tb_addition = """
        punnett_tb_act = QAction("🧬 Di truyền Mendel", self)
        punnett_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền Mendel & Khung Punnett Square (Ctrl+Alt+G)")
        punnett_tb_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        punnett_tb_act.triggered.connect(self._on_toggle_punnett_genetics)
        nav_tb.addAction(punnett_tb_act)"""

    if tb_target in content and "punnett_tb_act" not in content:
        content = content.replace(tb_target, tb_target + tb_addition, 1)
        print("Added punnett_tb_act to nav_tb")

    # 3. In-air Tool Dispatch
    dispatch_target = """            elif val in ("sorting", "sort", "sap_xep", "big_o", "bubble_sort", "quick_sort", "merge_sort"):
                self._on_toggle_sorting()"""

    dispatch_addition = """            elif val in ("sorting", "sort", "sap_xep", "big_o", "bubble_sort", "quick_sort", "merge_sort"):
                self._on_toggle_sorting()
            elif val in ("punnett", "genetics", "mendel", "di_truyen", "khung_punnett", "lai_giong"):
                self._on_toggle_punnett_genetics()"""

    if dispatch_target in content and "di_truyen" not in content:
        content = content.replace(dispatch_target, dispatch_addition, 1)
        print("Added punnett to in-air dispatch")

    # 4. Slot (target right before _on_cycle_laser_color)
    slot_target = """    @Slot()
    def _on_cycle_laser_color(self) -> None:"""

    slot_addition = """    @Slot()
    def _on_toggle_punnett_genetics(self) -> bool:
        \"\"\"Toggle interactive Biology Mendel Genetics & Punnett Square visualizer on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_punnett_genetics()
        self._status.showMessage(
            f"Di truyền Mendel & Khung Punnett: {'Hiển thị (Ctrl+Alt+G)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_cycle_laser_color(self) -> None:"""

    if slot_target in content and "_on_toggle_punnett_genetics" not in content:
        content = content.replace(slot_target, slot_addition, 1)
        print("Added _on_toggle_punnett_genetics slot")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done wiring punnett genetics into MainWindow.")

if __name__ == "__main__":
    main()
