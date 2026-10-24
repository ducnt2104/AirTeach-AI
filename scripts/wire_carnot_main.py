"""
Script to wire InteractiveCarnotEngineItem into MainWindow in src/ui/main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View Menu Action
    view_target = """        photoelectric_act = QAction("⚡ Hiệu ứng Quang điện & Thuyết Lượng tử (Photoelectric Effect)", self)
        photoelectric_act.setShortcut(QKeySequence("Ctrl+Alt+P"))
        photoelectric_act.triggered.connect(self._on_toggle_photoelectric_effect)
        view_menu.addAction(photoelectric_act)"""

    view_addition = """

        carnot_act = QAction("🔥 Động cơ nhiệt Carnot & Chu trình PV - TS (Carnot Heat Engine)", self)
        carnot_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_act.triggered.connect(self._on_toggle_carnot_engine)
        view_menu.addAction(carnot_act)"""

    if view_target not in content:
        print("ERROR: view_target not found!")
        sys.exit(1)
    if "carnot_act" not in content:
        content = content.replace(view_target, view_target + view_addition, 1)

    # 2. Toolbar Action
    tb_target = """        photoelectric_tb_act = QAction("⚡ Quang điện", self)
        photoelectric_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu ứng Quang điện & Thuyết Lượng tử Ánh sáng Einstein (Ctrl+Alt+P)")
        photoelectric_tb_act.setShortcut(QKeySequence("Ctrl+Alt+P"))
        photoelectric_tb_act.triggered.connect(self._on_toggle_photoelectric_effect)
        nav_tb.addAction(photoelectric_tb_act)"""

    tb_addition = """

        carnot_tb_act = QAction("🔥 Chu trình Carnot", self)
        carnot_tb_act.setToolTip("Bật/Tắt mô phỏng Động cơ nhiệt lý tưởng Carnot & Đồ thị PV / TS (Ctrl+Alt+T)")
        carnot_tb_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_tb_act.triggered.connect(self._on_toggle_carnot_engine)
        nav_tb.addAction(carnot_tb_act)"""

    if tb_target not in content:
        print("ERROR: tb_target not found!")
        sys.exit(1)
    if "carnot_tb_act" not in content:
        content = content.replace(tb_target, tb_target + tb_addition, 1)

    # 3. In-air Tool Dispatch
    dispatch_target = """            elif val in ("thin_lens", "lens", "thau_kinh", "optical_bench", "bang_quang_hoc", "kinh_lup"):
                self._on_toggle_thin_lens()"""

    dispatch_addition = """
            elif val in ("photoelectric", "photoelectric_effect", "quang_dien", "einstein", "luong_tu", "hieu_ung_quang_dien"):
                self._on_toggle_photoelectric_effect()
            elif val in ("carnot", "carnot_engine", "dong_co_nhiet", "chu_trinh_carnot", "nhiet_dong_luc"):
                self._on_toggle_carnot_engine()"""

    if dispatch_target not in content:
        print("ERROR: dispatch_target not found!")
        sys.exit(1)
    if "carnot_engine" not in content:
        content = content.replace(dispatch_target, dispatch_target + dispatch_addition, 1)

    # 4. Slot Handler
    slot_target = """    @Slot()
    def _on_toggle_photoelectric_effect(self) -> bool:
        \"\"\"Toggle interactive Modern Physics Photoelectric Effect simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_photoelectric_effect()
        self._status.showMessage(
            f"Hiệu ứng Quang điện & Thuyết Lượng tử: {'Hiển thị (Ctrl+Alt+P)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    slot_addition = """

    @Slot()
    def _on_toggle_carnot_engine(self) -> bool:
        \"\"\"Toggle interactive Thermodynamics Carnot Heat Engine simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_carnot_engine()
        self._status.showMessage(
            f"Động cơ nhiệt Carnot (PV - TS): {'Hiển thị (Ctrl+Alt+T)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    if slot_target not in content:
        print("ERROR: slot_target not found!")
        sys.exit(1)
    if "_on_toggle_carnot_engine" not in content:
        content = content.replace(slot_target, slot_target + slot_addition, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wired Carnot Engine into MainWindow!")

if __name__ == "__main__":
    main()
