"""
Script to cleanly wire photoelectric, carnot, and bohr into main_window.py.
"""
import sys

def main():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View menu bohr_act
    carnot_menu_str = """        carnot_act = QAction("🔥 Động cơ nhiệt Carnot & Chu trình PV - TS (Carnot Heat Engine)", self)
        carnot_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_act.triggered.connect(self._on_toggle_carnot_engine)
        view_menu.addAction(carnot_act)"""

    bohr_menu_str = """
        bohr_act = QAction("⚛️ Mô hình Nguyên tử Bohr & Quang phổ Hydro (Bohr Atom & Hydrogen Spectrum)", self)
        bohr_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_act.triggered.connect(self._on_toggle_bohr_atom)
        view_menu.addAction(bohr_act)"""

    if carnot_menu_str in content and "bohr_act" not in content:
        content = content.replace(carnot_menu_str, carnot_menu_str + bohr_menu_str, 1)
        print("Added bohr_act to view_menu")

    # 2. Toolbar bohr_tb_act
    carnot_tb_str = """        carnot_tb_act = QAction("🔥 Chu trình Carnot", self)
        carnot_tb_act.setToolTip("Bật/Tắt mô phỏng Động cơ nhiệt lý tưởng Carnot & Đồ thị PV / TS (Ctrl+Alt+T)")
        carnot_tb_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        carnot_tb_act.triggered.connect(self._on_toggle_carnot_engine)
        nav_tb.addAction(carnot_tb_act)"""

    bohr_tb_str = """
        bohr_tb_act = QAction("⚛️ Nguyên tử Bohr", self)
        bohr_tb_act.setToolTip("Bật/Tắt mô phỏng Mô hình Nguyên tử Bohr & Dãy vạch quang phổ Hydro (Ctrl+Alt+B)")
        bohr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+B"))
        bohr_tb_act.triggered.connect(self._on_toggle_bohr_atom)
        nav_tb.addAction(bohr_tb_act)"""

    if carnot_tb_str in content and "bohr_tb_act" not in content:
        content = content.replace(carnot_tb_str, carnot_tb_str + bohr_tb_str, 1)
        print("Added bohr_tb_act to nav_tb")

    # 3. In-air Tool Dispatch
    thin_lens_dispatch = """            elif val in ("thin_lens", "lens", "thau_kinh", "optical_bench", "bang_quang_hoc", "kinh_lup"):
                self._on_toggle_thin_lens()"""

    new_dispatches = """            elif val in ("thin_lens", "lens", "thau_kinh", "optical_bench", "bang_quang_hoc", "kinh_lup"):
                self._on_toggle_thin_lens()
            elif val in ("photoelectric", "photoelectric_effect", "quang_dien", "einstein", "luong_tu", "hieu_ung_quang_dien"):
                self._on_toggle_photoelectric_effect()
            elif val in ("carnot", "carnot_engine", "dong_co_nhiet", "chu_trinh_carnot", "nhiet_dong_luc"):
                self._on_toggle_carnot_engine()
            elif val in ("bohr", "bohr_atom", "nguyen_tu_bohr", "quang_pho_hydro", "lyman", "balmer", "paschen"):
                self._on_toggle_bohr_atom()"""

    if thin_lens_dispatch in content and "quang_dien" not in content:
        content = content.replace(thin_lens_dispatch, new_dispatches, 1)
        print("Added in-air tool dispatches")

    # 4. Slots
    pe_slot = """    @Slot()
    def _on_toggle_photoelectric_effect(self) -> bool:
        \"\"\"Toggle interactive Modern Physics Photoelectric Effect simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_photoelectric_effect()
        self._status.showMessage(
            f"Hiệu ứng Quang điện & Thuyết Lượng tử: {'Hiển thị (Ctrl+Alt+P)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    new_slots = """    @Slot()
    def _on_toggle_photoelectric_effect(self) -> bool:
        \"\"\"Toggle interactive Modern Physics Photoelectric Effect simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_photoelectric_effect()
        self._status.showMessage(
            f"Hiệu ứng Quang điện & Thuyết Lượng tử: {'Hiển thị (Ctrl+Alt+P)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_carnot_engine(self) -> bool:
        \"\"\"Toggle interactive Thermodynamics Carnot Heat Engine simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_carnot_engine()
        self._status.showMessage(
            f"Động cơ nhiệt Carnot (PV - TS): {'Hiển thị (Ctrl+Alt+T)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_bohr_atom(self) -> bool:
        \"\"\"Toggle interactive Quantum Physics Bohr Atom simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_bohr_atom()
        self._status.showMessage(
            f"Mô hình Nguyên tử Bohr: {'Hiển thị (Ctrl+Alt+B)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    if pe_slot in content and "_on_toggle_carnot_engine" not in content:
        content = content.replace(pe_slot, new_slots, 1)
        print("Added _on_toggle_carnot_engine and _on_toggle_bohr_atom slots")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done wiring physics tools.")

if __name__ == "__main__":
    main()
