"""
Update src/ui/main_window.py with Thin Lens menu action, toolbar button, in-air trigger, and slot.
"""

def main():
    with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. View menu action
    target_menu = '''        rlc_act = QAction("⚡ Mạch điện xoay chiều RLC & Giản đồ Fresnel (AC Series RLC Circuit & Phasor)", self)
        rlc_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        rlc_act.triggered.connect(self._on_toggle_rlc_circuit)
        view_menu.addAction(rlc_act)'''

    rep_menu = target_menu + '''\n
        thin_lens_act = QAction("🔬 Thấu kính mỏng & Băng quang học (Thin Lens & Optical Bench)", self)
        thin_lens_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        thin_lens_act.triggered.connect(self._on_toggle_thin_lens)
        view_menu.addAction(thin_lens_act)'''

    assert target_menu in content, "target_menu not found"
    content = content.replace(target_menu, rep_menu, 1)

    # 2. Toolbar action
    target_tb = '''        rlc_tb_act = QAction("⚡ Mạch RLC", self)
        rlc_tb_act.setToolTip("Bật/Tắt mô phỏng Mạch điện xoay chiều RLC nối tiếp & Giản đồ Fresnel (Ctrl+Alt+E)")
        rlc_tb_act.setShortcut(QKeySequence("Ctrl+Alt+E"))
        rlc_tb_act.triggered.connect(self._on_toggle_rlc_circuit)
        nav_tb.addAction(rlc_tb_act)'''

    rep_tb = target_tb + '''\n
        thin_lens_tb_act = QAction("🔬 Thấu kính mỏng", self)
        thin_lens_tb_act.setToolTip("Bật/Tắt mô phỏng Thấu kính mỏng & Băng quang học vẽ tia sáng (Ctrl+Alt+O)")
        thin_lens_tb_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        thin_lens_tb_act.triggered.connect(self._on_toggle_thin_lens)
        nav_tb.addAction(thin_lens_tb_act)'''

    assert target_tb in content, "target_tb not found"
    content = content.replace(target_tb, rep_tb, 1)

    # 3. In-air menu trigger
    target_inair = '''            elif val in ("rlc_circuit", "rlc", "fresnel", "phasor", "xoay_chieu", "dien_xoay_chieu", "cong_huong"):
                self._on_toggle_rlc_circuit()'''

    rep_inair = target_inair + '''\n            elif val in ("thin_lens", "lens", "thau_kinh", "optical_bench", "bang_quang_hoc", "kinh_lup"):
                self._on_toggle_thin_lens()'''

    assert target_inair in content, "target_inair not found"
    content = content.replace(target_inair, rep_inair, 1)

    # 4. Slot
    target_slot = '''    @Slot()
    def _on_toggle_rlc_circuit(self) -> bool:
        """Toggle interactive AC Series RLC Circuit & Phasor simulator on LessonBoard."""
        is_visible = self._scene.toggle_rlc_circuit()
        self._status.showMessage(
            f"Mạch RLC xoay chiều & Fresnel: {'Hiển thị (Ctrl+Alt+E)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible'''

    rep_slot = target_slot + '''\n
    @Slot()
    def _on_toggle_thin_lens(self) -> bool:
        """Toggle interactive Geometric Optics Thin Lens & Optical Bench simulator on LessonBoard."""
        is_visible = self._scene.toggle_thin_lens()
        self._status.showMessage(
            f"Thấu kính mỏng & Băng quang học: {'Hiển thị (Ctrl+Alt+O)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible'''

    assert target_slot in content, "target_slot not found"
    content = content.replace(target_slot, rep_slot, 1)

    with open('src/ui/main_window.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated src/ui/main_window.py!")

if __name__ == '__main__':
    main()
