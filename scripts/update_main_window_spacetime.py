"""
Update src/ui/main_window.py with Spacetime Relativity menu action, toolbar button, in-air trigger, and slot.
"""

def main():
    with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. View menu action
    target_menu = '''        orbits_act = QAction("🌌 Quỹ đạo vạn vật hấp dẫn & Điểm Lagrange (Gravitational Orbits & Lagrange Points)", self)
        orbits_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        orbits_act.triggered.connect(self._on_toggle_gravitational_orbits)
        view_menu.addAction(orbits_act)'''

    rep_menu = target_menu + '''\n
        relativity_act = QAction("⚡ Không-thời gian Minkowski & Biến đổi Lorentz (Minkowski Spacetime & Lorentz Boost)", self)
        relativity_act.setShortcut(QKeySequence("Ctrl+Alt+M"))
        relativity_act.triggered.connect(self._on_toggle_spacetime_relativity)
        view_menu.addAction(relativity_act)'''

    assert target_menu in content, "target_menu not found"
    content = content.replace(target_menu, rep_menu, 1)

    # 2. Toolbar action
    target_tb = '''        orbits_tb_act = QAction("🌌 Quỹ đạo hấp dẫn", self)
        orbits_tb_act.setToolTip("Bật/Tắt mô phỏng Quỹ đạo vạn vật hấp dẫn N-vật thể & Điểm Lagrange (Ctrl+Alt+G)")
        orbits_tb_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        orbits_tb_act.triggered.connect(self._on_toggle_gravitational_orbits)
        nav_tb.addAction(orbits_tb_act)'''

    rep_tb = target_tb + '''\n
        relativity_tb_act = QAction("⚡ Minkowski", self)
        relativity_tb_act.setToolTip("Bật/Tắt mô phỏng Không-thời gian Minkowski & Biến đổi Lorentz (Ctrl+Alt+M)")
        relativity_tb_act.setShortcut(QKeySequence("Ctrl+Alt+M"))
        relativity_tb_act.triggered.connect(self._on_toggle_spacetime_relativity)
        nav_tb.addAction(relativity_tb_act)'''

    assert target_tb in content, "target_tb not found"
    content = content.replace(target_tb, rep_tb, 1)

    # 3. In-air menu trigger
    target_inair = '''            elif val in ("gravitational_orbits", "orbits", "lagrange", "n_body", "kepler_nbody", "hap_dan"):
                self._on_toggle_gravitational_orbits()'''

    rep_inair = target_inair + '''\n            elif val in ("spacetime_relativity", "minkowski", "relativity", "lorentz", "tuong_doi", "khong_thoi_gian"):
                self._on_toggle_spacetime_relativity()'''

    assert target_inair in content, "target_inair not found"
    content = content.replace(target_inair, rep_inair, 1)

    # 4. Slot
    target_slot = '''    @Slot()
    def _on_toggle_gravitational_orbits(self) -> bool:
        """Toggle interactive gravitational orbits & Lagrange points simulator on LessonBoard."""
        is_visible = self._scene.toggle_gravitational_orbits()
        self._status.showMessage(
            f"Quỹ đạo hấp dẫn & Lagrange: {'Hiển thị (Ctrl+Alt+G)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible'''

    rep_slot = target_slot + '''\n
    @Slot()
    def _on_toggle_spacetime_relativity(self) -> bool:
        """Toggle interactive Special Relativity Minkowski Spacetime simulator on LessonBoard."""
        is_visible = self._scene.toggle_spacetime_relativity()
        self._status.showMessage(
            f"Không-thời gian Minkowski: {'Hiển thị (Ctrl+Alt+M)' if is_visible else 'Đã ẩn'}",
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
