import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''        hplc_act = QAction("🧪 Sắc ký Lỏng HPLC & Hiệu suất Cột Van Deemter (HPLC)", self)
        hplc_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        hplc_act.triggered.connect(self._on_toggle_hplc)
        view_menu.addAction(hplc_act)'''

replace1 = '''        hplc_act = QAction("🧪 Sắc ký Lỏng HPLC & Hiệu suất Cột Van Deemter (HPLC)", self)
        hplc_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        hplc_act.triggered.connect(self._on_toggle_hplc)
        view_menu.addAction(hplc_act)

        photoelectric_act = QAction("⚡ Quang điện Einstein & Lượng tử Ánh sáng (Photoelectric)", self)
        photoelectric_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_act.triggered.connect(self._on_toggle_photoelectric)
        view_menu.addAction(photoelectric_act)'''

target2 = '''        hplc_tb_act = QAction("🧪 Sắc ký HPLC", self)
        hplc_tb_act.setToolTip("Bật/Tắt mô phỏng Sắc ký lỏng HPLC & Hiệu suất cột Van Deemter (Ctrl+Alt+L)")
        hplc_tb_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        hplc_tb_act.triggered.connect(self._on_toggle_hplc)
        nav_tb.addAction(hplc_tb_act)'''

replace2 = '''        hplc_tb_act = QAction("🧪 Sắc ký HPLC", self)
        hplc_tb_act.setToolTip("Bật/Tắt mô phỏng Sắc ký lỏng HPLC & Hiệu suất cột Van Deemter (Ctrl+Alt+L)")
        hplc_tb_act.setShortcut(QKeySequence("Ctrl+Alt+L"))
        hplc_tb_act.triggered.connect(self._on_toggle_hplc)
        nav_tb.addAction(hplc_tb_act)

        photoelectric_tb_act = QAction("⚡ Quang điện", self)
        photoelectric_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu ứng quang điện ngoài Einstein & Công thoát kim loại (Ctrl+Alt+Q)")
        photoelectric_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_tb_act.triggered.connect(self._on_toggle_photoelectric)
        nav_tb.addAction(photoelectric_tb_act)'''

target3 = '''    @Slot()
    def _on_toggle_hplc(self) -> bool:
        """Toggle interactive HPLC & Van Deemter simulator on LessonBoard."""
        is_visible = self._scene.toggle_hplc()
        status_msg = "Sắc ký Lỏng HPLC: " + ("Hiển thị (Ctrl+Alt+L)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

replace3 = '''    @Slot()
    def _on_toggle_hplc(self) -> bool:
        """Toggle interactive HPLC & Van Deemter simulator on LessonBoard."""
        is_visible = self._scene.toggle_hplc()
        status_msg = "Sắc ký Lỏng HPLC: " + ("Hiển thị (Ctrl+Alt+L)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_photoelectric(self) -> bool:
        """Toggle interactive Photoelectric Effect simulator on LessonBoard."""
        is_visible = self._scene.toggle_photoelectric()
        status_msg = "Hiệu ứng Quang điện Einstein: " + ("Hiển thị (Ctrl+Alt+Q)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

text = text.replace('\r\n', '\n')
target1_clean = target1.replace('\r\n', '\n')
target2_clean = target2.replace('\r\n', '\n')
target3_clean = target3.replace('\r\n', '\n')

if target1_clean not in text:
    print("target1 not found")
    sys.exit(1)
text = text.replace(target1_clean, replace1, 1)

if target2_clean not in text:
    print("target2 not found")
    sys.exit(1)
text = text.replace(target2_clean, replace2, 1)

if target3_clean not in text:
    print("target3 not found")
    sys.exit(1)
text = text.replace(target3_clean, replace3, 1)

with open('src/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully updated main_window.py with Photoelectric Effect!")
