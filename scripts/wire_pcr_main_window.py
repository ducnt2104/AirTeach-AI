import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''        photoelectric_act = QAction("⚡ Quang điện Einstein & Lượng tử Ánh sáng (Photoelectric)", self)
        photoelectric_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_act.triggered.connect(self._on_toggle_photoelectric)
        view_menu.addAction(photoelectric_act)'''

replace1 = '''        photoelectric_act = QAction("⚡ Quang điện Einstein & Lượng tử Ánh sáng (Photoelectric)", self)
        photoelectric_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_act.triggered.connect(self._on_toggle_photoelectric)
        view_menu.addAction(photoelectric_act)

        pcr_act = QAction("🧬 Phản ứng Chuỗi Polymerase PCR & Điện di Gel (PCR)", self)
        pcr_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_act.triggered.connect(self._on_toggle_pcr)
        view_menu.addAction(pcr_act)'''

target2 = '''        photoelectric_tb_act = QAction("⚡ Quang điện", self)
        photoelectric_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu ứng quang điện ngoài Einstein & Công thoát kim loại (Ctrl+Alt+Q)")
        photoelectric_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_tb_act.triggered.connect(self._on_toggle_photoelectric)
        nav_tb.addAction(photoelectric_tb_act)'''

replace2 = '''        photoelectric_tb_act = QAction("⚡ Quang điện", self)
        photoelectric_tb_act.setToolTip("Bật/Tắt mô phỏng Hiệu ứng quang điện ngoài Einstein & Công thoát kim loại (Ctrl+Alt+Q)")
        photoelectric_tb_act.setShortcut(QKeySequence("Ctrl+Alt+Q"))
        photoelectric_tb_act.triggered.connect(self._on_toggle_photoelectric)
        nav_tb.addAction(photoelectric_tb_act)

        pcr_tb_act = QAction("🧬 PCR & Điện di", self)
        pcr_tb_act.setToolTip("Bật/Tắt mô phỏng Phản ứng chuỗi Polymerase PCR & Điện di gel agarose (Ctrl+Alt+R)")
        pcr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_tb_act.triggered.connect(self._on_toggle_pcr)
        nav_tb.addAction(pcr_tb_act)'''

target3 = '''    @Slot()
    def _on_toggle_photoelectric(self) -> bool:
        """Toggle interactive Photoelectric Effect simulator on LessonBoard."""
        is_visible = self._scene.toggle_photoelectric()
        status_msg = "Hiệu ứng Quang điện Einstein: " + ("Hiển thị (Ctrl+Alt+Q)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

replace3 = '''    @Slot()
    def _on_toggle_photoelectric(self) -> bool:
        """Toggle interactive Photoelectric Effect simulator on LessonBoard."""
        is_visible = self._scene.toggle_photoelectric()
        status_msg = "Hiệu ứng Quang điện Einstein: " + ("Hiển thị (Ctrl+Alt+Q)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_pcr(self) -> bool:
        """Toggle interactive PCR & Gel Electrophoresis simulator on LessonBoard."""
        is_visible = self._scene.toggle_pcr()
        status_msg = "Phản ứng Chuỗi PCR & Điện di: " + ("Hiển thị (Ctrl+Alt+R)" if is_visible else "Đã ẩn")
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

print("Successfully updated main_window.py with PCR!")
