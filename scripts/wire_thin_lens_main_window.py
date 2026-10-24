import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''        pcr_act = QAction("🧬 Phản ứng Chuỗi Polymerase PCR & Điện di Gel (PCR)", self)
        pcr_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_act.triggered.connect(self._on_toggle_pcr)
        view_menu.addAction(pcr_act)'''

replace1 = '''        pcr_act = QAction("🧬 Phản ứng Chuỗi Polymerase PCR & Điện di Gel (PCR)", self)
        pcr_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_act.triggered.connect(self._on_toggle_pcr)
        view_menu.addAction(pcr_act)

        thin_lens_act = QAction("🔍 Hệ Thấu kính Mỏng & Vẽ Tia sáng (Thin Lens)", self)
        thin_lens_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        thin_lens_act.triggered.connect(self._on_toggle_thin_lens)
        view_menu.addAction(thin_lens_act)'''

target2 = '''        pcr_tb_act = QAction("🧬 PCR & Điện di", self)
        pcr_tb_act.setToolTip("Bật/Tắt mô phỏng Phản ứng chuỗi Polymerase PCR & Điện di gel agarose (Ctrl+Alt+R)")
        pcr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_tb_act.triggered.connect(self._on_toggle_pcr)
        nav_tb.addAction(pcr_tb_act)'''

replace2 = '''        pcr_tb_act = QAction("🧬 PCR & Điện di", self)
        pcr_tb_act.setToolTip("Bật/Tắt mô phỏng Phản ứng chuỗi Polymerase PCR & Điện di gel agarose (Ctrl+Alt+R)")
        pcr_tb_act.setShortcut(QKeySequence("Ctrl+Alt+R"))
        pcr_tb_act.triggered.connect(self._on_toggle_pcr)
        nav_tb.addAction(pcr_tb_act)

        thin_lens_tb_act = QAction("🔍 Thấu kính Mỏng", self)
        thin_lens_tb_act.setToolTip("Bật/Tắt mô phỏng Hệ thấu kính mỏng, tạo ảnh thật/ảo & ma trận ABCD (Ctrl+Alt+T)")
        thin_lens_tb_act.setShortcut(QKeySequence("Ctrl+Alt+T"))
        thin_lens_tb_act.triggered.connect(self._on_toggle_thin_lens)
        nav_tb.addAction(thin_lens_tb_act)'''

target3 = '''    @Slot()
    def _on_toggle_pcr(self) -> bool:
        """Toggle interactive PCR & Gel Electrophoresis simulator on LessonBoard."""
        is_visible = self._scene.toggle_pcr()
        status_msg = "Phản ứng Chuỗi PCR & Điện di: " + ("Hiển thị (Ctrl+Alt+R)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

replace3 = '''    @Slot()
    def _on_toggle_pcr(self) -> bool:
        """Toggle interactive PCR & Gel Electrophoresis simulator on LessonBoard."""
        is_visible = self._scene.toggle_pcr()
        status_msg = "Phản ứng Chuỗi PCR & Điện di: " + ("Hiển thị (Ctrl+Alt+R)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_thin_lens(self) -> bool:
        """Toggle interactive Thin Lens Systems & Geometrical Optics simulator on LessonBoard."""
        is_visible = self._scene.toggle_thin_lens()
        status_msg = "Hệ Thấu kính Mỏng: " + ("Hiển thị (Ctrl+Alt+T)" if is_visible else "Đã ẩn")
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

print("Successfully updated main_window.py with Thin Lens!")
