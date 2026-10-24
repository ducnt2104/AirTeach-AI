import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''        hardy_weinberg_act = QAction("🧬 Cân bằng Di truyền Quần thể Hardy-Weinberg (Hardy-Weinberg)", self)
        hardy_weinberg_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_act.triggered.connect(self._on_toggle_hardy_weinberg)
        view_menu.addAction(hardy_weinberg_act)'''

replace1 = '''        hardy_weinberg_act = QAction("🧬 Cân bằng Di truyền Quần thể Hardy-Weinberg (Hardy-Weinberg)", self)
        hardy_weinberg_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_act.triggered.connect(self._on_toggle_hardy_weinberg)
        view_menu.addAction(hardy_weinberg_act)

        fft_act = QAction("📊 Biến đổi Fourier Nhanh FFT & Phổ Tần số (Fast Fourier Transform)", self)
        fft_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_act.triggered.connect(self._on_toggle_fft)
        view_menu.addAction(fft_act)'''

target2 = '''        hardy_weinberg_tb_act = QAction("🧬 Hardy-Weinberg", self)
        hardy_weinberg_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền học quần thể Hardy-Weinberg & phiêu bạt gen (Ctrl+Alt+H)")
        hardy_weinberg_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_tb_act.triggered.connect(self._on_toggle_hardy_weinberg)
        nav_tb.addAction(hardy_weinberg_tb_act)'''

replace2 = '''        hardy_weinberg_tb_act = QAction("🧬 Hardy-Weinberg", self)
        hardy_weinberg_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền học quần thể Hardy-Weinberg & phiêu bạt gen (Ctrl+Alt+H)")
        hardy_weinberg_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        hardy_weinberg_tb_act.triggered.connect(self._on_toggle_hardy_weinberg)
        nav_tb.addAction(hardy_weinberg_tb_act)

        fft_tb_act = QAction("📊 Phân tích Phổ FFT", self)
        fft_tb_act.setToolTip("Bật/Tắt mô phỏng Biến đổi Fourier Nhanh Cooley-Tukey FFT & Phân tích Phổ tần số (Ctrl+Alt+F)")
        fft_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_tb_act.triggered.connect(self._on_toggle_fft)
        nav_tb.addAction(fft_tb_act)'''

target3 = '''    @Slot()
    def _on_toggle_hardy_weinberg(self) -> bool:
        """Toggle interactive Hardy-Weinberg population genetics simulator on LessonBoard."""
        is_visible = self._scene.toggle_hardy_weinberg()
        status_msg = "Cân bằng Hardy-Weinberg: " + ("Hiển thị (Ctrl+Alt+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

replace3 = '''    @Slot()
    def _on_toggle_hardy_weinberg(self) -> bool:
        """Toggle interactive Hardy-Weinberg population genetics simulator on LessonBoard."""
        is_visible = self._scene.toggle_hardy_weinberg()
        status_msg = "Cân bằng Hardy-Weinberg: " + ("Hiển thị (Ctrl+Alt+H)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_fft(self) -> bool:
        """Toggle interactive Cooley-Tukey Fast Fourier Transform (FFT) simulator on LessonBoard."""
        is_visible = self._scene.toggle_fft()
        status_msg = "Phân tích Phổ FFT: " + ("Hiển thị (Ctrl+Alt+F)" if is_visible else "Đã ẩn")
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

print("Successfully updated main_window.py with FFT!")
