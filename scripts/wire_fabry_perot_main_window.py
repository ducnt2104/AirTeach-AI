import sys

with open('src/ui/main_window.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''        fft_act = QAction("📊 Biến đổi Fourier Nhanh FFT & Phổ Tần số (Fast Fourier Transform)", self)
        fft_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_act.triggered.connect(self._on_toggle_fft)
        view_menu.addAction(fft_act)'''

replace1 = '''        fft_act = QAction("📊 Biến đổi Fourier Nhanh FFT & Phổ Tần số (Fast Fourier Transform)", self)
        fft_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_act.triggered.connect(self._on_toggle_fft)
        view_menu.addAction(fft_act)

        fabry_perot_act = QAction("🔬 Giao thoa kế Fabry-Pérot & Hốc Cộng hưởng Laser (Fabry-Perot)", self)
        fabry_perot_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        fabry_perot_act.triggered.connect(self._on_toggle_fabry_perot)
        view_menu.addAction(fabry_perot_act)'''

target2 = '''        fft_tb_act = QAction("📊 Phân tích Phổ FFT", self)
        fft_tb_act.setToolTip("Bật/Tắt mô phỏng Biến đổi Fourier Nhanh Cooley-Tukey FFT & Phân tích Phổ tần số (Ctrl+Alt+F)")
        fft_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_tb_act.triggered.connect(self._on_toggle_fft)
        nav_tb.addAction(fft_tb_act)'''

replace2 = '''        fft_tb_act = QAction("📊 Phân tích Phổ FFT", self)
        fft_tb_act.setToolTip("Bật/Tắt mô phỏng Biến đổi Fourier Nhanh Cooley-Tukey FFT & Phân tích Phổ tần số (Ctrl+Alt+F)")
        fft_tb_act.setShortcut(QKeySequence("Ctrl+Alt+F"))
        fft_tb_act.triggered.connect(self._on_toggle_fft)
        nav_tb.addAction(fft_tb_act)

        fabry_perot_tb_act = QAction("🔬 Fabry-Pérot", self)
        fabry_perot_tb_act.setToolTip("Bật/Tắt mô phỏng Giao thoa kế Fabry-Pérot & Hốc quang học Finesse cao (Ctrl+Alt+O)")
        fabry_perot_tb_act.setShortcut(QKeySequence("Ctrl+Alt+O"))
        fabry_perot_tb_act.triggered.connect(self._on_toggle_fabry_perot)
        nav_tb.addAction(fabry_perot_tb_act)'''

target3 = '''    @Slot()
    def _on_toggle_fft(self) -> bool:
        """Toggle interactive Cooley-Tukey Fast Fourier Transform (FFT) simulator on LessonBoard."""
        is_visible = self._scene.toggle_fft()
        status_msg = "Phân tích Phổ FFT: " + ("Hiển thị (Ctrl+Alt+F)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible'''

replace3 = '''    @Slot()
    def _on_toggle_fft(self) -> bool:
        """Toggle interactive Cooley-Tukey Fast Fourier Transform (FFT) simulator on LessonBoard."""
        is_visible = self._scene.toggle_fft()
        status_msg = "Phân tích Phổ FFT: " + ("Hiển thị (Ctrl+Alt+F)" if is_visible else "Đã ẩn")
        self._status.showMessage(status_msg, 2000)
        return is_visible

    @Slot()
    def _on_toggle_fabry_perot(self) -> bool:
        """Toggle interactive Fabry-Perot Interferometer & Cavity Resonator simulator on LessonBoard."""
        is_visible = self._scene.toggle_fabry_perot()
        status_msg = "Giao thoa kế Fabry-Pérot: " + ("Hiển thị (Ctrl+Alt+O)" if is_visible else "Đã ẩn")
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

print("Successfully updated main_window.py with Fabry-Pérot!")
