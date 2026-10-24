import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

fft_tool_code = '''

    @staticmethod
    def create_fft_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "AUDIO_A440_CONCERT_PITCH",
        f0_hz: float = 125.0,
        sampling_rate_hz: float = 1000.0,
        peak_freq_hz: float = 125.0,
        resolution_hz: float = 15.625,
        nyquist_hz: float = 500.0,
        time_energy: float = 42.0,
        freq_energy: float = 42.0,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Cooley-Tukey Fast Fourier Transform (FFT) & Frequency Spectrum."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#070b14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'TIN HỌC & XỬ LÝ TÍN HIỆU SỐ: THUẬT TOÁN COOLEY-TUKEY FFT O(N log N) – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 38.0
        vw = 360.0
        vh = 120.0

        # 1. Time Domain Waveform Box
        t_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        t_box.setBrush(QBrush(QColor('#020617')))
        t_box.setPen(QPen(QColor('#1e293b'), 1.0))
        t_box.setZValue(10.2)
        items.append(t_box)

        t_lbl = QGraphicsTextItem('MIỀN THỜI GIAN x[n] (Dao động ký):')
        t_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        t_lbl.setDefaultTextColor(QColor('#38bdf8'))
        t_lbl.setPos(vx0 + 8.0, vy0 + 4.0)
        t_lbl.setZValue(10.4)
        items.append(t_lbl)

        # Draw sine wave
        t_mid = vy0 + vh * 0.58
        w_path = QPainterPath()
        w_path.moveTo(vx0 + 30.0, t_mid)
        n_pts = 40
        for i in range(n_pts + 1):
            px = vx0 + 30.0 + (i / float(n_pts)) * (vw - 50.0)
            py = t_mid - 32.0 * math.sin((i / float(n_pts)) * 4.0 * math.pi)
            w_path.lineTo(px, py)
        w_item = QGraphicsPathItem(w_path)
        w_item.setPen(QPen(QColor('#38bdf8'), 1.8))
        w_item.setZValue(10.3)
        items.append(w_item)

        # 2. Frequency Domain Spectrum Box
        fy0 = vy0 + vh + 10.0
        fh = 110.0
        f_box = QGraphicsRectItem(vx0, fy0, vw, fh)
        f_box.setBrush(QBrush(QColor('#020617')))
        f_box.setPen(QPen(QColor('#1e293b'), 1.0))
        f_box.setZValue(10.2)
        items.append(f_box)

        f_lbl = QGraphicsTextItem(f'MIỀN TẦN SỐ |X[k]| (Đỉnh: {peak_freq_hz:.1f} Hz):')
        f_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        f_lbl.setDefaultTextColor(QColor('#f59e0b'))
        f_lbl.setPos(vx0 + 8.0, fy0 + 4.0)
        f_lbl.setZValue(10.4)
        items.append(f_lbl)

        # Draw spectral bars
        f_base = fy0 + fh - 15.0
        bars = [
            (30.0, 10.0, '#f59e0b'),
            (60.0, 15.0, '#f59e0b'),
            (90.0, 75.0, '#ef4444'),   # Peak at f0
            (120.0, 12.0, '#f59e0b'),
            (150.0, 8.0, '#f59e0b'),
            (180.0, 35.0, '#f59e0b'),  # 2nd harmonic
            (210.0, 5.0, '#f59e0b'),
            (240.0, 18.0, '#f59e0b'),  # 3rd harmonic
        ]
        for bx_rel, b_h, col in bars:
            bar_item = QGraphicsRectItem(vx0 + bx_rel + 30.0, f_base - b_h, 12.0, b_h)
            bar_item.setBrush(QBrush(QColor(col)))
            bar_item.setPen(Qt.PenStyle.NoPen)
            bar_item.setZValue(10.4)
            items.append(bar_item)

        # 3. Right Panel: Butterfly Flow & Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh + fh + 10.0

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        # Butterfly miniature graph
        bf_lbl = QGraphicsTextItem('SƠ ĐỒ BƯỚM RADIX-2 DIT:')
        bf_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        bf_lbl.setDefaultTextColor(QColor('#10b981'))
        bf_lbl.setPos(rx0 + 10.0, vy0 + 6.0)
        bf_lbl.setZValue(10.4)
        items.append(bf_lbl)

        # Draw crossed butterfly lines
        for r_idx in range(4):
            y_a = vy0 + 30.0 + r_idx * 16.0
            y_b = y_a + 8.0
            l1 = QGraphicsLineItem(rx0 + 15.0, y_a, rx0 + 90.0, y_b)
            l1.setPen(QPen(QColor('#10b981'), 1.2))
            l1.setZValue(10.3)
            items.append(l1)

            l2 = QGraphicsLineItem(rx0 + 15.0, y_b, rx0 + 90.0, y_a)
            l2.setPen(QPen(QColor('#a855f7'), 1.2))
            l2.setZValue(10.3)
            items.append(l2)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ PHÂN TÍCH PHỔ FFT:\\n'
            f'• Tần số cơ bản f₀: {f0_hz:.1f} Hz\\n'
            f'• Tốc độ lấy mẫu f_s: {sampling_rate_hz:.0f} Hz\\n'
            f'• Tần số Nyquist: {nyquist_hz:.1f} Hz\\n'
            f'• Độ phân giải phổ Δf: {resolution_hz:.2f} Hz/bin\\n'
            f'• Đỉnh phổ cực đại: {peak_freq_hz:.1f} Hz\\n\\n'
            f'ĐỊNH LÝ BẢO TOÀN PARSEVAL:\\n'
            f'• Năng lượng thời gian Σ|x[n]|² = {time_energy:.2f}\\n'
            f'• Năng lượng tần số (1/N)Σ|X[k]|² = {freq_energy:.2f}\\n'
            f'• Sai số bảo toàn: 0.00e+00\\n'
            f'• Độ phức tạp: O(N log₂ N) = 384 phép tính'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 95.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Thuật toán Biến đổi Fourier Nhanh (Cooley-Tukey FFT, 1965): Giảm độ phức tạp từ O(N²) xuống O(N log N) '
            'bằng kỹ thuật chia để trị Decimation-in-Time và nhân thừa số quay W_N^k. '
            'Định lý Shannon-Nyquist khẳng định f_s ≥ 2 f_max để ngăn chặn hiện tượng chồng phổ (aliasing).'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + rh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items
'''

dialog_idx = text.find('class StemToolDialog')
if dialog_idx == -1:
    dialog_idx = text.find('# ─── STEM Subject Dialog')

if dialog_idx == -1:
    print('Failed to find class StemToolDialog!')
    sys.exit(1)

# Find last return items before StemToolDialog
ret_idx = text[:dialog_idx].rfind('        return items\n')
insert_pos = ret_idx + len('        return items\n')

new_text = text[:insert_pos] + fft_tool_code + text[insert_pos:]

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Successfully added create_fft_diagram to InformaticsTools!')
