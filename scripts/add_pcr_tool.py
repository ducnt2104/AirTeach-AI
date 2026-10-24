import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        cap = QGraphicsTextItem(
            'Định luật Hardy-Weinberg (1908): Trong quần thể ngẫu phối vô hạn, không chọn lọc tự nhiên, đột biến hay di nhập gen, '
            'tần số tương đối của các alen và kiểu gen duy trì không đổi qua các thế hệ. '
            'Kiểm định Chi-bình phương so sánh tần số thực nghiệm với kỳ vọng để phát hiện các nhân tố tiến hóa tác động.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items'''

tool_code = '''

    @staticmethod
    def create_pcr_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "COVID19_RTPCR_DIAGNOSTICS",
        target_gene: str = "Gen N nucleocapsid",
        amplicon_bp: int = 150,
        cycle_n: int = 25,
        dna_copies: float = 2.5e8,
        ct_threshold: float = 18.5,
        efficiency: float = 0.98,
        annealing_temp_c: float = 58.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of PCR thermal cycling & gel electrophoresis.
        Includes 3-step temperature profiles, DNA strand denaturation/annealing,
        fluorescent agarose gel electrophoresis lanes, and exponential copy metrics.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"PHẢN ỨNG CHUỖI POLYMERASE (PCR) & ĐIỆN DI GEL: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Mục tiêu: {target_gene} ({amplicon_bp} bp) | Chu kỳ n = {cycle_n} | "
            f"Số bản sao N = {dna_copies:.2e} | Ngưỡng C_t = {ct_threshold:.1f} | T_a = {annealing_temp_c:.1f}°C"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: 3-Step Thermal Cycler
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("CHU TRÌNH NHIỆT 3 BƯỚC:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # 3 Temperature Step Boxes
        step_w = (cw - 32.0) / 3.0
        step_h = 60.0
        steps_info = [
            ("Biến tính", "95°C", QColor("#ef4444")),
            ("Bắt cặp", f"{annealing_temp_c:.0f}°C", QColor("#38bdf8")),
            ("Kéo dài", "72°C", QColor("#22c55e")),
        ]
        for idx, (s_name, s_temp, s_col) in enumerate(steps_info):
            box_x = cx + 10.0 + idx * (step_w + 6.0)
            box_y = cy + 28.0
            box = QGraphicsRectItem(box_x, box_y, step_w, step_h)
            box.setBrush(QBrush(QColor(s_col.red(), s_col.green(), s_col.blue(), 50)))
            box.setPen(QPen(s_col, 1.2))
            box.setZValue(10.2)
            items.append(box)

            t1 = QGraphicsTextItem(s_name)
            t1.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            t1.setDefaultTextColor(s_col)
            t1.setPos(box_x + 4.0, box_y + 4.0)
            t1.setZValue(10.3)
            items.append(t1)

            t2 = QGraphicsTextItem(s_temp)
            t2.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            t2.setDefaultTextColor(QColor("#ffffff"))
            t2.setPos(box_x + 4.0, box_y + 24.0)
            t2.setZValue(10.3)
            items.append(t2)

        # DNA Strands diagram
        strand_y = cy + 115.0
        s_lbl = QGraphicsTextItem("Tách sợi đôi → Gắn mồi → Kéo dài:")
        s_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        s_lbl.setDefaultTextColor(QColor("#94a3b8"))
        s_lbl.setPos(cx + 10.0, strand_y)
        s_lbl.setZValue(10.2)
        items.append(s_lbl)

        line1 = QGraphicsLineItem(cx + 15.0, strand_y + 25.0, cx + cw - 15.0, strand_y + 25.0)
        line1.setPen(QPen(QColor("#ef4444"), 2.2))
        line1.setZValue(10.3)
        items.append(line1)

        line2 = QGraphicsLineItem(cx + 15.0, strand_y + 45.0, cx + cw - 15.0, strand_y + 45.0)
        line2.setPen(QPen(QColor("#38bdf8"), 2.2))
        line2.setZValue(10.3)
        items.append(line2)

        # Primers
        p1 = QGraphicsLineItem(cx + 25.0, strand_y + 32.0, cx + 85.0, strand_y + 32.0)
        p1.setPen(QPen(QColor("#22c55e"), 3.5))
        p1.setZValue(10.4)
        items.append(p1)

        p2 = QGraphicsLineItem(cx + cw - 85.0, strand_y + 38.0, cx + cw - 25.0, strand_y + 38.0)
        p2.setPen(QPen(QColor("#22c55e"), 3.5))
        p2.setZValue(10.4)
        items.append(p2)

        # 4. Right Panel: Agarose Gel Electrophoresis
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("BẢN ĐIỆN DI GEL AGAROSE (302 nm UV):")
        r_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_title.setDefaultTextColor(QColor("#38bdf8"))
        r_title.setPos(rx + 8.0, ry + 6.0)
        r_title.setZValue(10.2)
        items.append(r_title)

        # Gel Slab
        gel_w = rw - 50.0
        gel_h = rh - 55.0
        gel_x = rx + 25.0
        gel_y = ry + 25.0

        gel = QGraphicsRectItem(gel_x, gel_y, gel_w, gel_h)
        gel.setBrush(QBrush(QColor(10, 20, 45)))
        gel.setPen(QPen(QColor(30, 58, 138), 1.4))
        gel.setZValue(10.2)
        items.append(gel)

        # Wells
        lane_w = gel_w / 3.0
        for l in range(3):
            wx = gel_x + l * lane_w + lane_w * 0.2
            well = QGraphicsRectItem(wx, gel_y + 4.0, lane_w * 0.6, 6.0)
            well.setBrush(QBrush(QColor(2, 6, 23)))
            well.setPen(QPen(QColor(71, 85, 105), 0.8))
            well.setZValue(10.3)
            items.append(well)

        # DNA Ladder Bands in Lane 1
        l1_x = gel_x + 0.2 * lane_w
        for frac, bp in [(0.25, 1500), (0.40, 1000), (0.55, 750), (0.70, 500), (0.85, 250), (0.95, 100)]:
            by = gel_y + 15.0 + frac * (gel_h - 25.0)
            band = QGraphicsRectItem(l1_x, by, lane_w * 0.6, 3.0)
            band.setBrush(QBrush(QColor("#eab308")))
            band.setPen(Qt.PenStyle.NoPen)
            band.setZValue(10.4)
            items.append(band)

        # Target PCR band in Lane 2
        l2_x = gel_x + 1.2 * lane_w
        target_frac = 0.90 if amplicon_bp <= 150 else (0.70 if amplicon_bp <= 500 else 0.45)
        by_target = gel_y + 15.0 + target_frac * (gel_h - 25.0)
        t_band = QGraphicsRectItem(l2_x, by_target, lane_w * 0.6, 4.5)
        t_band.setBrush(QBrush(QColor("#38bdf8")))
        t_band.setPen(Qt.PenStyle.NoPen)
        t_band.setZValue(10.4)
        items.append(t_band)

        band_lbl = QGraphicsTextItem(f"{amplicon_bp} bp")
        band_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        band_lbl.setDefaultTextColor(QColor("#38bdf8"))
        band_lbl.setPos(l2_x + lane_w * 0.6 + 4.0, by_target - 6.0)
        band_lbl.setZValue(10.4)
        items.append(band_lbl)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Khuếch đại N(n) = N_0(1+E)^n: {dna_copies:.2e} bản sao    • Ngưỡng chu kỳ C_t: {ct_threshold:.2f}    • Hiệu suất E: {efficiency*100:.1f}%\\n"
            f"• Nhiệt độ bắt cặp T_a: {annealing_temp_c:.1f}°C    • Kích thước amplicons: {amplicon_bp} bp ({target_gene})"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, py + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phản ứng chuỗi Polymerase (PCR) tạo ra hàng tỷ bản sao đoạn ADN đích qua 30-40 chu trình nhiệt lặp lại: biến tính 95°C, bắt cặp mồi 55-60°C và kéo dài 72°C. "
            "Kỹ thuật RT-qPCR đo cường độ huỳnh quang theo chu kỳ xác định ngưỡng C_t tỷ lệ nghịch với tải lượng virus khởi đầu."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items'''

text = text.replace('\r\n', '\n')
target_clean = target.replace('\r\n', '\n')

if target_clean not in text:
    print("Could not find target in stem_tools.py")
    sys.exit(1)

text = text.replace(target_clean, target_clean + tool_code, 1)

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully added create_pcr_diagram to stem_tools.py!")
