import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        cap = QGraphicsTextItem(
            'Giản đồ pha (Phase Diagram) biểu diễn các vùng cân bằng nhiệt động Solid-Liquid-Gas. '
            'Đường hóa hơi kết thúc tại Điểm tới hạn (Critical Point), mở ra trạng thái Chất lưu siêu tới hạn. '
            'Độ dốc nóng chảy âm của nước là hiện tượng bất thường duy nhất cho phép trượt băng dưới áp suất lớn.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items'''

tool_code = '''

    @staticmethod
    def create_hplc_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "REVERSE_PHASE_ANALGESICS",
        col_length_mm: float = 150.0,
        dp_um: float = 5.0,
        flow_rate: float = 1.0,
        plate_height_um: float = 15.0,
        plates_n: float = 10000.0,
        backpressure_bar: float = 120.0,
        resolution_rs: float = 2.2,
        void_time_s: float = 90.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of HPLC column elution & UV chromatogram.
        Includes stainless steel column cross-section, separated analyte migration bands,
        Gaussian absorbance peaks with resolution R_s, and Van Deemter efficiency metrics.
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
        title_txt = QGraphicsTextItem(f"SẮC KÝ LỎNG HIỆU NĂNG CAO HPLC & HIỆU SUẤT CỘT: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Cột C18 L = {col_length_mm:.0f} mm | Hạt d_p = {dp_um:.1f} μm | Lưu lượng = {flow_rate:.2f} mL/min | "
            f"Số đĩa N = {plates_n:.0f} | Áp suất ΔP = {backpressure_bar:.0f} bar | R_s = {resolution_rs:.2f}"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Column & Analyte Bands
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("MÔ HÌNH CỘT SẮC KÝ & DẢI CHẤT PHÂN TÁCH:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # Column Tube
        col_w = cw - 60.0
        col_h = 32.0
        col_x = cx + 30.0
        col_y = cy + 40.0

        tube = QGraphicsRectItem(col_x, col_y, col_w, col_h)
        tube.setBrush(QBrush(QColor(30, 41, 59)))
        tube.setPen(QPen(QColor(148, 163, 184), 1.4))
        tube.setZValue(10.2)
        items.append(tube)

        # Analyte bands inside column
        colors = ["#38bdf8", "#eab308", "#f43f5e"]
        for i, pos_frac in enumerate([0.25, 0.55, 0.85]):
            band_x = col_x + pos_frac * (col_w - 20.0)
            band = QGraphicsRectItem(band_x, col_y + 2.0, 16.0, col_h - 4.0)
            band.setBrush(QBrush(QColor(colors[i % len(colors)])))
            band.setPen(Qt.PenStyle.NoPen)
            band.setZValue(10.3)
            items.append(band)

        # Tubing lines
        in_pipe = QGraphicsLineItem(cx + 8.0, col_y + col_h / 2.0, col_x, col_y + col_h / 2.0)
        in_pipe.setPen(QPen(QColor("#38bdf8"), 2.0))
        in_pipe.setZValue(10.3)
        items.append(in_pipe)

        out_pipe = QGraphicsLineItem(col_x + col_w, col_y + col_h / 2.0, cx + cw - 8.0, col_y + col_h / 2.0)
        out_pipe.setPen(QPen(QColor("#38bdf8"), 2.0))
        out_pipe.setZValue(10.3)
        items.append(out_pipe)

        # Column dimension label
        col_lbl = QGraphicsTextItem(f"Cột Pha Đảo C18 ({col_length_mm:.0f} mm × 4.6 mm, {dp_um:.1f} μm)")
        col_lbl.setFont(QFont("Segoe UI", 7))
        col_lbl.setDefaultTextColor(QColor("#94a3b8"))
        col_lbl.setPos(cx + 25.0, col_y + col_h + 10.0)
        col_lbl.setZValue(10.2)
        items.append(col_lbl)

        # 4. Right Panel: UV Absorbance Chromatogram
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("SẮC KÝ ĐỒ UV ABSORBANCE & ĐỘ PHÂN GIẢI R_s:")
        r_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_title.setDefaultTextColor(QColor("#38bdf8"))
        r_title.setPos(rx + 8.0, ry + 6.0)
        r_title.setZValue(10.2)
        items.append(r_title)

        # Axes
        px = rx + 30.0
        py = ry + 25.0
        pw = rw - 45.0
        ph = rh - 85.0

        axis_x = QGraphicsLineItem(px, py + ph, px + pw, py + ph)
        axis_x.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_x.setZValue(10.2)
        items.append(axis_x)

        axis_y = QGraphicsLineItem(px, py, px, py + ph)
        axis_y.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_y.setZValue(10.2)
        items.append(axis_y)

        # Plot 3 Gaussian peaks on chromatogram
        peak_centers = [0.25, 0.55, 0.85]
        peak_heights = [0.85, 0.70, 0.95]
        chrom_path = QPainterPath()
        steps = int(pw)
        for s in range(steps + 1):
            x_frac = s / float(steps)
            tot_abs = 0.0
            for pc, ph_val in zip(peak_centers, peak_heights):
                sigma = 0.035
                tot_abs += ph_val * math.exp(-0.5 * ((x_frac - pc) / sigma) ** 2)
            tot_abs = min(1.0, tot_abs)
            gx = px + s
            gy = py + ph - tot_abs * ph
            if s == 0:
                chrom_path.moveTo(gx, gy)
            else:
                chrom_path.lineTo(gx, gy)

        chrom_item = QGraphicsPathItem(chrom_path)
        chrom_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        chrom_item.setZValue(10.4)
        items.append(chrom_item)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Chiều cao đĩa H = A + B/u + Cu: {plate_height_um:.2f} μm    • Số đĩa N = L/H: {plates_n:.0f}    • Áp suất ΔP: {backpressure_bar:.0f} bar\\n"
            f"• Thời gian chết t_0: {void_time_s:.1f} s    • Độ phân giải đỉnh R_s: {resolution_rs:.2f} ({'Tách nền hoàn toàn' if resolution_rs >= 1.5 else 'Chưa đạt tách nền'})"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, py + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương trình Van Deemter H = A + B/u + C·u định lượng ba nguồn gốc gây giãn rộng pic: khuếch tán xoáy (A), khuếch tán dọc (B) và kháng truyền khối (C). "
            "Độ phân giải sắc ký R_s = 2(t_R2 - t_R1)/(W1 + W2) yêu cầu R_s ≥ 1.5 để đạt tiêu chuẩn tách nền hoàn toàn 99.7% giữa hai đỉnh chất phân tích."
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

print("Successfully added create_hplc_diagram to stem_tools.py!")
