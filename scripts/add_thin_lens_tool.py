import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương trình lượng tử Einstein E_k = hν - Φ = e·U_h chứng minh bản chất hạt của ánh sáng: mỗi photon truyền toàn bộ năng lượng hν cho một electron. "
            "Hiện tượng quang điện chỉ xảy ra khi năng lượng photon vượt qua công thoát kim loại (hν ≥ Φ), độc lập hoàn toàn với cường độ sáng."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items'''

tool_code = '''

    @staticmethod
    def create_thin_lens_system_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CONVEX_LENS_REAL_IMAGE",
        f1_mm: float = 100.0,
        do_mm: float = 250.0,
        di_mm: float = 166.7,
        magnification: float = -0.67,
        is_real: bool = True,
        is_two_lens: bool = False,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Thin Lens Geometrical Optics Ray Tracing.
        Includes principal optical axis, convex/concave lens symbols, focal points,
        object arrow, principal rays (parallel and central), real/virtual image arrow, and ABCD matrix metrics.
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
        title_txt = QGraphicsTextItem(f"QUANG HÌNH HỌC & HỆ THẤU KÍNH MỎNG: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        lens_type_str = "Hội tụ (Convex)" if f1_mm > 0 else "Phân kỳ (Concave)"
        img_type_str = "ẢNH THẬT (Real, Ngược chiều)" if is_real else "ẢNH ẢO (Virtual, Cùng chiều)"
        sub_txt = QGraphicsTextItem(
            f"Thấu kính: {lens_type_str} f = {f1_mm:+.1f} mm | d_o = {do_mm:.1f} mm | d_i = {di_mm:+.1f} mm | "
            f"Độ phóng đại m = {magnification:+.2f} | {img_type_str}"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Ray Tracing Optical Bench Panel
        bw = width - 32.0
        bh = height - 165.0
        bx = x0 + 16.0
        by = y0 + 58.0

        b_bg = QGraphicsRectItem(bx, by, bw, bh)
        b_bg.setBrush(QBrush(QColor(4, 7, 16)))
        b_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        b_bg.setZValue(10.1)
        items.append(b_bg)

        # Principal Axis
        axis_y = by + bh / 2.0
        axis_line = QGraphicsLineItem(bx + 10.0, axis_y, bx + bw - 10.0, axis_y)
        axis_line.setPen(QPen(QColor(71, 85, 105), 1.2, Qt.PenStyle.DashLine))
        axis_line.setZValue(10.2)
        items.append(axis_line)

        # Lens 1 Line & Arrowheads
        scale_x = (bw - 60.0) / 500.0
        lens_x = bx + bw * 0.52
        lens_top = axis_y - 65.0
        lens_bot = axis_y + 65.0

        lens_line = QGraphicsLineItem(lens_x, lens_top, lens_x, lens_bot)
        lens_line.setPen(QPen(QColor("#38bdf8"), 2.4))
        lens_line.setZValue(10.3)
        items.append(lens_line)

        # Arrowheads on Lens
        is_convex = f1_mm > 0
        arr_sz = 6.0
        if is_convex:
            # Arrows pointing out
            a1 = QGraphicsLineItem(lens_x - arr_sz, lens_top + arr_sz, lens_x, lens_top)
            a1.setPen(QPen(QColor("#38bdf8"), 2.0))
            a1.setZValue(10.3)
            items.append(a1)
            a2 = QGraphicsLineItem(lens_x + arr_sz, lens_top + arr_sz, lens_x, lens_top)
            a2.setPen(QPen(QColor("#38bdf8"), 2.0))
            a2.setZValue(10.3)
            items.append(a2)
        else:
            # Arrows pointing inward
            a1 = QGraphicsLineItem(lens_x - arr_sz, lens_top, lens_x, lens_top + arr_sz)
            a1.setPen(QPen(QColor("#38bdf8"), 2.0))
            a1.setZValue(10.3)
            items.append(a1)
            a2 = QGraphicsLineItem(lens_x + arr_sz, lens_top, lens_x, lens_top + arr_sz)
            a2.setPen(QPen(QColor("#38bdf8"), 2.0))
            a2.setZValue(10.3)
            items.append(a2)

        # Focal points F1 and F'1
        f_px = abs(f1_mm) * scale_x * 0.7
        f_left_x = lens_x - f_px
        f_right_x = lens_x + f_px

        dot_f1 = QGraphicsEllipseItem(f_left_x - 3.0, axis_y - 3.0, 6.0, 6.0)
        dot_f1.setBrush(QBrush(QColor("#f43f5e")))
        dot_f1.setPen(Qt.PenStyle.NoPen)
        dot_f1.setZValue(10.4)
        items.append(dot_f1)

        dot_f2 = QGraphicsEllipseItem(f_right_x - 3.0, axis_y - 3.0, 6.0, 6.0)
        dot_f2.setBrush(QBrush(QColor("#f43f5e")))
        dot_f2.setPen(Qt.PenStyle.NoPen)
        dot_f2.setZValue(10.4)
        items.append(dot_f2)

        f_lbl = QGraphicsTextItem(f"F (f={abs(f1_mm):.0f}mm)")
        f_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        f_lbl.setDefaultTextColor(QColor("#f43f5e"))
        f_lbl.setPos(f_left_x - 14.0, axis_y + 4.0)
        f_lbl.setZValue(10.4)
        items.append(f_lbl)

        # Object Arrow AB (Yellow)
        obj_dist_px = do_mm * scale_x * 0.7
        obj_x = lens_x - obj_dist_px
        obj_h_px = 35.0
        obj_tip_y = axis_y - obj_h_px

        obj_line = QGraphicsLineItem(obj_x, axis_y, obj_x, obj_tip_y)
        obj_line.setPen(QPen(QColor("#facc15"), 2.5))
        obj_line.setZValue(10.4)
        items.append(obj_line)

        obj_lbl = QGraphicsTextItem("Vật AB")
        obj_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        obj_lbl.setDefaultTextColor(QColor("#facc15"))
        obj_lbl.setPos(obj_x - 10.0, obj_tip_y - 14.0)
        obj_lbl.setZValue(10.4)
        items.append(obj_lbl)

        # Image Arrow A'B'
        img_dist_px = di_mm * scale_x * 0.7
        img_x = lens_x + img_dist_px
        img_h_px = obj_h_px * abs(magnification)
        img_tip_y = axis_y + (img_h_px if is_real else -img_h_px)

        img_col = QColor("#22c55e" if is_real else "#f43f5e")
        img_line = QGraphicsLineItem(img_x, axis_y, img_x, img_tip_y)
        img_line.setPen(QPen(img_col, 2.5, Qt.PenStyle.SolidLine if is_real else Qt.PenStyle.DashLine))
        img_line.setZValue(10.4)
        items.append(img_line)

        img_lbl = QGraphicsTextItem("Ảnh A'B'")
        img_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        img_lbl.setDefaultTextColor(img_col)
        img_lbl.setPos(img_x - 12.0, img_tip_y + (4.0 if is_real else -14.0))
        img_lbl.setZValue(10.4)
        items.append(img_lbl)

        # Ray 1: Parallel ray refracts through F'1
        r1_a = QGraphicsLineItem(obj_x, obj_tip_y, lens_x, obj_tip_y)
        r1_a.setPen(QPen(QColor("#ef4444"), 1.6))
        r1_a.setZValue(10.3)
        items.append(r1_a)

        r1_b = QGraphicsLineItem(lens_x, obj_tip_y, img_x, img_tip_y)
        r1_b.setPen(QPen(QColor("#ef4444"), 1.6))
        r1_b.setZValue(10.3)
        items.append(r1_b)

        # Ray 2: Central ray through optical center O
        r2 = QGraphicsLineItem(obj_x, obj_tip_y, img_x, img_tip_y)
        r2.setPen(QPen(QColor("#22c55e"), 1.6))
        r2.setZValue(10.3)
        items.append(r2)

        # 4. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Công thức Gauss: 1/f = 1/d_o + 1/d_i    • Độ phóng đại: m = -d_i / d_o = {magnification:+.2f}    • Độ tụ: D = 1/f = {1000.0/f1_mm:+.1f} dp\\n"
            f"• Khoảng cách vật d_o: {do_mm:.1f} mm    • Khoảng cách ảnh d_i: {di_mm:+.1f} mm    • Ma trận ABCD: det(M) = AD - BC = 1.000"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(bx + 8.0, by + bh + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 5. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Công thức thấu kính mỏng Gauss 1/f = 1/d_o + 1/d_i và ma trận truyền tia ABCD cho phép xác định chính xác vị trí, độ lớn và tính chất ảnh. "
            "Thấu kính hội tụ (f > 0) cho ảnh thật ngược chiều khi d_o > f và ảnh ảo phóng đại khi d_o < f. Thấu kính phân kỳ (f < 0) luôn tạo ảnh ảo cùng chiều thu nhỏ."
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

print("Successfully added create_thin_lens_system_diagram to stem_tools.py!")
