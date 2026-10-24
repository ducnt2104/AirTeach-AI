"""
Insert create_thin_lens_diagram into PhysicsTools in src/ui/stem_tools.py
"""

def main():
    with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if 'def create_thin_lens_diagram(' in content:
        print("create_thin_lens_diagram already exists!")
        return

    method_code = '''    @staticmethod
    def create_thin_lens_diagram(
        center: tuple[float, float],
        width: float = 480.0,
        height: float = 300.0,
        focal_length_cm: float = 12.0,
        object_d_cm: float = 18.0,
        object_h_cm: float = 6.0,
        lens_type: str = "CONVERGING",
        preset: str = "PROJECTOR_LENS",
    ) -> list[QGraphicsItem]:
        """Creates an authentic Thin Lens & Optical Bench ray tracing vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        f_val = focal_length_cm
        d_val = object_d_cm
        h_val = object_h_cm
        scale = 5.5

        # Image math: 1/f = 1/d + 1/d'
        if abs(d_val - f_val) < 0.05:
            d_prime = 999.0
            k_val = 99.0
        else:
            d_prime = (d_val * f_val) / (d_val - f_val)
            k_val = -d_prime / d_val

        # 1. Title Banner
        mode_str = "HỘI TỤ (f > 0)" if lens_type.upper() == "CONVERGING" else "PHÂN KỲ (f < 0)"
        banner_txt = f"QUANG HÌNH HỌC: THẤU KÍNH {mode_str} (f = {f_val:+.1f}cm, d = {d_val:.1f}cm, d' = {d_prime:+.1f}cm)"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        # 2. Viewport Dark Cosmic Pane
        pane_x = x_left + 10.0
        pane_y = y_top + 10.0
        pane_w = 280.0
        pane_h = 240.0

        bg_pane = QGraphicsRectItem(pane_x, pane_y, pane_w, pane_h)
        bg_pane.setBrush(QBrush(QColor("#030712")))
        bg_pane.setPen(QPen(QColor("#1e293b"), 1.2))
        bg_pane.setZValue(10.0)
        items.append(bg_pane)

        ox = pane_x + 130.0
        oy = pane_y + pane_h / 2.0

        # 3. Optical Axis
        axis_line = QGraphicsLineItem(pane_x + 5.0, oy, pane_x + pane_w - 5.0, oy)
        axis_line.setPen(QPen(QColor("#64748b"), 1.2))
        axis_line.setZValue(10.2)
        items.append(axis_line)

        # 4. Thin Lens Plane
        lens_line = QGraphicsLineItem(ox, pane_y + 15.0, ox, pane_y + pane_h - 15.0)
        lens_line.setPen(QPen(QColor("#38bdf8"), 2.0))
        lens_line.setZValue(10.4)
        items.append(lens_line)

        # 5. Focal Points F and F'
        f_px = abs(f_val) * scale
        fx = ox - f_px if lens_type.upper() == "CONVERGING" else ox + f_px
        fpx = ox + f_px if lens_type.upper() == "CONVERGING" else ox - f_px

        dot_f = QGraphicsEllipseItem(fx - 3.0, oy - 3.0, 6.0, 6.0)
        dot_f.setBrush(QBrush(QColor("#fbbf24")))
        dot_f.setPen(Qt.PenStyle.NoPen)
        dot_f.setZValue(10.5)
        items.append(dot_f)

        dot_fp = QGraphicsEllipseItem(fpx - 3.0, oy - 3.0, 6.0, 6.0)
        dot_fp.setBrush(QBrush(QColor("#fbbf24")))
        dot_fp.setPen(Qt.PenStyle.NoPen)
        dot_fp.setZValue(10.5)
        items.append(dot_fp)

        # 6. Object Arrow AB
        obj_x = ox - d_val * scale
        obj_y_tip = oy - h_val * scale
        obj_arrow = QGraphicsLineItem(obj_x, oy, obj_x, obj_y_tip)
        obj_arrow.setPen(QPen(QColor("#f59e0b"), 2.2))
        obj_arrow.setZValue(10.6)
        items.append(obj_arrow)

        lbl_ab = QGraphicsTextItem("AB")
        lbl_ab.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_ab.setDefaultTextColor(QColor("#fbbf24"))
        lbl_ab.setPos(obj_x - 14.0, obj_y_tip - 10.0)
        lbl_ab.setZValue(10.7)
        items.append(lbl_ab)

        # 7. Image Arrow A'B'
        if abs(d_prime) < 100.0:
            img_x = ox + d_prime * scale
            img_y_tip = oy - (k_val * h_val) * scale
            is_real = d_prime > 0
            img_col = QColor("#10b981") if is_real else QColor("#ec4899")
            img_arrow = QGraphicsLineItem(img_x, oy, img_x, img_y_tip)
            img_arrow.setPen(QPen(img_col, 2.2, Qt.PenStyle.SolidLine if is_real else Qt.PenStyle.DashLine))
            img_arrow.setZValue(10.6)
            items.append(img_arrow)

            lbl_apbp = QGraphicsTextItem("A'B'")
            lbl_apbp.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            lbl_apbp.setDefaultTextColor(img_col)
            lbl_apbp.setPos(img_x + 4.0, img_y_tip - 8.0)
            lbl_apbp.setZValue(10.7)
            items.append(lbl_apbp)

        # 8. Principal Rays
        # Ray 1: Parallel to axis, refracts through F'
        ray1_in = QGraphicsLineItem(obj_x, obj_y_tip, ox, obj_y_tip)
        ray1_in.setPen(QPen(QColor("#eab308"), 1.4))
        ray1_in.setZValue(10.3)
        items.append(ray1_in)

        ray1_out = QGraphicsLineItem(ox, obj_y_tip, fpx, oy)
        ray1_out.setPen(QPen(QColor("#eab308"), 1.4))
        ray1_out.setZValue(10.3)
        items.append(ray1_out)

        # Ray 2: Central ray through O
        ray2 = QGraphicsLineItem(obj_x, obj_y_tip, ox, oy)
        ray2.setPen(QPen(QColor("#f97316"), 1.4))
        ray2.setZValue(10.3)
        items.append(ray2)

        # 9. Right-side Theory & Metrics Card
        card_x = pane_x + pane_w + 12.0
        card_y = pane_y
        card_w = width - pane_w - 34.0
        card_h = pane_h

        info_bg = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_bg.setBrush(QBrush(QColor(30, 41, 59, 230)))
        info_bg.setPen(QPen(QColor("#334155"), 1.0))
        info_bg.setZValue(10.0)
        items.append(info_bg)

        nature_txt = "ẢNH THẬT, NGƯỢC CHIỀU" if d_prime > 0 else "ẢNH ẢO, CÙNG CHIỀU"
        mag_str = "Phóng đại" if abs(k_val) > 1.0 else ("Thu nhỏ" if abs(k_val) < 1.0 else "Bằng vật")
        diopter = 100.0 / f_val if f_val != 0 else 0.0

        lines = [
            ("LÝ THUYẾT QUANG HỌC THẤU KÍNH", QColor("#f8fafc"), True, 9),
            (f"• Loại thấu kính: {mode_str}", QColor("#38bdf8"), False, 8),
            (f"• Tiêu cự: f = {f_val:+.1f} cm (D = {diopter:+.2f} dp)", QColor("#fbbf24"), False, 8),
            (f"• Khoảng cách vật: d = {d_val:.1f} cm", QColor("#e2e8f0"), False, 8),
            (f"• Khoảng cách ảnh: d' = {d_prime:+.1f} cm", QColor("#10b981") if d_prime > 0 else QColor("#ec4899"), True, 8),
            (f"• Tính chất: {nature_txt} ({mag_str})", QColor("#f8fafc"), False, 8),
            (f"• Độ phóng đại: k = -d'/d = {k_val:+.2f}", QColor("#38bdf8"), False, 8),
            ("• Công thức Gauss: 1/f = 1/d + 1/d'", QColor("#94a3b8"), False, 8),
            ("• 3 tia đặc biệt: Song song (F'), Quang tâm (O), Tiêu điểm (F)", QColor("#94a3b8"), False, 8),
        ]

        text_y = card_y + 8.0
        for txt, color, is_bold, sz in lines:
            t_item = QGraphicsTextItem(txt)
            t_item.setFont(QFont("Segoe UI", sz, QFont.Weight.Bold if is_bold else QFont.Weight.Normal))
            t_item.setDefaultTextColor(color)
            t_item.setPos(card_x + 8.0, text_y)
            t_item.setZValue(10.5)
            items.append(t_item)
            text_y += 24.0

        return items
'''
    target = 'class ChemistryTools:'
    assert target in content, "target not found"
    new_content = content.replace(target, method_code + '\n\n' + target, 1)

    with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully inserted create_thin_lens_diagram into stem_tools.py!")

if __name__ == '__main__':
    main()
