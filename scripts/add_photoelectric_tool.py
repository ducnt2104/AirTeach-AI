import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Hàm truyền qua Airy T(δ) = 1 / [1 + F sin²(δ/2)] mô tả giao thoa đa chùm tia phản xạ giữa hai gương song song. "
            "Khi phản xạ R tiến gần 1.0, độ sắc nét Finesse F tăng vọt, các đỉnh truyền qua trở nên siêu hẹp (FWHM << FSR), "
            "cho phép chọn lọc chính xác các mode dọc quang học trong buồng cộng hưởng laser và phân giải siêu cao phổ vạch nguyên tử."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items'''

tool_code = '''

    @staticmethod
    def create_photoelectric_effect_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CESIUM_VISIBLE_LIGHT_THRESHOLD",
        metal_name: str = "Xesi (Cs)",
        work_function_ev: float = 2.14,
        wavelength_nm: float = 532.0,
        photon_energy_ev: float = 2.33,
        stopping_potential_v: float = 0.19,
        voltage_uak_v: float = 1.0,
        photocurrent_ua: float = 5.0,
        red_threshold_nm: float = 579.4,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Einstein's Photoelectric Effect.
        Includes evacuated quartz phototube, cathode metal plate, anode ring, light rays,
        photoelectron trajectories, current-voltage I-U_AK curve, and quantum metrics.
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
        title_txt = QGraphicsTextItem(f"HIỆU ỨNG QUANG ĐIỆN EINSTEIN & THUYẾT LƯỢNG TỬ: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Catot: {metal_name} (Φ = {work_function_ev:.2f} eV) | λ = {wavelength_nm:.1f} nm (E = {photon_energy_ev:.2f} eV) | "
            f"Thế hãm U_h = {stopping_potential_v:.2f} V | U_AK = {voltage_uak_v:+.2f} V | I = {photocurrent_ua:.2f} μA"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Quartz Vacuum Phototube
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("TẾ BÀO QUANG ĐIỆN CHÂN KHÔNG:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # Quartz bulb
        bulb = QGraphicsRectItem(cx + 25.0, cy + 28.0, cw - 50.0, ch - 56.0)
        bulb.setBrush(QBrush(QColor(15, 23, 42, 120)))
        bulb.setPen(QPen(QColor(56, 189, 248, 160), 1.5))
        bulb.setZValue(10.2)
        items.append(bulb)

        # Cathode Plate
        cat_x = cx + 45.0
        cat_y = cy + 45.0
        cat_w = 8.0
        cat_h = ch - 90.0
        cat = QGraphicsRectItem(cat_x, cat_y, cat_w, cat_h)
        cat.setBrush(QBrush(QColor(250, 204, 21)))
        cat.setPen(QPen(QColor(248, 250, 252), 1.2))
        cat.setZValue(10.3)
        items.append(cat)

        cat_lbl = QGraphicsTextItem("Catot (K)")
        cat_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        cat_lbl.setDefaultTextColor(QColor("#e2e8f0"))
        cat_lbl.setPos(cat_x - 8.0, cat_y - 14.0)
        cat_lbl.setZValue(10.3)
        items.append(cat_lbl)

        # Anode Ring
        an_x = cx + cw - 55.0
        an = QGraphicsEllipseItem(an_x, cat_y + 10.0, 10.0, cat_h - 20.0)
        an.setBrush(Qt.BrushStyle.NoBrush)
        an.setPen(QPen(QColor(148, 163, 184), 2.0))
        an.setZValue(10.3)
        items.append(an)

        an_lbl = QGraphicsTextItem("Anot (A)")
        an_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        an_lbl.setDefaultTextColor(QColor("#e2e8f0"))
        an_lbl.setPos(an_x - 4.0, cat_y - 14.0)
        an_lbl.setZValue(10.3)
        items.append(an_lbl)

        # Light Rays
        ray_color = QColor("#22c55e" if wavelength_nm > 500 else "#a855f7")
        for ry in [cat_y + 15.0, cat_y + 40.0, cat_y + 65.0]:
            ray = QGraphicsLineItem(cx + 8.0, ry - 15.0, cat_x, ry)
            ray.setPen(QPen(ray_color, 1.5, Qt.PenStyle.DashLine))
            ray.setZValue(10.3)
            items.append(ray)

        # Photoelectrons
        if photon_energy_ev >= work_function_ev:
            for ex, ey in [(cat_x + 30.0, cat_y + 25.0), (cat_x + 70.0, cat_y + 35.0), (cat_x + 110.0, cat_y + 20.0), (cat_x + 150.0, cat_y + 45.0)]:
                elec = QGraphicsEllipseItem(ex, ey, 5.0, 5.0)
                elec.setBrush(QBrush(QColor("#38bdf8")))
                elec.setPen(Qt.PenStyle.NoPen)
                elec.setZValue(10.4)
                items.append(elec)

        # Microammeter badge
        meter_bg = QGraphicsRectItem(cx + cw - 95.0, cy + ch - 24.0, 85.0, 18.0)
        meter_bg.setBrush(QBrush(QColor(9, 14, 25)))
        meter_bg.setPen(QPen(QColor(56, 189, 248), 1.0))
        meter_bg.setZValue(10.3)
        items.append(meter_bg)

        meter_txt = QGraphicsTextItem(f"I: {photocurrent_ua:.2f} μA")
        meter_txt.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        meter_txt.setDefaultTextColor(QColor("#22c55e" if photocurrent_ua > 0 else "#94a3b8"))
        meter_txt.setPos(cx + cw - 92.0, cy + ch - 24.0)
        meter_txt.setZValue(10.4)
        items.append(meter_txt)

        # 4. Right Panel: Current-Voltage Curve I - U_AK
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("ĐƯỜNG ĐẶC TRƯNG VÔN-AMPE I(U_AK):")
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

        zero_x = px + pw * 0.45
        axis_y = QGraphicsLineItem(zero_x, py, zero_x, py + ph)
        axis_y.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_y.setZValue(10.2)
        items.append(axis_y)

        # Plot I - U_AK curve
        iv_path = QPainterPath()
        steps = int(pw)
        uh_pt_x = zero_x - (stopping_potential_v / 3.0) * (zero_x - px)
        uh_pt_x = max(px, uh_pt_x)

        for s in range(steps + 1):
            gx = px + s
            if gx <= uh_pt_x:
                gy = py + ph
            elif gx < zero_x:
                frac = (gx - uh_pt_x) / max(1.0, (zero_x - uh_pt_x))
                gy = py + ph - (frac ** 2) * (ph * 0.7)
            else:
                gy = py + ph - ph * 0.7
            if s == 0:
                iv_path.moveTo(gx, gy)
            else:
                iv_path.lineTo(gx, gy)

        iv_item = QGraphicsPathItem(iv_path)
        iv_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        iv_item.setZValue(10.4)
        items.append(iv_item)

        # -Uh marker
        uh_line = QGraphicsLineItem(uh_pt_x, py + 10.0, uh_pt_x, py + ph)
        uh_line.setPen(QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DashLine))
        uh_line.setZValue(10.3)
        items.append(uh_line)

        uh_lbl = QGraphicsTextItem(f"-U_h = -{stopping_potential_v:.2f}V")
        uh_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        uh_lbl.setDefaultTextColor(QColor("#f43f5e"))
        uh_lbl.setPos(uh_pt_x - 18.0, py - 4.0)
        uh_lbl.setZValue(10.3)
        items.append(uh_lbl)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Năng lượng photon E = hc/λ: {photon_energy_ev:.2f} eV    • Công thoát Φ: {work_function_ev:.2f} eV    • Giới hạn quang điện λ₀: {red_threshold_nm:.1f} nm\\n"
            f"• Động năng cực đại E_k: {max(0.0, photon_energy_ev - work_function_ev):.2f} eV    • Thế hãm U_h: {stopping_potential_v:.2f} V    • Dòng quang điện I: {photocurrent_ua:.2f} μA"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, py + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
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

text = text.replace('\r\n', '\n')
target_clean = target.replace('\r\n', '\n')

if target_clean not in text:
    print("Could not find target in stem_tools.py")
    sys.exit(1)

text = text.replace(target_clean, target_clean + tool_code, 1)

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully added create_photoelectric_effect_diagram to stem_tools.py!")
