import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        cap = QGraphicsTextItem(
            'Hệ phương trình Maxwell khẳng định sóng điện từ lan truyền trong chân không với vận tốc ánh sáng c = 1/√(ε₀μ₀). '
            'Vectơ điện E⃗ và từ B⃗ luôn dao động vuông góc với nhau và vuông góc với phương truyền sóng k⃗. '
            'Vectơ Poynting S⃗ = (1/μ₀)(E⃗ × B⃗) định lượng dòng năng lượng và áp suất bức xạ truyền tải trên mỗi đơn vị diện tích.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items'''

tool_code = '''

    @staticmethod
    def create_fabry_perot_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "HELIUM_NEON_LASER_CAVITY",
        wavelength_nm: float = 632.8,
        cavity_d_um: float = 20.0,
        reflectance_r: float = 0.96,
        finesse: float = 76.8,
        fsr_ghz: float = 7494.8,
        fwhm_ghz: float = 97.6,
        transmittance: float = 1.0,
        order_m: float = 63.2,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of a Fabry-Pérot Cavity & Airy Transmission Spectrum.
        Includes dual semi-reflective dielectric mirrors, multi-beam internal bounces,
        transmitted interference rays, and plotted Airy resonance curve with FSR and FWHM metrics.
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
        title_txt = QGraphicsTextItem(f"GIAO THOA KẾ FABRY-PÉROT & HỐC CỘNG HƯỞNG LASER: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Bước sóng λ = {wavelength_nm:.1f} nm | Khoảng cách gương d = {cavity_d_um:.1f} μm | "
            f"Phản xạ R = {reflectance_r:.3f} | Finesse F = {finesse:.1f} | FSR = {fsr_ghz:.1f} GHz"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Cavity & Multi-beam reflections
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("HỐC QUANG HỌC & ĐA PHẢN XẠ NHIỀU CHÙM TIA:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # Mirrors M1 & M2
        m1_x = cx + cw * 0.28
        m2_x = cx + cw * 0.74
        m_top = cy + 30.0
        m_h = ch - 50.0
        m_thick = 12.0

        m1 = QGraphicsRectItem(m1_x - m_thick, m_top, m_thick, m_h)
        m1.setBrush(QBrush(QColor(56, 189, 248, 180)))
        m1.setPen(QPen(QColor(148, 163, 184), 1.2))
        m1.setZValue(10.3)
        items.append(m1)

        m2 = QGraphicsRectItem(m2_x, m_top, m_thick, m_h)
        m2.setBrush(QBrush(QColor(56, 189, 248, 180)))
        m2.setPen(QPen(QColor(148, 163, 184), 1.2))
        m2.setZValue(10.3)
        items.append(m2)

        # Incident Beam
        beam_y = m_top + 35.0
        in_line = QGraphicsLineItem(cx + 10.0, beam_y, m1_x, beam_y)
        in_line.setPen(QPen(QColor("#ef4444" if wavelength_nm > 600 else "#22c55e"), 2.2))
        in_line.setZValue(10.3)
        items.append(in_line)

        # Internal Multi-reflection path
        b_path = QPainterPath()
        b_path.moveTo(m1_x, beam_y)
        b_path.lineTo(m2_x, beam_y + 20.0)
        b_path.lineTo(m1_x, beam_y + 40.0)
        b_path.lineTo(m2_x, beam_y + 60.0)
        b_path.lineTo(m1_x, beam_y + 80.0)
        b_item = QGraphicsPathItem(b_path)
        b_item.setPen(QPen(QColor("#fca5a5" if wavelength_nm > 600 else "#86efac"), 1.8))
        b_item.setZValue(10.3)
        items.append(b_item)

        # Transmitted output rays
        for ty in [beam_y + 20.0, beam_y + 60.0]:
            t_line = QGraphicsLineItem(m2_x + m_thick, ty, cx + cw - 10.0, ty)
            t_line.setPen(QPen(QColor("#ef4444" if wavelength_nm > 600 else "#22c55e"), 1.6))
            t_line.setZValue(10.3)
            items.append(t_line)

        # 4. Right Panel: Airy Transmission Curve
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("ĐƯỜNG TRUYỀN QUA HÀM AIRY T(δ) & VẠCH CỘNG HƯỞNG:")
        r_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_title.setDefaultTextColor(QColor("#38bdf8"))
        r_title.setPos(rx + 8.0, ry + 6.0)
        r_title.setZValue(10.2)
        items.append(r_title)

        # Airy curve plot
        px = rx + 30.0
        py = ry + 25.0
        pw = rw - 45.0
        ph = rh - 85.0

        # Plot axes
        axis_x = QGraphicsLineItem(px, py + ph, px + pw, py + ph)
        axis_x.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_x.setZValue(10.2)
        items.append(axis_x)

        axis_y = QGraphicsLineItem(px, py, px, py + ph)
        axis_y.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_y.setZValue(10.2)
        items.append(axis_y)

        # Calculate Airy path over 3 FSR periods
        f_coeff = (4.0 * reflectance_r) / max(1e-4, (1.0 - reflectance_r) ** 2)
        a_path = QPainterPath()
        steps = int(pw)
        delta_span = 6.0 * 3.14159265
        for s in range(steps + 1):
            delta = -3.0 * 3.14159265 + (s / float(steps)) * delta_span
            t_val = 1.0 / (1.0 + f_coeff * (math.sin(delta / 2.0) ** 2))
            gx = px + s
            gy = py + ph - t_val * ph
            if s == 0:
                a_path.moveTo(gx, gy)
            else:
                a_path.lineTo(gx, gy)

        a_item = QGraphicsPathItem(a_path)
        a_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        a_item.setZValue(10.4)
        items.append(a_item)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Finesse F = π√R/(1-R): {finesse:.1f}    • Hệ số F = 4R/(1-R)²: {f_coeff:.1f}    • Khoảng phổ tự do FSR: {fsr_ghz:.1f} GHz\\n"
            f"• Bề rộng nửa cực đại FWHM δν: {fwhm_ghz:.2f} GHz    • Bậc mode dọc m: {order_m:.1f}    • Độ truyền qua cực đại: {transmittance*100:.1f}%"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, py + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
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

text = text.replace('\r\n', '\n')
target_clean = target.replace('\r\n', '\n')

if target_clean not in text:
    print("Could not find target in stem_tools.py")
    sys.exit(1)

text = text.replace(target_clean, target_clean + tool_code, 1)

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully added create_fabry_perot_diagram to stem_tools.py!")
