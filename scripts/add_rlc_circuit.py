"""
Insert create_rlc_circuit_diagram into PhysicsTools in src/ui/stem_tools.py
"""

def main():
    with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if 'def create_rlc_circuit_diagram(' in content:
        print("create_rlc_circuit_diagram already exists!")
        return

    method_code = '''    @staticmethod
    def create_rlc_circuit_diagram(
        center: tuple[float, float],
        width: float = 480.0,
        height: float = 300.0,
        voltage_u: float = 220.0,
        frequency_f: float = 50.0,
        resistor_r: float = 50.0,
        inductor_l: float = 0.3183,
        capacitor_c_uf: float = 31.83,
        impedance_z: Optional[float] = None,
        current_i: Optional[float] = None,
        phase_deg: Optional[float] = None,
        preset: str = "RESONANCE_50HZ",
    ) -> list[QGraphicsItem]:
        """Creates an authentic AC Series RLC & Fresnel Phasor vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        omega = 2.0 * math.pi * frequency_f
        zl = omega * inductor_l
        zc = 1.0 / (omega * capacitor_c_uf * 1e-6) if omega > 0 and capacitor_c_uf > 0 else 0.0
        z_val = impedance_z if impedance_z is not None else math.sqrt(resistor_r * resistor_r + (zl - zc) ** 2)
        i_val = current_i if current_i is not None else (voltage_u / max(0.01, z_val))
        phi_deg = phase_deg if phase_deg is not None else math.degrees(math.atan((zl - zc) / max(0.01, resistor_r)))

        # 1. Title Banner
        banner_txt = f"MẠCH ĐIỆN XOAY CHIỀU RLC (f = {frequency_f:.1f}Hz, Z = {z_val:.1f}Ω, I = {i_val:.2f}A, φ = {phi_deg:+.1f}°)"
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

        # 3. Schematic Loop on Left (Top portion of viewport)
        sch_y = pane_y + 35.0
        r_rect = QGraphicsRectItem(pane_x + 35.0, sch_y - 7.0, 36.0, 14.0)
        r_rect.setBrush(QBrush(QColor("#1e293b")))
        r_rect.setPen(QPen(QColor("#38bdf8"), 1.2))
        r_rect.setZValue(10.2)
        items.append(r_rect)

        lbl_r = QGraphicsTextItem(f"R={resistor_r:.0f}Ω")
        lbl_r.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_r.setDefaultTextColor(QColor("#38bdf8"))
        lbl_r.setPos(pane_x + 32.0, sch_y - 20.0)
        lbl_r.setZValue(10.3)
        items.append(lbl_r)

        # Inductor L
        ind_path = QPainterPath()
        ind_path.moveTo(pane_x + 95.0, sch_y)
        for k in range(3):
            ind_path.arcTo(QRectF(pane_x + 95.0 + k * 12.0, sch_y - 8.0, 12.0, 16.0), 180.0, -180.0)
        ind_item = QGraphicsPathItem(ind_path)
        ind_item.setPen(QPen(QColor("#10b981"), 1.4))
        ind_item.setZValue(10.2)
        items.append(ind_item)

        lbl_l = QGraphicsTextItem(f"L={inductor_l:.2f}H")
        lbl_l.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_l.setDefaultTextColor(QColor("#10b981"))
        lbl_l.setPos(pane_x + 95.0, sch_y - 20.0)
        lbl_l.setZValue(10.3)
        items.append(lbl_l)

        # Capacitor C
        cap_x = pane_x + 160.0
        c_plate1 = QGraphicsLineItem(cap_x, sch_y - 9.0, cap_x, sch_y + 9.0)
        c_plate1.setPen(QPen(QColor("#f59e0b"), 2.0))
        c_plate1.setZValue(10.2)
        items.append(c_plate1)

        c_plate2 = QGraphicsLineItem(cap_x + 6.0, sch_y - 9.0, cap_x + 6.0, sch_y + 9.0)
        c_plate2.setPen(QPen(QColor("#f59e0b"), 2.0))
        c_plate2.setZValue(10.2)
        items.append(c_plate2)

        lbl_c = QGraphicsTextItem(f"C={capacitor_c_uf:.1f}μF")
        lbl_c.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_c.setDefaultTextColor(QColor("#f59e0b"))
        lbl_c.setPos(cap_x - 6.0, sch_y - 20.0)
        lbl_c.setZValue(10.3)
        items.append(lbl_c)

        # Wire connections
        wire1 = QGraphicsLineItem(pane_x + 15.0, sch_y, pane_x + 35.0, sch_y)
        wire1.setPen(QPen(QColor("#64748b"), 1.4))
        wire1.setZValue(10.2)
        items.append(wire1)

        wire2 = QGraphicsLineItem(pane_x + 71.0, sch_y, pane_x + 95.0, sch_y)
        wire2.setPen(QPen(QColor("#64748b"), 1.4))
        wire2.setZValue(10.2)
        items.append(wire2)

        wire3 = QGraphicsLineItem(pane_x + 131.0, sch_y, cap_x, sch_y)
        wire3.setPen(QPen(QColor("#64748b"), 1.4))
        wire3.setZValue(10.2)
        items.append(wire3)

        wire4 = QGraphicsLineItem(cap_x + 6.0, sch_y, pane_x + 210.0, sch_y)
        wire4.setPen(QPen(QColor("#64748b"), 1.4))
        wire4.setZValue(10.2)
        items.append(wire4)

        # 4. Fresnel Phasor Diagram in Lower Viewport
        fc_x = pane_x + 60.0
        fc_y = pane_y + 160.0
        scale_v = 0.35

        # Horizontal U_R vector
        px_ur = min(70.0, (i_val * resistor_r) * scale_v)
        vec_ur = QGraphicsLineItem(fc_x, fc_y, fc_x + px_ur, fc_y)
        vec_ur.setPen(QPen(QColor("#38bdf8"), 2.0))
        vec_ur.setZValue(10.5)
        items.append(vec_ur)

        lbl_vur = QGraphicsTextItem("U_R")
        lbl_vur.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_vur.setDefaultTextColor(QColor("#38bdf8"))
        lbl_vur.setPos(fc_x + px_ur + 2.0, fc_y - 2.0)
        lbl_vur.setZValue(10.6)
        items.append(lbl_vur)

        # Vertical U_L and U_C
        px_ul = min(50.0, (i_val * zl) * scale_v)
        px_uc = min(50.0, (i_val * zc) * scale_v)
        if px_ul > 1.0:
            vec_ul = QGraphicsLineItem(fc_x, fc_y, fc_x, fc_y - px_ul)
            vec_ul.setPen(QPen(QColor("#10b981"), 1.8))
            vec_ul.setZValue(10.5)
            items.append(vec_ul)

        if px_uc > 1.0:
            vec_uc = QGraphicsLineItem(fc_x, fc_y, fc_x, fc_y + px_uc)
            vec_uc.setPen(QPen(QColor("#f59e0b"), 1.8))
            vec_uc.setZValue(10.5)
            items.append(vec_uc)

        # Resultant U vector
        px_ulc = px_ul - px_uc
        vec_u = QGraphicsLineItem(fc_x, fc_y, fc_x + px_ur, fc_y - px_ulc)
        vec_u.setPen(QPen(QColor("#f43f5e"), 2.2))
        vec_u.setZValue(10.7)
        items.append(vec_u)

        lbl_vu = QGraphicsTextItem(f"U ({voltage_u:.0f}V)")
        lbl_vu.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_vu.setDefaultTextColor(QColor("#f43f5e"))
        lbl_vu.setPos(fc_x + px_ur + 2.0, fc_y - px_ulc - 8.0)
        lbl_vu.setZValue(10.8)
        items.append(lbl_vu)

        # 5. Right-side Theory & Metrics Card
        card_x = pane_x + pane_w + 12.0
        card_y = pane_y
        card_w = width - pane_w - 34.0
        card_h = pane_h

        info_bg = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_bg.setBrush(QBrush(QColor(30, 41, 59, 230)))
        info_bg.setPen(QPen(QColor("#334155"), 1.0))
        info_bg.setZValue(10.0)
        items.append(info_bg)

        # Resonance frequency
        f0_hz = 1.0 / (2.0 * math.pi * math.sqrt(inductor_l * capacitor_c_uf * 1e-6)) if inductor_l > 0 and capacitor_c_uf > 0 else 0.0
        q_fac = (1.0 / resistor_r) * math.sqrt(inductor_l / (capacitor_c_uf * 1e-6)) if resistor_r > 0 and capacitor_c_uf > 0 else 0.0
        cos_phi = resistor_r / max(0.01, z_val)
        p_act = voltage_u * i_val * cos_phi

        lines = [
            ("LÝ THUYẾT ĐIỆN XOAY CHIỀU RLC", QColor("#f8fafc"), True, 9),
            (f"• Cảm kháng: Z_L = 2πfL = {zl:.1f} Ω", QColor("#10b981"), False, 8),
            (f"• Dung kháng: Z_C = 1/(2πfC) = {zc:.1f} Ω", QColor("#f59e0b"), False, 8),
            (f"• Tổng trở: Z = √(R² + (Z_L−Z_C)²) = {z_val:.1f} Ω", QColor("#38bdf8"), True, 8),
            (f"• Cường độ dòng điện: I = U/Z = {i_val:.2f} A", QColor("#f8fafc"), False, 8),
            (f"• Độ lệch pha: φ = arctan(X/R) = {phi_deg:+.1f}°", QColor("#f43f5e"), False, 8),
            (f"• Hệ số công suất: cos(φ) = {cos_phi:.3f}", QColor("#cbd5e1"), False, 8),
            (f"• Công suất tiêu thụ: P = UI·cos(φ) = {p_act:.1f} W", QColor("#cbd5e1"), False, 8),
            (f"• Tần số cộng hưởng: f₀ = {f0_hz:.1f} Hz (Q = {q_fac:.2f})", QColor("#fbbf24"), True, 8),
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
    print("Successfully inserted create_rlc_circuit_diagram into stem_tools.py!")

if __name__ == '__main__':
    main()
