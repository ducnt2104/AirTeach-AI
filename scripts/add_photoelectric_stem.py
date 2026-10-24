"""
Script to insert create_photoelectric_effect_diagram into PhysicsTools in src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "class ChemistryTools:"
    if target not in content:
        print("ERROR: Target class ChemistryTools not found!")
        sys.exit(1)

    if "create_photoelectric_effect_diagram" in content:
        print("create_photoelectric_effect_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_photoelectric_effect_diagram(
        center: tuple[float, float],
        width: float = 520.0,
        height: float = 320.0,
        metal_key: str = "Cs",
        wavelength_nm: float = 450.0,
        voltage_uak: float = 1.5,
        intensity_pct: float = 80.0,
        preset: str = "CESIUM_VISIBLE",
    ) -> list[QGraphicsItem]:
        """Creates an authentic Modern Physics Photoelectric Effect vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        # Authentic work functions
        metals_data = {
            "Cs": ("Xesi (Cesium)", 2.14, "#fbbf24"),
            "K": ("Kali (Potassium)", 2.30, "#c084fc"),
            "Na": ("Natri (Sodium)", 2.46, "#38bdf8"),
            "Zn": ("Kẽm (Zinc)", 4.31, "#94a3b8"),
            "Pt": ("Bạch kim (Platinum)", 6.35, "#e2e8f0"),
        }
        m_name, a_ev, m_color = metals_data.get(metal_key, metals_data["Cs"])

        # Physics formulas:
        # ε = hc / λ (eV)
        # λ₀ = hc / A (nm)
        hc_ev_nm = 1239.841984
        e_charge = 1.602176634e-19
        m_electron = 9.1093837e-31

        photon_energy = hc_ev_nm / max(100.0, wavelength_nm)
        lambda_0 = hc_ev_nm / a_ev if a_ev > 0 else 9999.0
        is_emitted = photon_energy >= a_ev

        if is_emitted:
            ek_max_ev = photon_energy - a_ev
            ek_max_j = ek_max_ev * e_charge
            v_max = math.sqrt(2.0 * ek_max_j / m_electron)
            u_h = ek_max_ev
        else:
            ek_max_ev = 0.0
            v_max = 0.0
            u_h = 0.0

        # Photocurrent
        if not is_emitted or voltage_uak <= -u_h:
            photocurrent = 0.0
        else:
            i_sat = 25.0 * (intensity_pct / 100.0)
            u_sat = 1.5
            if voltage_uak >= u_sat:
                photocurrent = i_sat
            else:
                frac = max(0.0, min(1.0, (voltage_uak + u_h) / (u_sat + u_h)))
                photocurrent = i_sat * (frac ** 1.35)

        # 1. Title Banner
        status_banner = "PHÁT QUANG ĐIỆN" if is_emitted else "DƯỚI NGƯỠNG (KHÔNG PHÁT)"
        banner_txt = f"HIỆU ỨNG QUANG ĐIỆN & THUYẾT LƯỢNG TỬ EINSTEIN [{m_name}, λ={wavelength_nm:.0f}nm] - {status_banner}"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        # 2. Viewport Dark Cosmic Pane (Left)
        pane_x = x_left + 10.0
        pane_y = y_top + 10.0
        pane_w = 260.0
        pane_h = 290.0

        bg_pane = QGraphicsRectItem(pane_x, pane_y, pane_w, pane_h)
        bg_pane.setBrush(QBrush(QColor("#030712")))
        bg_pane.setPen(QPen(QColor("#1e293b"), 1.2))
        bg_pane.setZValue(10.0)
        items.append(bg_pane)

        # Glass Tube (Capsule)
        tube_rect = QRectF(pane_x + 15.0, pane_y + 40.0, 230.0, 160.0)
        tube = QGraphicsRectItem(tube_rect)
        tube.setBrush(QBrush(QColor(15, 23, 42, 180)))
        tube.setPen(QPen(QColor("#38bdf8"), 1.5))
        tube.setZValue(10.2)
        items.append(tube)

        # Cathode Plate (Left)
        cat_rect = QRectF(pane_x + 40.0, pane_y + 65.0, 10.0, 110.0)
        cathode = QGraphicsRectItem(cat_rect)
        cathode.setBrush(QBrush(QColor(m_color)))
        cathode.setPen(QPen(QColor("#ffffff"), 1.0))
        cathode.setZValue(10.5)
        items.append(cathode)

        cat_lbl = QGraphicsTextItem("K (Catôt)")
        cat_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        cat_lbl.setDefaultTextColor(QColor("#f8fafc"))
        cat_lbl.setPos(cat_rect.left() - 10.0, cat_rect.bottom() + 4.0)
        cat_lbl.setZValue(10.6)
        items.append(cat_lbl)

        # Anode Plate (Right)
        an_rect = QRectF(pane_x + 190.0, pane_y + 65.0, 8.0, 110.0)
        anode = QGraphicsRectItem(an_rect)
        anode.setBrush(QBrush(QColor("#475569")))
        anode.setPen(QPen(QColor("#94a3b8"), 1.0))
        anode.setZValue(10.5)
        items.append(anode)

        an_lbl = QGraphicsTextItem("A (Anôt)")
        an_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        an_lbl.setDefaultTextColor(QColor("#f8fafc"))
        an_lbl.setPos(an_rect.left() - 10.0, an_rect.bottom() + 4.0)
        an_lbl.setZValue(10.6)
        items.append(an_lbl)

        # Light Beam
        beam_color = QColor("#9333ea") if wavelength_nm < 380 else (QColor("#3b82f6") if wavelength_nm < 490 else (QColor("#10b981") if wavelength_nm < 580 else QColor("#ef4444")))
        beam_path = QPainterPath()
        beam_path.moveTo(pane_x + 10.0, pane_y + 15.0)
        beam_path.lineTo(pane_x + 40.0, pane_y + 15.0)
        beam_path.lineTo(cat_rect.right(), cat_rect.center().y() + 20.0)
        beam_path.lineTo(cat_rect.right(), cat_rect.center().y() - 30.0)
        beam_path.closeSubpath()

        beam_item = QGraphicsPathItem(beam_path)
        beam_item.setBrush(QBrush(QColor(beam_color.red(), beam_color.green(), beam_color.blue(), 100)))
        beam_item.setPen(QPen(beam_color, 1.0, Qt.PenStyle.DashLine))
        beam_item.setZValue(10.3)
        items.append(beam_item)

        # Photoelectrons and trajectories
        if is_emitted:
            num_electrons = 6
            for idx in range(num_electrons):
                ey = cat_rect.top() + 15.0 + idx * 16.0
                dist_frac = 0.35 + (idx % 3) * 0.25
                ex = cat_rect.right() + dist_frac * (an_rect.left() - cat_rect.right())
                el = QGraphicsEllipseItem(ex - 3.0, ey - 3.0, 6.0, 6.0)
                el.setBrush(QBrush(QColor("#00f5ff")))
                el.setPen(QPen(QColor("#ffffff"), 0.8))
                el.setZValue(10.7)
                items.append(el)

                # Arrow indicating velocity
                if voltage_uak > -u_h:
                    arr = QGraphicsLineItem(ex, ey, ex + 14.0, ey)
                    arr.setPen(QPen(QColor("#00f5ff"), 1.2))
                    arr.setZValue(10.7)
                    items.append(arr)

        # Voltage arrow U_AK and meter
        v_arrow = QGraphicsLineItem(cat_rect.center().x(), pane_y + 225.0, an_rect.center().x(), pane_y + 225.0)
        v_arrow.setPen(QPen(QColor("#10b981") if voltage_uak >= 0 else QColor("#ef4444"), 1.5))
        v_arrow.setZValue(10.5)
        items.append(v_arrow)

        v_txt = QGraphicsTextItem(f"U_AK = {voltage_uak:+.2f}V  |  I = {photocurrent:.2f}μA")
        v_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        v_txt.setDefaultTextColor(QColor("#38bdf8"))
        v_txt.setPos(pane_x + 35.0, pane_y + 235.0)
        v_txt.setZValue(10.6)
        items.append(v_txt)

        # 3. Right Side: Theory & Metrics Card
        card_x = x_left + 280.0
        card_y = y_top + 10.0
        card_w = width - 290.0
        card_h = 290.0

        info_bg = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_bg.setBrush(QBrush(QColor("#0f172a")))
        info_bg.setPen(QPen(QColor("#334155"), 1.0))
        info_bg.setZValue(10.0)
        items.append(info_bg)

        condition_str = "THỎA MÃN" if is_emitted else "KHÔNG THỎA MÃN"
        lines = [
            ("QUANG ĐIỆN & HỆ THỨC EINSTEIN", QColor("#f8fafc"), True, 9),
            (f"• Kim loại Catôt: {m_name}", QColor("#38bdf8"), False, 8),
            (f"• Công thoát A: {a_ev:.2f} eV", QColor("#fbbf24"), False, 8),
            (f"• Giới hạn quang điện λ₀: {lambda_0:.1f} nm", QColor("#fbbf24"), False, 8),
            (f"• Bước sóng chùm sáng λ: {wavelength_nm:.1f} nm", QColor("#cbd5e1"), False, 8),
            (f"• Năng lượng photon ε = hc/λ: {photon_energy:.2f} eV", QColor("#e2e8f0"), False, 8),
            (f"• ĐK quang điện: ε ≥ A => {condition_str}",
             QColor("#10b981") if is_emitted else QColor("#ef4444"), True, 8),
            (f"• Động năng cực đại: E_k,max = {ek_max_ev:.2f} eV", QColor("#38bdf8"), False, 8),
            (f"• Vận tốc ban đầu v_max: {v_max / 1000.0:.1f} km/s", QColor("#38bdf8"), False, 8),
            (f"• Hiệu điện thế hãm U_h = E_k/e: {u_h:.2f} V", QColor("#f43f5e"), True, 8),
            (f"• Phương trình Einstein: h·f = A + e·U_h", QColor("#94a3b8"), False, 8),
            ("• Millikan: Độ dốc U_h(f) là h/e ≈ 4.14×10⁻¹⁵ V·s", QColor("#94a3b8"), False, 8),
        ]

        text_y = card_y + 8.0
        for txt, color, is_bold, sz in lines:
            t_item = QGraphicsTextItem(txt)
            t_item.setFont(QFont("Segoe UI", sz, QFont.Weight.Bold if is_bold else QFont.Weight.Normal))
            t_item.setDefaultTextColor(color)
            t_item.setPos(card_x + 8.0, text_y)
            t_item.setZValue(10.5)
            items.append(t_item)
            text_y += 22.0

        return items

'''

    new_content = content.replace(target, method_code + "\n" + target, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully added create_photoelectric_effect_diagram to stem_tools.py")

if __name__ == "__main__":
    main()
