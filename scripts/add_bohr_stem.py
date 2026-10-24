"""
Script to insert create_bohr_atom_diagram into PhysicsTools in src/ui/stem_tools.py.
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

    if "create_bohr_atom_diagram" in content:
        print("create_bohr_atom_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_bohr_atom_diagram(
        center: tuple[float, float],
        width: float = 520.0,
        height: float = 320.0,
        level_m: int = 3,
        level_n: int = 2,
        preset: str = "BALMER_ALPHA",
    ) -> list[QGraphicsItem]:
        """Creates an authentic Quantum Physics Bohr Atom & Hydrogen spectral line vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        rydberg_ev = 13.605693
        hc_ev_nm = 1239.841984
        bohr_r0 = 0.529177  # Å
        v1_kms = 2187.69

        m_val = max(1, min(6, int(level_m)))
        n_val = max(1, min(6, int(level_n)))

        e_m = -rydberg_ev / (m_val ** 2)
        e_n = -rydberg_ev / (n_val ** 2)
        delta_e = abs(e_m - e_n)
        wl_nm = (hc_ev_nm / delta_e) if delta_e > 0.001 else 9999.0

        lower_n = min(m_val, n_val)
        if lower_n == 1:
            series_name = "Dãy Lyman (Tử ngoại - UV)"
            col_photon = QColor("#a855f7")
        elif lower_n == 2:
            series_name = "Dãy Balmer (Khả kiến & Gần UV)"
            if abs(wl_nm - 656.3) < 15.0:
                col_photon = QColor("#ef4444")
            elif abs(wl_nm - 486.1) < 15.0:
                col_photon = QColor("#06b6d4")
            elif abs(wl_nm - 434.1) < 15.0:
                col_photon = QColor("#6366f1")
            else:
                col_photon = QColor("#8b5cf6")
        elif lower_n == 3:
            series_name = "Dãy Paschen (Hồng ngoại gần - IR)"
            col_photon = QColor("#dc2626")
        else:
            series_name = f"Dãy Brackett (n = {lower_n})"
            col_photon = QColor("#b91c1c")

        orbit_letters = {1: "K", 2: "L", 3: "M", 4: "N", 5: "O", 6: "P"}

        # 1. Title Banner
        banner_txt = f"VẬT LÝ LƯỢNG TỬ: NGUYÊN TỬ BOHR [{orbit_letters.get(m_val, str(m_val))} → {orbit_letters.get(n_val, str(n_val))}, n={m_val}→{n_val}] - {series_name} (λ = {wl_nm:.1f}nm)"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        # 2. Viewport Dark Cosmic Pane (Left: Concentric Orbits)
        pane_x = x_left + 10.0
        pane_y = y_top + 10.0
        pane_w = 260.0
        pane_h = 290.0

        bg_pane = QGraphicsRectItem(pane_x, pane_y, pane_w, pane_h)
        bg_pane.setBrush(QBrush(QColor("#030712")))
        bg_pane.setPen(QPen(QColor("#1e293b"), 1.2))
        bg_pane.setZValue(10.0)
        items.append(bg_pane)

        atom_cx = pane_x + pane_w / 2.0
        atom_cy = pane_y + pane_h / 2.0 - 15.0

        # Draw concentric orbits
        for i in range(1, 7):
            r_px = 24.0 + i * 16.0
            is_active = (i == m_val or i == n_val)
            orb_pen = QPen(QColor("#38bdf8") if i == m_val else (QColor("#f59e0b") if i == n_val else QColor("#334155")), 1.4 if is_active else 0.8, Qt.PenStyle.SolidLine if is_active else Qt.PenStyle.DotLine)
            orb_circle = QGraphicsEllipseItem(atom_cx - r_px, atom_cy - r_px, r_px * 2.0, r_px * 2.0)
            orb_circle.setBrush(Qt.BrushStyle.NoBrush)
            orb_circle.setPen(orb_pen)
            orb_circle.setZValue(10.2)
            items.append(orb_circle)

            t_let = QGraphicsTextItem(orbit_letters.get(i, str(i)))
            t_let.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            t_let.setDefaultTextColor(QColor("#cbd5e1") if is_active else QColor("#64748b"))
            t_let.setPos(atom_cx + r_px - 8.0, atom_cy - 12.0)
            t_let.setZValue(10.3)
            items.append(t_let)

        # Proton Nucleus (+e)
        nuc = QGraphicsEllipseItem(atom_cx - 7.0, atom_cy - 7.0, 14.0, 14.0)
        nuc.setBrush(QBrush(QColor("#ef4444")))
        nuc.setPen(QPen(QColor("#ffffff"), 1.0))
        nuc.setZValue(10.5)
        items.append(nuc)

        t_plus = QGraphicsTextItem("+")
        t_plus.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        t_plus.setDefaultTextColor(QColor("#ffffff"))
        t_plus.setPos(atom_cx - 5.0, atom_cy - 10.0)
        t_plus.setZValue(10.6)
        items.append(t_plus)

        # Electron on level m
        r_m = 24.0 + m_val * 16.0
        ex = atom_cx + r_m * 0.707
        ey = atom_cy - r_m * 0.707
        elec = QGraphicsEllipseItem(ex - 4.0, ey - 4.0, 8.0, 8.0)
        elec.setBrush(QBrush(QColor("#00f5ff")))
        elec.setPen(QPen(QColor("#ffffff"), 1.0))
        elec.setZValue(10.7)
        items.append(elec)

        # Transition wavy arrow to orbit n
        r_n = 24.0 + n_val * 16.0
        tx = atom_cx + r_n * 0.707
        ty = atom_cy - r_n * 0.707

        arr_line = QGraphicsLineItem(ex, ey, tx, ty)
        arr_line.setPen(QPen(col_photon, 2.0, Qt.PenStyle.DashLine))
        arr_line.setZValue(10.6)
        items.append(arr_line)

        # Radiating photon label
        photon_txt = QGraphicsTextItem(f"Photon γ (λ={wl_nm:.1f}nm)")
        photon_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        photon_txt.setDefaultTextColor(col_photon)
        photon_txt.setPos(pane_x + 10.0, pane_y + pane_h - 26.0)
        photon_txt.setZValue(10.5)
        items.append(photon_txt)

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

        r_ang = (m_val ** 2) * bohr_r0
        v_kms = v1_kms / float(m_val)

        lines = [
            ("MÔ HÌNH NGUYÊN TỬ BOHR & HYDRO", QColor("#f8fafc"), True, 9),
            (f"• Quỹ đạo đầu: {orbit_letters.get(m_val, '')} (n={m_val}, E_{m_val} = {e_m:.2f} eV)", QColor("#38bdf8"), False, 8),
            (f"• Quỹ đạo cuối: {orbit_letters.get(n_val, '')} (n={n_val}, E_{n_val} = {e_n:.2f} eV)", QColor("#f59e0b"), False, 8),
            (f"• Năng lượng photon phát xạ: ε = {delta_e:.3f} eV", QColor("#22c55e"), True, 8),
            (f"• Bước sóng photon: λ = {wl_nm:.1f} nm", col_photon, True, 8),
            (f"• Dãy quang phổ: {series_name}", QColor("#cbd5e1"), False, 8),
            (f"• Bán kính quỹ đạo r_{m_val} = {m_val}²·r₀: {r_ang:.2f} Å (r₀ = 0.53 Å)", QColor("#94a3b8"), False, 8),
            (f"• Tốc độ electron v_{m_val} = v₁/{m_val}: {v_kms:.0f} km/s (v₁ = 2188 km/s)", QColor("#94a3b8"), False, 8),
            ("• Hệ thức Bohr: h·f = |E_m - E_n| = h·c / λ", QColor("#94a3b8"), False, 8),
            ("• Công thức Rydberg: 1/λ = R_H (1/n² - 1/m²)", QColor("#94a3b8"), False, 8),
            ("• 4 vạch nhìn thấy Dãy Balmer: H_α (656nm), H_β (486nm), H_γ, H_δ", QColor("#cbd5e1"), False, 8),
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
    print("Successfully added create_bohr_atom_diagram to stem_tools.py")

if __name__ == "__main__":
    main()
