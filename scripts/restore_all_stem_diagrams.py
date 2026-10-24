"""
Enhanced script to restore and guarantee all interactive STEM diagram methods in src/ui/stem_tools.py
with full rich graphical components (>= 7 items each).
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Re-insert enriched versions of the methods:

    # 1. Physics methods
    phys_methods = '''    @staticmethod
    def create_photoelectric_effect_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'CESIUM_VISIBLE_EMISSION',
        metal_name: str = 'Xesi (Cs)',
        work_function_ev: float = 2.14,
        wavelength_nm: float = 500.0,
        stopping_voltage_v: float = 0.34,
        is_emitted: bool = True,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Photoelectric Effect phototube."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'HIỆN TƯỢNG QUANG ĐIỆN NGOÀI EINSTEIN – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        # Vacuum Phototube
        tube = QGraphicsRectItem(x0 + 30.0, y0 + 50.0, 240.0, 130.0)
        tube.setBrush(QBrush(QColor(15, 23, 42, 100)))
        tube.setPen(QPen(QColor('#38bdf8'), 1.5))
        tube.setZValue(10.2)
        items.append(tube)

        # Cathode
        cat = QGraphicsRectItem(x0 + 55.0, y0 + 75.0, 12.0, 80.0)
        cat.setBrush(QBrush(QColor('#f59e0b')))
        cat.setPen(QPen(QColor('#ffffff'), 1.0))
        cat.setZValue(10.4)
        items.append(cat)

        # Anode
        an = QGraphicsRectItem(x0 + 230.0, y0 + 75.0, 12.0, 80.0)
        an.setBrush(QBrush(QColor('#64748b')))
        an.setPen(QPen(QColor('#ffffff'), 1.0))
        an.setZValue(10.4)
        items.append(an)

        # Photon Beam
        beam = QGraphicsLineItem(x0 + 15.0, y0 + 115.0, x0 + 55.0, y0 + 115.0)
        beam.setPen(QPen(QColor('#22c55e'), 2.5))
        beam.setZValue(10.3)
        items.append(beam)

        info = QGraphicsTextItem(
            f'THÔNG SỐ QUANG ĐIỆN:\\n'
            f'• Kim loại: {metal_name} | Công thoát: {work_function_ev:.2f} eV\\n'
            f'• Bước sóng tới: {wavelength_nm:.1f} nm | Điện thế hãm: {stopping_voltage_v:.2f} V\\n'
            f'• Trạng thái: {"Có quang điện (λ ≤ λ₀)" if is_emitted else "Triệt tiêu (λ > λ₀)"}'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 300.0, y0 + 50.0)
        info.setZValue(10.4)
        items.append(info)

        return items

    @staticmethod
    def create_hohmann_transfer_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'LEO_TO_GEO_TRANSFER',
        r1_km: float = 6678.0,
        r2_km: float = 42164.0,
        delta_v1_kms: float = 2.425,
        delta_v2_kms: float = 1.465,
        transfer_hours: float = 5.26,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Hohmann Orbital Transfer."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'CƠ HỌC THIÊN THỂ: CHUYỂN QUỸ ĐẠO HOHMANN – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        ox = x0 + 150.0
        oy = y0 + 160.0

        # Central Body (Earth)
        body = QGraphicsEllipseItem(ox - 18.0, oy - 18.0, 36.0, 36.0)
        body.setBrush(QBrush(QColor('#0284c7')))
        body.setPen(QPen(QColor('#38bdf8'), 1.5))
        body.setZValue(10.3)
        items.append(body)

        # Orbit 1 (LEO)
        o1 = QGraphicsEllipseItem(ox - 45.0, oy - 45.0, 90.0, 90.0)
        o1.setPen(QPen(QColor('#38bdf8'), 1.2, Qt.PenStyle.DashLine))
        o1.setBrush(Qt.BrushStyle.NoBrush)
        o1.setZValue(10.2)
        items.append(o1)

        # Orbit 2 (GEO)
        o2 = QGraphicsEllipseItem(ox - 110.0, oy - 110.0, 220.0, 220.0)
        o2.setPen(QPen(QColor('#f59e0b'), 1.2, Qt.PenStyle.DashLine))
        o2.setBrush(Qt.BrushStyle.NoBrush)
        o2.setZValue(10.2)
        items.append(o2)

        # Transfer Ellipse
        trans = QGraphicsEllipseItem(ox - 110.0, oy - 77.5, 155.0, 155.0)
        trans.setPen(QPen(QColor('#22c55e'), 1.8))
        trans.setBrush(Qt.BrushStyle.NoBrush)
        trans.setZValue(10.3)
        items.append(trans)

        info = QGraphicsTextItem(
            f'THÔNG SỐ QUỸ ĐẠO:\\n'
            f'• Bán kính r₁: {r1_km:,.0f} km | Bán kính r₂: {r2_km:,.0f} km\\n'
            f'• Δv₁: {delta_v1_kms:.3f} km/s | Δv₂: {delta_v2_kms:.3f} km/s\\n'
            f'• Tổng Δv: {delta_v1_kms + delta_v2_kms:.3f} km/s\\n'
            f'• Thời gian bay: {transfer_hours:.2f} giờ'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 300.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items

    @staticmethod
    def create_quantum_well_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'GROUND_STATE_N1',
        n_quantum: int = 1,
        well_width_nm: float = 1.0,
        energy_ev: float = 0.376,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of 1D Quantum Well and Schrödinger Wavefunction."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'CƠ HỌC LƯỢNG TỬ: GIẾNG THẾ VÔ HẠN 1D (SCHRÖDINGER) – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        wx0 = x0 + 40.0
        wy0 = y0 + 60.0
        ww = 220.0
        wh = 140.0

        # Left Barrier (V = infinity)
        b_l = QGraphicsLineItem(wx0, wy0, wx0, wy0 + wh)
        b_l.setPen(QPen(QColor('#38bdf8'), 3.0))
        b_l.setZValue(10.3)
        items.append(b_l)

        # Right Barrier
        b_r = QGraphicsLineItem(wx0 + ww, wy0, wx0 + ww, wy0 + wh)
        b_r.setPen(QPen(QColor('#38bdf8'), 3.0))
        b_r.setZValue(10.3)
        items.append(b_r)

        # Well Floor
        floor = QGraphicsLineItem(wx0, wy0 + wh, wx0 + ww, wy0 + wh)
        floor.setPen(QPen(QColor('#475569'), 1.5))
        floor.setZValue(10.2)
        items.append(floor)

        # Standing Wavefunction psi_n(x)
        wave_path = QPainterPath()
        wave_y = wy0 + wh - 40.0
        wave_path.moveTo(wx0, wave_y)
        for i in range(50):
            frac = i / 49.0
            px = wx0 + frac * ww
            py = wave_y - 30.0 * math.sin(n_quantum * math.pi * frac)
            wave_path.lineTo(px, py)
        wave_item = QGraphicsPathItem(wave_path)
        wave_item.setPen(QPen(QColor('#a855f7'), 2.0))
        wave_item.setZValue(10.4)
        items.append(wave_item)

        info = QGraphicsTextItem(
            f'THÔNG SỐ TRẠNG THÁI DỪNG:\\n'
            f'• Mức lượng tử n = {n_quantum} | Bề rộng giếng L = {well_width_nm:.2f} nm\\n'
            f'• Năng lượng lượng tử hóa E_n: {energy_ev:.4f} eV\\n'
            f'• Hàm sóng: ψ_n(x) = √(2/L) · sin(nπx/L)'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 300.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items
'''

    # 2. Chemistry methods
    chem_methods = '''    @staticmethod
    def create_reaction_kinetics_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'EXOTHERMIC_CATALYSIS_DECOMPOSITION',
        reaction_str: str = '2 H₂O₂ ➔ 2 H₂O + O₂',
        ea_uncat_kj: float = 75.0,
        ea_cat_kj: float = 28.0,
        delta_h_kj: float = -98.0,
        temp_k: float = 298.15,
        has_catalyst: bool = True,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Chemical Reaction Kinetics and Energy Coordinate."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#f59e0b'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'ĐỘNG HÓA HỌC & ĐỊNH LUẬT ARRHENIUS: {reaction_str} – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        px0 = x0 + 30.0
        py0 = y0 + 60.0
        pw = 230.0
        ph = 140.0

        # Energy profile box
        coord_box = QGraphicsRectItem(px0, py0, pw, ph)
        coord_box.setBrush(QBrush(QColor('#020617')))
        coord_box.setPen(QPen(QColor('#1e293b'), 1.0))
        coord_box.setZValue(10.2)
        items.append(coord_box)

        # Baseline
        base_line = QGraphicsLineItem(px0 + 10.0, py0 + ph - 20.0, px0 + pw - 10.0, py0 + ph - 20.0)
        base_line.setPen(QPen(QColor('#475569'), 1.0))
        base_line.setZValue(10.3)
        items.append(base_line)

        # Uncatalyzed curve
        uncat_path = QPainterPath()
        uncat_path.moveTo(px0 + 20.0, py0 + ph - 50.0)
        uncat_path.quadTo(px0 + pw * 0.5, py0 + 20.0, px0 + pw - 20.0, py0 + ph - 30.0)
        uncat_item = QGraphicsPathItem(uncat_path)
        uncat_item.setPen(QPen(QColor('#ef4444'), 2.0))
        uncat_item.setZValue(10.4)
        items.append(uncat_item)

        # Catalyzed curve
        cat_path = QPainterPath()
        cat_path.moveTo(px0 + 20.0, py0 + ph - 50.0)
        cat_path.quadTo(px0 + pw * 0.5, py0 + 65.0, px0 + pw - 20.0, py0 + ph - 30.0)
        cat_item = QGraphicsPathItem(cat_path)
        cat_item.setPen(QPen(QColor('#22c55e'), 2.0, Qt.PenStyle.DashLine))
        cat_item.setZValue(10.4)
        items.append(cat_item)

        info = QGraphicsTextItem(
            f'THÔNG SỐ ĐỘNG HỌC:\\n'
            f'• Phản ứng: {reaction_str}\\n'
            f'• E_a không xúc tác: {ea_uncat_kj:.1f} kJ/mol | E_a có xúc tác: {ea_cat_kj:.1f} kJ/mol\\n'
            f'• Biến thiên Entanpi ΔH: {delta_h_kj:.1f} kJ/mol ({"Tỏa nhiệt" if delta_h_kj < 0 else "Thu nhiệt"})\\n'
            f'• Nhiệt độ: {temp_k:.1f} K'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 290.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items
'''

    # 3. Biology methods
    bio_methods = '''    @staticmethod
    def create_punnett_genetics_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'MONOHYBRID_PEA_PLANT',
        parent_cross: str = 'Aa × Aa',
        phenotype_ratio: str = '3 Vàng : 1 Xanh',
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Mendelian Genetics & Punnett Square."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#22c55e'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'DI TRUYỀN HỌC MENDEL & KHUNG PUNNETT: {parent_cross} – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        # 2x2 Punnett Grid Cells
        gx0 = x0 + 40.0
        gy0 = y0 + 60.0
        cell_s = 60.0

        for r in range(2):
            for c in range(2):
                cell = QGraphicsRectItem(gx0 + c * cell_s, gy0 + r * cell_s, cell_s, cell_s)
                cell.setBrush(QBrush(QColor('#020617')))
                cell.setPen(QPen(QColor('#22c55e'), 1.2))
                cell.setZValue(10.3)
                items.append(cell)

        info = QGraphicsTextItem(
            f'THÔNG SỐ PHÉP LAI:\\n'
            f'• Phép lai P: {parent_cross}\\n'
            f'• Tỷ lệ phân ly kiểu hình F₁: {phenotype_ratio}\\n'
            f'• Quy luật: Phân ly độc lập & Tương tác alen'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 200.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items
'''

    # Replace earlier stubs:
    # Remove existing definitions if present
    for func in ["create_photoelectric_effect_diagram", "create_hohmann_transfer_diagram", "create_quantum_well_diagram"]:
        while f"    @staticmethod\n    def {func}" in content:
            start_idx = content.find(f"    @staticmethod\n    def {func}")
            end_idx = content.find("        return items\n\n", start_idx)
            if end_idx != -1:
                content = content[:start_idx] + content[end_idx + len("        return items\n\n"):]
            else:
                break

    for func in ["create_reaction_kinetics_diagram"]:
        while f"    @staticmethod\n    def {func}" in content:
            start_idx = content.find(f"    @staticmethod\n    def {func}")
            end_idx = content.find("        return items\n\n", start_idx)
            if end_idx != -1:
                content = content[:start_idx] + content[end_idx + len("        return items\n\n"):]
            else:
                break

    for func in ["create_punnett_genetics_diagram"]:
        while f"    @staticmethod\n    def {func}" in content:
            start_idx = content.find(f"    @staticmethod\n    def {func}")
            end_idx = content.find("        return items\n\n", start_idx)
            if end_idx != -1:
                content = content[:start_idx] + content[end_idx + len("        return items\n\n"):]
            else:
                break

    # Now cleanly insert rich versions
    content = content.replace("class ChemistryTools:", phys_methods + "\nclass ChemistryTools:", 1)
    content = content.replace("class BiologyTools:", chem_methods + "\nclass BiologyTools:", 1)
    content = content.replace("class InformaticsTools:", bio_methods + "\nclass InformaticsTools:", 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully enriched all STEM diagram methods!")

if __name__ == "__main__":
    main()
