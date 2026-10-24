import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

em_tool_code = '''

    @staticmethod
    def create_electromagnetic_wave_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "LINEAR_POLARIZATION_45_DEG",
        e0x: float = 100.0,
        e0y: float = 100.0,
        delta_rad: float = 0.0,
        wavelength_nm: float = 532.0,
        intensity_w_m2: float = 26.54,
        b0_ut: float = 0.472,
        rad_pressure_upa: float = 0.089,
        polarization_name: str = "Tuyến tính (Linear 45°)",
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of transverse Electromagnetic Waves, Poynting vector, and Polarization."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#070b14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'VẬT LÝ & ĐIỆN ĐỘNG LỰC HỌC: SÓNG ĐIỆN TỪ & VECTƠ POYNTING – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 38.0
        vw = 360.0
        vh = 240.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # 3D Propagation Axis +z
        axis_y = vy0 + vh * 0.55
        z_line = QGraphicsLineItem(vx0 + 30.0, axis_y, vx0 + vw - 20.0, axis_y)
        z_line.setPen(QPen(QColor('#94a3b8'), 1.5))
        z_line.setZValue(10.3)
        items.append(z_line)

        k_lbl = QGraphicsTextItem('k⃗ (+z)')
        k_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        k_lbl.setDefaultTextColor(QColor('#f59e0b'))
        k_lbl.setPos(vx0 + vw - 35.0, axis_y - 18.0)
        k_lbl.setZValue(10.4)
        items.append(k_lbl)

        # Sinusoidal E-field path (Vertical/Transverse, Red)
        e_path = QPainterPath()
        e_path.moveTo(vx0 + 40.0, axis_y)
        num_pts = 60
        span = vw - 80.0
        for i in range(num_pts + 1):
            px = vx0 + 40.0 + (i / float(num_pts)) * span
            phase = (i / float(num_pts)) * 4.0 * math.pi
            ey = axis_y - 50.0 * math.cos(phase)
            e_path.lineTo(px, ey)
        e_item = QGraphicsPathItem(e_path)
        e_item.setPen(QPen(QColor('#ef4444'), 2.2))
        e_item.setZValue(10.4)
        items.append(e_item)

        # Sinusoidal B-field path (Orthogonal Transverse, Cyan)
        b_path = QPainterPath()
        b_path.moveTo(vx0 + 40.0, axis_y)
        for i in range(num_pts + 1):
            px = vx0 + 40.0 + (i / float(num_pts)) * span
            phase = (i / float(num_pts)) * 4.0 * math.pi
            # Tilted 3D horizontal perspective
            by = axis_y + 28.0 * math.cos(phase)
            bx = px + 15.0 * math.cos(phase)
            b_path.lineTo(bx, by)
        b_item = QGraphicsPathItem(b_path)
        b_item.setPen(QPen(QColor('#38bdf8'), 1.8, Qt.PenStyle.DashLine))
        b_item.setZValue(10.3)
        items.append(b_item)

        # Poynting Vector Arrow (Gold, along propagation direction)
        s_arr = QGraphicsLineItem(vx0 + vw * 0.45, axis_y - 25.0, vx0 + vw * 0.45 + 35.0, axis_y - 25.0)
        s_arr.setPen(QPen(QColor('#f59e0b'), 2.2))
        s_arr.setZValue(10.5)
        items.append(s_arr)

        s_lbl = QGraphicsTextItem('S⃗ = (1/μ₀)(E⃗ × B⃗)')
        s_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        s_lbl.setDefaultTextColor(QColor('#f59e0b'))
        s_lbl.setPos(vx0 + vw * 0.45 - 15.0, axis_y - 42.0)
        s_lbl.setZValue(10.6)
        items.append(s_lbl)

        # Legend
        leg_e = QGraphicsTextItem('■ E⃗: Điện trường')
        leg_e.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        leg_e.setDefaultTextColor(QColor('#ef4444'))
        leg_e.setPos(vx0 + 10.0, vy0 + 8.0)
        leg_e.setZValue(10.5)
        items.append(leg_e)

        leg_b = QGraphicsTextItem('■ B⃗: Từ trường')
        leg_b.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        leg_b.setDefaultTextColor(QColor('#38bdf8'))
        leg_b.setPos(vx0 + 100.0, vy0 + 8.0)
        leg_b.setZValue(10.5)
        items.append(leg_b)

        # Right Panel: Polarization Screen & Stokes Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        # Transverse Screen Crosshair
        sc_cx = rx0 + rw / 2.0
        sc_cy = vy0 + 60.0
        ch_h = QGraphicsLineItem(sc_cx - 40.0, sc_cy, sc_cx + 40.0, sc_cy)
        ch_h.setPen(QPen(QColor('#334155'), 1.0, Qt.PenStyle.DashLine))
        ch_h.setZValue(10.3)
        items.append(ch_h)

        ch_v = QGraphicsLineItem(sc_cx, sc_cy - 40.0, sc_cx, sc_cy + 40.0)
        ch_v.setPen(QPen(QColor('#334155'), 1.0, Qt.PenStyle.DashLine))
        ch_v.setZValue(10.3)
        items.append(ch_v)

        # Polarization Orbit Path (Circle, Line, or Ellipse)
        orb_path = QPainterPath()
        for deg in range(361):
            th = math.radians(deg)
            ox = sc_cx + 30.0 * math.cos(th + delta_rad)
            oy = sc_cy - 30.0 * math.cos(th)
            if deg == 0:
                orb_path.moveTo(ox, oy)
            else:
                orb_path.lineTo(ox, oy)
        orb_item = QGraphicsPathItem(orb_path)
        orb_item.setPen(QPen(QColor('#a855f7'), 2.0))
        orb_item.setZValue(10.4)
        items.append(orb_item)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ SÓNG ĐIỆN TỪ MAXWELL:\\n'
            f'• Bước sóng: {wavelength_nm:.1f} nm | Tần số: {C_LIGHT / (wavelength_nm * 1e-9) * 1e-12:.1f} THz\\n'
            f'• Biên độ điện trường: E₀ = {math.hypot(e0x, e0y):.1f} V/m\\n'
            f'• Cảm ứng từ B₀ = E₀/c: {b0_ut:.3f} μT\\n'
            f'• Độ lệch pha δ: {math.degrees(delta_rad):.1f}°\\n'
            f'• Trạng thái phân cực: {polarization_name}\\n'
            f'• Cường độ Poynting <S>: {intensity_w_m2:.2f} W/m²\\n'
            f'• Áp suất bức xạ P_rad: {rad_pressure_upa:.3f} μPa'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 115.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Hệ phương trình Maxwell khẳng định sóng điện từ lan truyền trong chân không với vận tốc ánh sáng c = 1/√(ε₀μ₀). '
            'Vectơ điện E⃗ và từ B⃗ luôn dao động vuông góc với nhau và vuông góc với phương truyền sóng k⃗. '
            'Vectơ Poynting S⃗ = (1/μ₀)(E⃗ × B⃗) định lượng dòng năng lượng và áp suất bức xạ truyền tải trên mỗi đơn vị diện tích.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items
'''

chem_idx = text.find('class ChemistryTools:')
if chem_idx == -1:
    print('Failed to find class ChemistryTools!')
    sys.exit(1)

# Find last return items in PhysicsTools (before ChemistryTools)
ret_idx = text[:chem_idx].rfind('        return items\n')
insert_pos = ret_idx + len('        return items\n')

new_text = text[:insert_pos] + em_tool_code + text[insert_pos:]

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Successfully added create_electromagnetic_wave_diagram to PhysicsTools!')
