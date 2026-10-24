import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

phase_tool_code = '''

    @staticmethod
    def create_phase_diagram_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "WATER_H2O_ANOMALOUS_MELTING",
        substance_name: str = "Nước",
        formula: str = "H₂O",
        temp_c: float = 25.0,
        press_atm: float = 1.0,
        current_phase: str = "LỎNG (LIQUID)",
        t_triple: float = 0.01,
        p_triple: float = 0.006,
        t_crit: float = 373.95,
        p_crit: float = 217.75,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of thermodynamic P-T Phase Diagram and Clausius-Clapeyron curves."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'HÓA LÝ & NHIỆT ĐỘNG HỌC: GIẢN ĐỒ PHA P-T CỦA {substance_name.upper()} ({formula}) – {preset_code}')
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

        # Coordinate axes & 1 atm reference line
        y_1atm = vy0 + vh * 0.48
        line_1atm = QGraphicsLineItem(vx0 + 40.0, y_1atm, vx0 + vw - 10.0, y_1atm)
        line_1atm.setPen(QPen(QColor('#f59e0b'), 1.2, Qt.PenStyle.DotLine))
        line_1atm.setZValue(10.3)
        items.append(line_1atm)

        lbl_1atm = QGraphicsTextItem('1 atm')
        lbl_1atm.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        lbl_1atm.setDefaultTextColor(QColor('#f59e0b'))
        lbl_1atm.setPos(vx0 + 8.0, y_1atm - 10.0)
        lbl_1atm.setZValue(10.4)
        items.append(lbl_1atm)

        # Triple Point
        tx = vx0 + 110.0
        ty = vy0 + vh * 0.72
        trip_dot = QGraphicsEllipseItem(tx - 4.0, ty - 4.0, 8.0, 8.0)
        trip_dot.setBrush(QBrush(QColor('#f59e0b')))
        trip_dot.setPen(QPen(QColor('#ffffff'), 1.2))
        trip_dot.setZValue(10.5)
        items.append(trip_dot)

        trip_lbl = QGraphicsTextItem(f'T ({t_triple:g}°C, {p_triple:g} atm)')
        trip_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        trip_lbl.setDefaultTextColor(QColor('#f59e0b'))
        trip_lbl.setPos(tx + 4.0, ty + 2.0)
        trip_lbl.setZValue(10.6)
        items.append(trip_lbl)

        # Critical Point
        cx_pt = vx0 + vw - 40.0
        cy_pt = vy0 + 35.0
        crit_dot = QGraphicsEllipseItem(cx_pt - 5.0, cy_pt - 5.0, 10.0, 10.0)
        crit_dot.setBrush(QBrush(QColor('#ef4444')))
        crit_dot.setPen(QPen(QColor('#ffffff'), 1.5))
        crit_dot.setZValue(10.5)
        items.append(crit_dot)

        crit_lbl = QGraphicsTextItem(f'C ({t_crit:g}°C, {p_crit:g} atm)')
        crit_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        crit_lbl.setDefaultTextColor(QColor('#ef4444'))
        crit_lbl.setPos(cx_pt - 65.0, cy_pt - 14.0)
        crit_lbl.setZValue(10.6)
        items.append(crit_lbl)

        # Sublimation Curve (Solid - Gas)
        sub_path = QPainterPath()
        sub_path.moveTo(vx0 + 40.0, vy0 + vh - 20.0)
        sub_path.quadTo(vx0 + 70.0, vy0 + vh * 0.85, tx, ty)
        sub_item = QGraphicsPathItem(sub_path)
        sub_item.setPen(QPen(QColor('#f59e0b'), 2.0))
        sub_item.setZValue(10.4)
        items.append(sub_item)

        # Vaporization Curve (Liquid - Gas) Clausius-Clapeyron
        vap_path = QPainterPath()
        vap_path.moveTo(tx, ty)
        vap_path.quadTo((tx + cx_pt) / 2.0 + 15.0, (ty + cy_pt) / 2.0 + 25.0, cx_pt, cy_pt)
        vap_item = QGraphicsPathItem(vap_path)
        vap_item.setPen(QPen(QColor('#38bdf8'), 2.2))
        vap_item.setZValue(10.4)
        items.append(vap_item)

        # Melting Curve (Solid - Liquid)
        melt_path = QPainterPath()
        melt_path.moveTo(tx, ty)
        if 'WATER' in preset_code or 'H2O' in formula:
            # Anomalous negative melting slope (tilted left)
            melt_path.lineTo(tx - 25.0, vy0 + 25.0)
        else:
            # Positive melting slope (tilted right)
            melt_path.lineTo(tx + 30.0, vy0 + 25.0)
        melt_item = QGraphicsPathItem(melt_path)
        melt_item.setPen(QPen(QColor('#10b981'), 2.2))
        melt_item.setZValue(10.4)
        items.append(melt_item)

        # Labels for phases
        s_lbl = QGraphicsTextItem('RẮN')
        s_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        s_lbl.setDefaultTextColor(QColor('#93c5fd'))
        s_lbl.setPos(vx0 + 55.0, vy0 + 60.0)
        s_lbl.setZValue(10.5)
        items.append(s_lbl)

        l_lbl = QGraphicsTextItem('LỎNG')
        l_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        l_lbl.setDefaultTextColor(QColor('#2dd4bf'))
        l_lbl.setPos(tx + 40.0, vy0 + 60.0)
        l_lbl.setZValue(10.5)
        items.append(l_lbl)

        g_lbl = QGraphicsTextItem('KHÍ / HƠI')
        g_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        g_lbl.setDefaultTextColor(QColor('#fdba74'))
        g_lbl.setPos(vx0 + vw * 0.65, vy0 + vh * 0.70)
        g_lbl.setZValue(10.5)
        items.append(g_lbl)

        # Right Panel: Metrics & Theory
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ CÂN BẰNG PHA ({formula}):\\n'
            f'• Nhiệt độ T: {temp_c:.2f} °C ({temp_c + 273.15:.2f} K)\\n'
            f'• Áp suất P: {press_atm:g} atm ({press_atm * 101.325:.1f} kPa)\\n'
            f'• Trạng thái pha hiện tại: {current_phase}\\n\\n'
            f'ĐẶC TRƯNG NHIỆT ĐỘNG HỌC:\\n'
            f'• Điểm tam giác T: ({t_triple:g}°C, {p_triple:g} atm)\\n'
            f'• Điểm tới hạn C: ({t_crit:g}°C, {p_crit:g} atm)\\n'
            f'• Phương trình Clausius-Clapeyron:\\n'
            f'   dP/dT = ΔH / (T·ΔV)\\n'
            f'• Nước: dP/dT < 0 (Băng nhẹ hơn nước lỏng)\\n'
            f'• CO₂: P_triple > 1 atm (Băng khô thăng hoa)'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Giản đồ pha (Phase Diagram) biểu diễn các vùng cân bằng nhiệt động Solid-Liquid-Gas. '
            'Đường hóa hơi kết thúc tại Điểm tới hạn (Critical Point), mở ra trạng thái Chất lưu siêu tới hạn. '
            'Độ dốc nóng chảy âm của nước là hiện tượng bất thường duy nhất cho phép trượt băng dưới áp suất lớn.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items
'''

target = '        items.append(info)\n\n        return items\n\n\n'
replacement = '        items.append(info)\n\n        return items\n' + phase_tool_code + '\n\n'

if target not in text:
    print('Failed to find target in stem_tools.py!')
    sys.exit(1)

new_text = text.replace(target, replacement, 1)
with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Successfully added create_phase_diagram_diagram to ChemistryTools!')
