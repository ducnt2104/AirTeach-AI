import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

hw_tool_code = '''

    @staticmethod
    def create_hardy_weinberg_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CLASSIC_IDEAL_EQUILIBRIUM",
        generation: int = 0,
        p: float = 0.7,
        q: float = 0.3,
        f_aa: float = 0.49,
        f_het: float = 0.42,
        f_rec: float = 0.09,
        pop_size: int = 0,
        chi_square: float = 0.0,
        is_hwe: bool = True,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Hardy-Weinberg Population Genetics Equilibrium."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#070b14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'SINH HỌC & DI TRUYỀN HỌC QUẦN THỂ: CÂN BẰNG HARDY-WEINBERG – {preset_code}')
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

        # Plot axes
        px0 = vx0 + 40.0
        py0 = vy0 + vh - 30.0
        pw = vw - 60.0
        ph = vh - 55.0

        ax_line = QGraphicsLineItem(px0, py0, px0 + pw, py0)
        ax_line.setPen(QPen(QColor('#64748b'), 1.2))
        ax_line.setZValue(10.3)
        items.append(ax_line)

        ay_line = QGraphicsLineItem(px0, py0, px0, py0 - ph)
        ay_line.setPen(QPen(QColor('#64748b'), 1.2))
        ay_line.setZValue(10.3)
        items.append(ay_line)

        # Theoretical curves p², 2pq, q²
        p_aa = QPainterPath()
        p_het = QPainterPath()
        p_rec = QPainterPath()

        steps = 40
        for i in range(steps + 1):
            cur_p = i / float(steps)
            cur_q = 1.0 - cur_p
            cur_x = px0 + cur_p * pw

            y_aa = py0 - (cur_p**2) * ph
            y_het = py0 - (2.0 * cur_p * cur_q) * ph
            y_rec = py0 - (cur_q**2) * ph

            if i == 0:
                p_aa.moveTo(cur_x, y_aa)
                p_het.moveTo(cur_x, y_het)
                p_rec.moveTo(cur_x, y_rec)
            else:
                p_aa.lineTo(cur_x, y_aa)
                p_het.lineTo(cur_x, y_het)
                p_rec.lineTo(cur_x, y_rec)

        # p² (AA) - Red
        item_aa = QGraphicsPathItem(p_aa)
        item_aa.setPen(QPen(QColor('#ef4444'), 2.2))
        item_aa.setZValue(10.4)
        items.append(item_aa)

        # 2pq (Aa) - Purple
        item_het = QGraphicsPathItem(p_het)
        item_het.setPen(QPen(QColor('#a855f7'), 2.2))
        item_het.setZValue(10.4)
        items.append(item_het)

        # q² (aa) - Cyan
        item_rec = QGraphicsPathItem(p_rec)
        item_rec.setPen(QPen(QColor('#06b6d4'), 2.2))
        item_rec.setZValue(10.4)
        items.append(item_rec)

        # Current p indicator line
        curr_x = px0 + p * pw
        p_line = QGraphicsLineItem(curr_x, py0, curr_x, py0 - ph)
        p_line.setPen(QPen(QColor('#f59e0b'), 1.5, Qt.PenStyle.DashLine))
        p_line.setZValue(10.5)
        items.append(p_line)

        # Genotype dots at current p
        dot_aa = QGraphicsEllipseItem(curr_x - 3.5, py0 - f_aa * ph - 3.5, 7.0, 7.0)
        dot_aa.setBrush(QBrush(QColor('#ef4444')))
        dot_aa.setPen(QPen(QColor('#ffffff'), 1.2))
        dot_aa.setZValue(10.6)
        items.append(dot_aa)

        dot_het = QGraphicsEllipseItem(curr_x - 3.5, py0 - f_het * ph - 3.5, 7.0, 7.0)
        dot_het.setBrush(QBrush(QColor('#a855f7')))
        dot_het.setPen(QPen(QColor('#ffffff'), 1.2))
        dot_het.setZValue(10.6)
        items.append(dot_het)

        dot_rec = QGraphicsEllipseItem(curr_x - 3.5, py0 - f_rec * ph - 3.5, 7.0, 7.0)
        dot_rec.setBrush(QBrush(QColor('#06b6d4')))
        dot_rec.setPen(QPen(QColor('#ffffff'), 1.2))
        dot_rec.setZValue(10.6)
        items.append(dot_rec)

        # Legend
        l_aa = QGraphicsTextItem('■ p² (AA)')
        l_aa.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l_aa.setDefaultTextColor(QColor('#ef4444'))
        l_aa.setPos(vx0 + 40.0, vy0 + 8.0)
        l_aa.setZValue(10.5)
        items.append(l_aa)

        l_het = QGraphicsTextItem('■ 2pq (Aa)')
        l_het.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l_het.setDefaultTextColor(QColor('#a855f7'))
        l_het.setPos(vx0 + 115.0, vy0 + 8.0)
        l_het.setZValue(10.5)
        items.append(l_het)

        l_rec = QGraphicsTextItem('■ q² (aa)')
        l_rec.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l_rec.setDefaultTextColor(QColor('#06b6d4'))
        l_rec.setPos(vx0 + 195.0, vy0 + 8.0)
        l_rec.setZValue(10.5)
        items.append(l_rec)

        # Right Panel: Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        pop_str = "Vô hạn (N → ∞)" if pop_size == 0 else f"{pop_size}"
        hwe_status = "ĐẠT CÂN BẰNG HWE (p > 0.05)" if is_hwe else "LỆCH CÂN BẰNG HWE (p < 0.05)"

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ DI TRUYỀN QUẦN THỂ:\\n'
            f'• Thế hệ: {generation} | Kích thước N: {pop_str}\\n'
            f'• Tần số alen A (p): {p:.4f}\\n'
            f'• Tần số alen a (q): {q:.4f} (p + q = 1.00)\\n\\n'
            f'TẦN SỐ KIỂU GEN (p² + 2pq + q² = 1.0):\\n'
            f'• f(AA) = p² = {f_aa:.4f} ({f_aa*100:.1f}%)\\n'
            f'• f(Aa) = 2pq = {f_het:.4f} ({f_het*100:.1f}%)\\n'
            f'• f(aa) = q² = {f_rec:.4f} ({f_rec*100:.1f}%)\\n\\n'
            f'KIỂM ĐỊNH CHI-BÌNH PHƯƠNG (df=1):\\n'
            f'• χ² = {chi_square:.3f} (Ngưỡng 3.841, α=0.05)\\n'
            f'• Trạng thái: {hwe_status}'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Định luật Hardy-Weinberg (1908): Trong quần thể ngẫu phối vô hạn, không chọn lọc tự nhiên, đột biến hay di nhập gen, '
            'tần số tương đối của các alen và kiểu gen duy trì không đổi qua các thế hệ. '
            'Kiểm định Chi-bình phương so sánh tần số thực nghiệm với kỳ vọng để phát hiện các nhân tố tiến hóa tác động.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items
'''

info_idx = text.find('class InformaticsTools:')
if info_idx == -1:
    print('Failed to find class InformaticsTools!')
    sys.exit(1)

# Find last return items before InformaticsTools
ret_idx = text[:info_idx].rfind('        return items\n')
insert_pos = ret_idx + len('        return items\n')

new_text = text[:insert_pos] + hw_tool_code + text[insert_pos:]

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Successfully added create_hardy_weinberg_diagram to BiologyTools!')
