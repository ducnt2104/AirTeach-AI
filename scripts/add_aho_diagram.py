import sys

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

aho_method = '''

    @staticmethod
    def create_aho_corasick_diagram(
        center: tuple[float, float],
        width: float = 660.0,
        height: float = 360.0,
        preset_code: str = 'CLASSIC_BELL_LABS_HE_SHE_HIS_HERS',
        patterns: Optional[list[str]] = None,
        sample_text: str = 'ushers',
        num_nodes: int = 10,
        num_matches: int = 3,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Aho-Corasick Multi-Pattern String Matching Automaton."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'TIN HỌC & LÝ THUYẾT OTOMAT: AHO-CORASICK MULTI-PATTERN DFA – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 340.0
        vh = 220.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Root (0)
        r_x, r_y = vx0 + 40.0, vy0 + 110.0
        root_c = QGraphicsEllipseItem(r_x - 14.0, r_y - 14.0, 28.0, 28.0)
        root_c.setBrush(QBrush(QColor('#1e293b')))
        root_c.setPen(QPen(QColor('#38bdf8'), 1.5))
        root_c.setZValue(10.4)
        items.append(root_c)

        r_lbl = QGraphicsTextItem('0')
        r_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        r_lbl.setDefaultTextColor(QColor('#f8fafc'))
        r_lbl.setPos(r_x - 6.0, r_y - 10.0)
        r_lbl.setZValue(10.5)
        items.append(r_lbl)

        # Node 1 ('h')
        n1_x, n1_y = vx0 + 120.0, vy0 + 60.0
        line_0_1 = QGraphicsLineItem(r_x, r_y, n1_x, n1_y)
        line_0_1.setPen(QPen(QColor('#38bdf8'), 1.8))
        line_0_1.setZValue(10.3)
        items.append(line_0_1)

        c1 = QGraphicsEllipseItem(n1_x - 12.0, n1_y - 12.0, 24.0, 24.0)
        c1.setBrush(QBrush(QColor('#0f172a')))
        c1.setPen(QPen(QColor('#38bdf8'), 1.5))
        c1.setZValue(10.4)
        items.append(c1)

        l1 = QGraphicsTextItem('1:h')
        l1.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l1.setDefaultTextColor(QColor('#38bdf8'))
        l1.setPos(n1_x - 9.0, n1_y - 9.0)
        l1.setZValue(10.5)
        items.append(l1)

        # Node 2 ('e') -> Terminal 'he'
        n2_x, n2_y = vx0 + 200.0, vy0 + 40.0
        line_1_2 = QGraphicsLineItem(n1_x, n1_y, n2_x, n2_y)
        line_1_2.setPen(QPen(QColor('#38bdf8'), 1.8))
        line_1_2.setZValue(10.3)
        items.append(line_1_2)

        c2 = QGraphicsEllipseItem(n2_x - 14.0, n2_y - 14.0, 28.0, 28.0)
        c2.setBrush(QBrush(QColor('#10b981')))
        c2.setPen(QPen(QColor('#ffffff'), 1.8))
        c2.setZValue(10.4)
        items.append(c2)

        l2 = QGraphicsTextItem('2:e★')
        l2.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l2.setDefaultTextColor(QColor('#020617'))
        l2.setPos(n2_x - 12.0, n2_y - 9.0)
        l2.setZValue(10.5)
        items.append(l2)

        # Node 3 ('s')
        n3_x, n3_y = vx0 + 120.0, vy0 + 160.0
        line_0_3 = QGraphicsLineItem(r_x, r_y, n3_x, n3_y)
        line_0_3.setPen(QPen(QColor('#a855f7'), 1.8))
        line_0_3.setZValue(10.3)
        items.append(line_0_3)

        c3 = QGraphicsEllipseItem(n3_x - 12.0, n3_y - 12.0, 24.0, 24.0)
        c3.setBrush(QBrush(QColor('#0f172a')))
        c3.setPen(QPen(QColor('#a855f7'), 1.5))
        c3.setZValue(10.4)
        items.append(c3)

        l3 = QGraphicsTextItem('3:s')
        l3.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l3.setDefaultTextColor(QColor('#a855f7'))
        l3.setPos(n3_x - 9.0, n3_y - 9.0)
        l3.setZValue(10.5)
        items.append(l3)

        # Failure Link: e.g. Node 6 ('sh') -> Node 1 ('h') (Dashed Amber Arc)
        f_line = QGraphicsLineItem(vx0 + 200.0, vy0 + 160.0, n1_x, n1_y + 12.0)
        pen_fail = QPen(QColor('#f59e0b'), 1.5, Qt.PenStyle.DashLine)
        f_line.setPen(pen_fail)
        f_line.setZValue(10.3)
        items.append(f_line)

        # Right Panel: Patterns & Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        p_list = patterns or ['he', 'she', 'his', 'hers']
        p_str = ', '.join([f'"{p}"' for p in p_list])

        info_txt = QGraphicsTextItem(
            f'CẤU TRÚC OTOMAT AHO-CORASICK:\\n'
            f'• Mẫu từ khóa: [{p_str}]\\n'
            f'• Văn bản kiểm tra: "{sample_text}"\\n'
            f'• Tổng số trạng thái (Trie nodes): {num_nodes}\\n'
            f'• Số khớp từ khóa phát hiện: {num_matches}\\n\\n'
            f'ĐẶC TRƯNG ĐỘ PHỨC TẠP:\\n'
            f'• Xây dựng cây tiền tố & failure links: O(m)\\n'
            f'• Quét luồng văn bản độ dài n: O(n + z)\\n'
            f'• z: Tổng số lần xuất hiện từ khóa khớp\\n'
            f'• Đảm bảo thời gian tuyến tính tuyệt đối'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Thuật toán Aho-Corasick (1975): Kết hợp cây tiền tố (Trie) với các hàm chuyển tiếp thất bại (Failure function) '
            'và lan truyền từ khóa hậu tố (Suffix outputs). Cho phép tìm kiếm đồng thời k mẫu từ khóa trên luồng văn bản '
            'chỉ trong một lần duyệt tuyến tính duy nhất, ứng dụng cốt lõi trong Snort IDS, Grep và Bio-informatics.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items
'''

target = '        return items\n\n\n# ─── STEM Subject Dialog'
replacement = '        return items' + aho_method + '\n\n# ─── STEM Subject Dialog'

new_text = text.replace(target, replacement, 1)
if new_text == text:
    print('Error: Target string not found!')
    sys.exit(1)

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Successfully inserted create_aho_corasick_diagram!')
