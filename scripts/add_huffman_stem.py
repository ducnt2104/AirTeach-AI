"""
Script to insert create_huffman_coding_diagram into InformaticsTools in src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "# ─── STEM Subject Dialog for Interactive Tool Parameter Entry ─────────────────"
    if target not in content:
        print("ERROR: Target line not found in stem_tools.py")
        sys.exit(1)

    if "create_huffman_coding_diagram" in content:
        print("create_huffman_coding_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_huffman_coding_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'CLASSIC_ABRACADABRA',
        text_sample: str = 'ABRACADABRA',
        entropy: float = 2.04,
        avg_code_len: float = 2.09,
        efficiency_pct: float = 97.6,
        saved_storage_pct: float = 73.9,
        codes: Optional[dict[str, str]] = None,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Huffman Prefix-Free Tree and Shannon Entropy analysis."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#10b981'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'TIN HỌC & LÝ THUYẾT THÔNG TIN: CÂY TIỀN TỐ HUFFMAN & ENTROPY SHANNON – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 42.0
        vw = 320.0
        vh = 200.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Draw representative Huffman Tree
        root_x = vx0 + vw / 2.0
        root_y = vy0 + 30.0

        # Root
        root_circle = QGraphicsEllipseItem(root_x - 14.0, root_y - 14.0, 28.0, 28.0)
        root_circle.setBrush(QBrush(QColor('#1e293b')))
        root_circle.setPen(QPen(QColor('#38bdf8'), 1.5))
        root_circle.setZValue(10.4)
        items.append(root_circle)

        root_lbl = QGraphicsTextItem('11')
        root_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        root_lbl.setDefaultTextColor(QColor('#f8fafc'))
        root_lbl.setPos(root_x - 7.0, root_y - 10.0)
        root_lbl.setZValue(10.5)
        items.append(root_lbl)

        # Left Child: Leaf 'A' (bit 0)
        leaf_a_x = root_x - 70.0
        leaf_a_y = root_y + 60.0
        line_l = QGraphicsLineItem(root_x, root_y, leaf_a_x, leaf_a_y)
        line_l.setPen(QPen(QColor('#38bdf8'), 2.0))
        line_l.setZValue(10.3)
        items.append(line_l)

        b0_lbl = QGraphicsTextItem('0')
        b0_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        b0_lbl.setDefaultTextColor(QColor('#38bdf8'))
        b0_lbl.setPos((root_x + leaf_a_x) / 2.0 - 15.0, (root_y + leaf_a_y) / 2.0 - 12.0)
        b0_lbl.setZValue(10.5)
        items.append(b0_lbl)

        box_a = QGraphicsRectItem(leaf_a_x - 16.0, leaf_a_y - 14.0, 32.0, 28.0)
        box_a.setBrush(QBrush(QColor('#10b981')))
        box_a.setPen(QPen(QColor('#ffffff'), 1.5))
        box_a.setZValue(10.4)
        items.append(box_a)

        lbl_a = QGraphicsTextItem("'A'\\n5")
        lbl_a.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        lbl_a.setDefaultTextColor(QColor('#020617'))
        lbl_a.setPos(leaf_a_x - 10.0, leaf_a_y - 14.0)
        lbl_a.setZValue(10.5)
        items.append(lbl_a)

        # Right Subtree: Internal Node (bit 1)
        sub_r_x = root_x + 70.0
        sub_r_y = root_y + 50.0
        line_r = QGraphicsLineItem(root_x, root_y, sub_r_x, sub_r_y)
        line_r.setPen(QPen(QColor('#f59e0b'), 2.0))
        line_r.setZValue(10.3)
        items.append(line_r)

        b1_lbl = QGraphicsTextItem('1')
        b1_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        b1_lbl.setDefaultTextColor(QColor('#f59e0b'))
        b1_lbl.setPos((root_x + sub_r_x) / 2.0 + 4.0, (root_y + sub_r_y) / 2.0 - 12.0)
        b1_lbl.setZValue(10.5)
        items.append(b1_lbl)

        node_r = QGraphicsEllipseItem(sub_r_x - 12.0, sub_r_y - 12.0, 24.0, 24.0)
        node_r.setBrush(QBrush(QColor('#1e293b')))
        node_r.setPen(QPen(QColor('#f59e0b'), 1.5))
        node_r.setZValue(10.4)
        items.append(node_r)

        # Children of Right Subtree
        # Leaf 'R'
        leaf_r_x = sub_r_x - 35.0
        leaf_r_y = sub_r_y + 55.0
        line_r_l = QGraphicsLineItem(sub_r_x, sub_r_y, leaf_r_x, leaf_r_y)
        line_r_l.setPen(QPen(QColor('#38bdf8'), 1.5))
        line_r_l.setZValue(10.3)
        items.append(line_r_l)

        box_r = QGraphicsRectItem(leaf_r_x - 14.0, leaf_r_y - 12.0, 28.0, 24.0)
        box_r.setBrush(QBrush(QColor('#10b981')))
        box_r.setPen(QPen(QColor('#ffffff'), 1.2))
        box_r.setZValue(10.4)
        items.append(box_r)

        lbl_r = QGraphicsTextItem("'R'\\n2")
        lbl_r.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        lbl_r.setDefaultTextColor(QColor('#020617'))
        lbl_r.setPos(leaf_r_x - 9.0, leaf_r_y - 13.0)
        lbl_r.setZValue(10.5)
        items.append(lbl_r)

        # Leaf 'B'
        leaf_b_x = sub_r_x + 35.0
        leaf_b_y = sub_r_y + 55.0
        line_r_r = QGraphicsLineItem(sub_r_x, sub_r_y, leaf_b_x, leaf_b_y)
        line_r_r.setPen(QPen(QColor('#f59e0b'), 1.5))
        line_r_r.setZValue(10.3)
        items.append(line_r_r)

        box_b = QGraphicsRectItem(leaf_b_x - 14.0, leaf_b_y - 12.0, 28.0, 24.0)
        box_b.setBrush(QBrush(QColor('#10b981')))
        box_b.setPen(QPen(QColor('#ffffff'), 1.2))
        box_b.setZValue(10.4)
        items.append(box_b)

        lbl_b = QGraphicsTextItem("'B'\\n2")
        lbl_b.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        lbl_b.setDefaultTextColor(QColor('#020617'))
        lbl_b.setPos(leaf_b_x - 9.0, leaf_b_y - 13.0)
        lbl_b.setZValue(10.5)
        items.append(lbl_b)

        # Right Panel: Prefix Codes & Shannon Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        sample_codes = codes or {'A': '0', 'B': '10', 'R': '11', 'C': '100', 'D': '101'}
        codes_str = '\\n'.join([f"• '{c}': {code} ({len(code)} bit)" for c, code in list(sample_codes.items())[:5]])

        info_txt = QGraphicsTextItem(
            f'BẢNG MÃ TIỀN TỐ HUFFMAN:\\n'
            f'{codes_str}\\n\\n'
            f'THÔNG SỐ LÝ THUYẾT THÔNG TIN:\\n'
            f'• Chuỗi mẫu: \"{text_sample}\"\\n'
            f'• Entropy Shannon H(X): {entropy:.3f} bits/ký tự\\n'
            f'• Độ dài mã trung bình L̄: {avg_code_len:.3f} bits/ký tự\\n'
            f'• Hiệu suất mã hóa: {efficiency_pct:.1f}% (Cận Shannon)\\n'
            f'• Tiết kiệm dung lượng: {saved_storage_pct:.1f}% vs ASCII'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Thuật toán Huffman: Xây dựng cây nhị phân tiền tố tối ưu bằng hàng đợi ưu tiên Min-Heap. '
            'Ký tự xuất hiện càng nhiều có mã càng ngắn. Định lý Mã hóa nguồn Shannon khẳng định độ dài trung bình '
            'luôn đạt cận H(X) ≤ L̄ < H(X) + 1, mang lại hiệu suất nén dữ liệu tối đa.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 12.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items

'''

    new_content = content.replace(target, method_code + "\n" + target, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully added create_huffman_coding_diagram to InformaticsTools!")

if __name__ == "__main__":
    main()
