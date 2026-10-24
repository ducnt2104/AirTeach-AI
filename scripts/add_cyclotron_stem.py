"""
Script to insert create_cyclotron_diagram into PhysicsTools in src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "class ChemistryTools:"
    if target not in content:
        print("ERROR: Target line not found in stem_tools.py")
        sys.exit(1)

    if "def create_cyclotron_diagram" in content:
        print("create_cyclotron_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_cyclotron_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'CLASSICAL_PROTON_ACCELERATOR',
        particle_name: str = 'Proton (¹H⁺)',
        b_field_tesla: float = 1.5,
        dee_voltage_kv: float = 50.0,
        freq_mhz: float = 22.88,
        max_energy_mev: float = 16.5,
        r_max_m: float = 0.8,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Cyclotron Particle Accelerator."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'VẬT LÝ HẠT NHÂN: MÁY GIA TỐC CYCLOTRON (LAWRENCE 1939) – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 310.0
        vh = 195.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        bx = vx0 + vw / 2.0
        by = vy0 + vh / 2.0
        r_dee = 75.0

        # Left Dee (D1)
        dee_l = QGraphicsEllipseItem(bx - r_dee - 4.0, by - r_dee, 2 * r_dee, 2 * r_dee)
        dee_l.setSpanAngle(180 * 16)
        dee_l.setStartAngle(90 * 16)
        dee_l.setBrush(QBrush(QColor(15, 23, 42)))
        dee_l.setPen(QPen(QColor('#0284c7'), 2.0))
        dee_l.setZValue(10.3)
        items.append(dee_l)

        # Right Dee (D2)
        dee_r = QGraphicsEllipseItem(bx - r_dee + 4.0, by - r_dee, 2 * r_dee, 2 * r_dee)
        dee_r.setSpanAngle(180 * 16)
        dee_r.setStartAngle(-90 * 16)
        dee_r.setBrush(QBrush(QColor(15, 23, 42)))
        dee_r.setPen(QPen(QColor('#0284c7'), 2.0))
        dee_r.setZValue(10.3)
        items.append(dee_r)

        # Spiral path
        spiral_path = QPainterPath()
        spiral_path.moveTo(bx, by)
        for i in range(1, 45):
            rad = i * 1.5
            angle = i * 0.4
            sx = bx + rad * math.cos(angle)
            sy = by + rad * math.sin(angle)
            spiral_path.lineTo(sx, sy)

        sp_item = QGraphicsPathItem(spiral_path)
        sp_item.setPen(QPen(QColor('#ef4444'), 1.8))
        sp_item.setZValue(10.4)
        items.append(sp_item)

        # Right box: Readouts & Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        info = QGraphicsTextItem(
            f'THÔNG SỐ GIA TỐC CYCLOTRON:\\n'
            f'• Hạt tích điện: {particle_name}\\n'
            f'• Cảm ứng từ B: {b_field_tesla:.2f} T (vuông góc mặt phẳng)\\n'
            f'• Điện thế cực Dees: {dee_voltage_kv:.1f} kV\\n'
            f'• Tần số cộng hưởng f_c: {freq_mhz:.2f} MHz\\n'
            f'• Bán kính buồng R_max: {r_max_m:.2f} m\\n'
            f'• Động năng cực đại tại bia: {max_energy_mev:.2f} MeV'
        )
        info.setFont(QFont('Segoe UI', 7))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(rx0 + 10.0, vy0 + 12.0)
        info.setZValue(10.4)
        items.append(info)

        cap = QGraphicsTextItem(
            'Máy gia tốc Cyclotron: Sử dụng từ trường tĩnh uốn hạt theo quỹ đạo xoắn ốc và điện trường xoay chiều '
            'tần số cao gia tốc hạt mỗi khi qua khe hở giữa 2 cực Dees. Ở vận tốc lớn (v > 0.1c), hiệu ứng tương đối tính '
            'làm tăng khối lượng m = γ·m₀, đòi hỏi biến điệu tần số RF (Synchrocyclotron).'
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
    print("Successfully added create_cyclotron_diagram to PhysicsTools!")

if __name__ == "__main__":
    main()
