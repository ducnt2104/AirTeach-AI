"""
Script to insert create_michelson_interferometer_diagram into PhysicsTools in src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "# ─── Vector Tools for Chemistry ───────────────────────────────────────────────"
    if target not in content:
        print("ERROR: Target line not found!")
        sys.exit(1)

    if "create_michelson_interferometer_diagram" in content:
        print("create_michelson_interferometer_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_michelson_interferometer_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'HE_NE_CIRCULAR',
        mode: str = 'circular',
        source_name: str = 'Laser He-Ne Đỏ (632.8 nm)',
        wavelength_nm: float = 632.8,
        delta_d_um: float = 10.0,
        visibility_pct: float = 100.0,
        fringe_count: float = 31.6,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Michelson Interferometer & LIGO Gravitational Wave Detector."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'QUANG HỌC & THIÊN VĂN: GIAO THOA KẾ MICHELSON & SÓNG HẤP DẪN LIGO – {preset_code}')
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

        bx_mid = vx0 + vw / 2.0
        by_mid = vy0 + vh / 2.0

        # Beam Splitter (45 deg)
        bs_line = QGraphicsLineItem(bx_mid - 15.0, by_mid - 15.0, bx_mid + 15.0, by_mid + 15.0)
        bs_line.setPen(QPen(QColor('#38bdf8'), 2.5))
        bs_line.setZValue(10.4)
        items.append(bs_line)

        # Mirror M1 (Top)
        m1 = QGraphicsLineItem(bx_mid - 25.0, vy0 + 20.0, bx_mid + 25.0, vy0 + 20.0)
        m1.setPen(QPen(QColor('#f8fafc'), 3.0))
        m1.setZValue(10.4)
        items.append(m1)

        m1_lbl = QGraphicsTextItem('Gương M₁ (Cố định)')
        m1_lbl.setFont(QFont('Segoe UI', 6))
        m1_lbl.setDefaultTextColor(QColor('#94a3b8'))
        m1_lbl.setPos(bx_mid - 35.0, vy0 + 6.0)
        m1_lbl.setZValue(10.5)
        items.append(m1_lbl)

        # Mirror M2 (Right)
        m2_x = vx0 + vw - 25.0
        m2 = QGraphicsLineItem(m2_x, by_mid - 25.0, m2_x, by_mid + 25.0)
        m2.setPen(QPen(QColor('#f59e0b'), 3.0))
        m2.setZValue(10.4)
        items.append(m2)

        m2_lbl = QGraphicsTextItem('Gương M₂\\n(Di động)')
        m2_lbl.setFont(QFont('Segoe UI', 6))
        m2_lbl.setDefaultTextColor(QColor('#f59e0b'))
        m2_lbl.setPos(m2_x - 35.0, by_mid + 28.0)
        m2_lbl.setZValue(10.5)
        items.append(m2_lbl)

        # Light Source (Left)
        src = QGraphicsRectItem(vx0 + 15.0, by_mid - 12.0, 30.0, 24.0)
        src.setBrush(QBrush(QColor('#ef4444')))
        src.setPen(QPen(QColor('#ffffff'), 1.0))
        src.setZValue(10.4)
        items.append(src)

        # Laser Ray Paths
        ray_pen = QPen(QColor('#ef4444'), 1.8)
        # Source to BS
        r1 = QGraphicsLineItem(vx0 + 45.0, by_mid, bx_mid, by_mid)
        r1.setPen(ray_pen)
        r1.setZValue(10.3)
        items.append(r1)

        # BS to M1
        r2 = QGraphicsLineItem(bx_mid, by_mid, bx_mid, vy0 + 20.0)
        r2.setPen(ray_pen)
        r2.setZValue(10.3)
        items.append(r2)

        # BS to M2
        r3 = QGraphicsLineItem(bx_mid, by_mid, m2_x, by_mid)
        r3.setPen(ray_pen)
        r3.setZValue(10.3)
        items.append(r3)

        # BS to Detector
        r4 = QGraphicsLineItem(bx_mid, by_mid, bx_mid, vy0 + vh - 20.0)
        r4.setPen(QPen(QColor('#ef4444'), 2.2))
        r4.setZValue(10.3)
        items.append(r4)

        # Detector (Bottom)
        det = QGraphicsRectItem(bx_mid - 25.0, vy0 + vh - 18.0, 50.0, 10.0)
        det.setBrush(QBrush(QColor('#22c55e')))
        det.setPen(QPen(QColor('#ffffff'), 1.0))
        det.setZValue(10.4)
        items.append(det)

        # Right Panel: Interference pattern and metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        # Concentric rings simulation
        ring_cx = rx0 + 60.0
        ring_cy = vy0 + 60.0
        for rad in [10.0, 20.0, 30.0, 40.0, 50.0]:
            ring = QGraphicsEllipseItem(ring_cx - rad, ring_cy - rad, rad * 2.0, rad * 2.0)
            ring.setPen(QPen(QColor('#ef4444'), 1.5))
            ring.setBrush(Qt.BrushStyle.NoBrush)
            ring.setZValue(10.3)
            items.append(ring)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ GIAO THOA:\\n'
            f'• Nguồn: {source_name}\\n'
            f'• Bước sóng λ: {wavelength_nm:.1f} nm\\n'
            f'• Hiệu quang trình: {2.0 * delta_d_um:+.2f} μm\\n'
            f'• Độ lệch vị trí M₂: {delta_d_um:+.2f} μm\\n'
            f'• Số vân dịch chuyển: {fringe_count:+.1f} vân\\n'
            f'• Độ tương phản V: {visibility_pct:.1f}%\\n'
            f'• Chế độ: {mode.upper()}'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 120.0, vy0 + 15.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Giao thoa kế Michelson: Tách biên độ sóng ánh sáng thành hai chùm tia trực giao. Hiệu quang trình Δ = 2(d₂ - d₁) '
            'quy định hệ vân giao thoa (vân tròn đồng tâm Haidinger hoặc vân nêm Fizeau). Đài quan sát LIGO ứng dụng nguyên lý này '
            'với cánh tay 4 km để bắt sóng hấp dẫn với độ nhạy biến dạng cỡ h ~ 10⁻²¹.'
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
    print("Successfully added create_michelson_interferometer_diagram to PhysicsTools!")

if __name__ == "__main__":
    main()
