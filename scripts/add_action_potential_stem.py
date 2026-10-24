"""
Script to insert create_action_potential_diagram into BiologyTools in src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "class InformaticsTools:"
    if target not in content:
        print("ERROR: Target line not found in stem_tools.py")
        sys.exit(1)

    if "def create_action_potential_diagram" in content:
        print("create_action_potential_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_action_potential_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'STANDARD_ACTION_POTENTIAL',
        v_peak_mv: float = 32.4,
        i_inj_ua: float = 10.0,
        toxin: str = 'none',
        temp_c: float = 6.3,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Hodgkin-Huxley Action Potential."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#f43f5e'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'SINH HỌC & ĐIỆN SINH LÝ: ĐIỆN THẾ HOẠT ĐỘNG HODGKIN-HUXLEY – {preset_code}')
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

        # Baseline & thresholds
        base_y = vy0 + vh - 55.0
        grid_rest = QGraphicsLineItem(vx0 + 10.0, base_y, vx0 + vw - 10.0, base_y)
        grid_rest.setPen(QPen(QColor('#38bdf8'), 1.0, Qt.PenStyle.DashLine))
        grid_rest.setZValue(10.3)
        items.append(grid_rest)

        # Action potential spike path
        spike_path = QPainterPath()
        spike_path.moveTo(vx0 + 20.0, base_y)
        spike_path.lineTo(vx0 + 60.0, base_y)
        # Depolarization up to peak (+35 mV)
        spike_path.lineTo(vx0 + 100.0, vy0 + 35.0)
        # Repolarization
        spike_path.lineTo(vx0 + 140.0, base_y + 20.0)  # Hyperpolarization undershoot
        # Recovery to rest
        spike_path.quadTo(vx0 + 180.0, base_y + 5.0, vx0 + 220.0, base_y)
        spike_path.lineTo(vx0 + vw - 20.0, base_y)

        spike_item = QGraphicsPathItem(spike_path)
        spike_item.setPen(QPen(QColor('#f43f5e'), 2.4))
        spike_item.setZValue(10.4)
        items.append(spike_item)

        # Peak badge
        peak_lbl = QGraphicsTextItem(f'Đỉnh: {v_peak_mv:+.1f} mV')
        peak_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        peak_lbl.setDefaultTextColor(QColor('#f43f5e'))
        peak_lbl.setPos(vx0 + 85.0, vy0 + 18.0)
        peak_lbl.setZValue(10.5)
        items.append(peak_lbl)

        # Right box: Equations & Bio-parameters
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        info = QGraphicsTextItem(
            f'THÔNG SỐ ĐIỆN SINH LÝ HỌC:\\n'
            f'• Dòng kích thích: {i_inj_ua:.1f} μA/cm²\\n'
            f'• Điện thế đỉnh: {v_peak_mv:+.1f} mV | Điện thế nghỉ: -65.0 mV\\n'
            f'• Kênh Na⁺: g_Na = ḡ_Na·m³h (E_Na = +50.0 mV)\\n'
            f'• Kênh K⁺: g_K = ḡ_K·n⁴ (E_K = -77.0 mV)\\n'
            f'• Độc tố ức chế: {toxin.upper()}\\n'
            f'• Nhiệt độ mô phỏng: {temp_c:.1f} °C'
        )
        info.setFont(QFont('Segoe UI', 7))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(rx0 + 10.0, vy0 + 12.0)
        info.setZValue(10.4)
        items.append(info)

        cap = QGraphicsTextItem(
            'Mô hình Hodgkin-Huxley (Giải Nobel Y sinh 1963): Điện thế hoạt động phát sinh từ sự biến thiên độ dẫn '
            'của các kênh ion Na⁺ và K⁺ phụ thuộc điện thế trên màng sợi trục thần kinh mực ống khổng lồ, '
            'tuân thủ quy luật Tất cả hoặc Không (All-or-None Law).'
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
    print("Successfully added create_action_potential_diagram to BiologyTools!")

if __name__ == "__main__":
    main()
