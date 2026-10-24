"""
Script to insert create_gcms_diagram into ChemistryTools in src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "# ─── Vector Tools for Biology ─────────────────────────────────────────────────"
    if target not in content:
        print("ERROR: Target line not found in stem_tools.py")
        sys.exit(1)

    if "def create_gcms_diagram" in content:
        print("create_gcms_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_gcms_diagram(
        center: tuple[float, float],
        preset_code: str = "PETROCHEMICAL_GASOLINE_BTEX",
        selected_peak_idx: int = 0,
        ramp_rate: float = 10.0,
        flow_rate: float = 1.0,
        width: float = 640.0,
        height: float = 340.0,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Gas Chromatography - Mass Spectrometry (GC-MS)."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#06b6d4'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'HÓA PHÂN TÍCH: SẮC KÝ KHÍ - KHỐI PHỔ GC-MS – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 300.0
        vh = 195.0

        # 1. TIC Chromatogram Box
        tic_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        tic_box.setBrush(QBrush(QColor('#020617')))
        tic_box.setPen(QPen(QColor('#1e293b'), 1.0))
        tic_box.setZValue(10.2)
        items.append(tic_box)

        tic_hdr = QGraphicsTextItem('SẮC KÝ ĐỒ TỔNG DÒNG ION (TIC CHROMATOGRAM)')
        tic_hdr.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        tic_hdr.setDefaultTextColor(QColor('#38bdf8'))
        tic_hdr.setPos(vx0 + 10.0, vy0 + 8.0)
        tic_hdr.setZValue(10.4)
        items.append(tic_hdr)

        # Baseline
        base_y = vy0 + vh - 25.0
        base_line = QGraphicsLineItem(vx0 + 25.0, base_y, vx0 + vw - 15.0, base_y)
        base_line.setPen(QPen(QColor('#475569'), 1.0))
        base_line.setZValue(10.3)
        items.append(base_line)

        # Draw representative chromatographic Gaussian peaks
        peaks_x = [vx0 + 60.0, vx0 + 110.0, vx0 + 165.0, vx0 + 215.0, vx0 + 265.0]
        peaks_h = [45.0, 95.0, 70.0, 55.0, 35.0]
        peak_names = ["Benzen", "Toluen", "Etylbenzen", "m/p-Xylen", "o-Xylen"]

        for i, (px, ph) in enumerate(zip(peaks_x, peaks_h)):
            peak_path = QPainterPath()
            peak_path.moveTo(px - 18.0, base_y)
            peak_path.quadTo(px, base_y - ph * 1.2, px + 18.0, base_y)

            p_item = QGraphicsPathItem(peak_path)
            is_sel = (i == selected_peak_idx)
            p_item.setPen(QPen(QColor('#f59e0b') if is_sel else QColor('#38bdf8'), 1.8 if is_sel else 1.2))
            p_item.setBrush(QBrush(QColor(245, 158, 11, 40) if is_sel else QColor(56, 189, 248, 25)))
            p_item.setZValue(10.4)
            items.append(p_item)

            lbl = QGraphicsTextItem(f"t_R{i+1}")
            lbl.setFont(QFont('Segoe UI', 6))
            lbl.setDefaultTextColor(QColor('#f59e0b') if is_sel else QColor('#94a3b8'))
            lbl.setPos(px - 10.0, base_y + 2.0)
            lbl.setZValue(10.5)
            items.append(lbl)

        # 2. Right Box: Mass Spectrum & Readout
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        ms_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        ms_box.setBrush(QBrush(QColor('#020617')))
        ms_box.setPen(QPen(QColor('#1e293b'), 1.0))
        ms_box.setZValue(10.2)
        items.append(ms_box)

        ms_hdr = QGraphicsTextItem('KHỐI PHỔ ION HÓA ĐIỆN TỬ (EI-MS 70 eV)')
        ms_hdr.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        ms_hdr.setDefaultTextColor(QColor('#10b981'))
        ms_hdr.setPos(rx0 + 10.0, vy0 + 8.0)
        ms_hdr.setZValue(10.4)
        items.append(ms_hdr)

        # Sticks
        ms_base_y = vy0 + 95.0
        ms_baseline = QGraphicsLineItem(rx0 + 15.0, ms_base_y, rx0 + rw - 15.0, ms_base_y)
        ms_baseline.setPen(QPen(QColor('#475569'), 1.0))
        ms_baseline.setZValue(10.3)
        items.append(ms_baseline)

        # Draw standard stick peaks (e.g. Toluen: m/z 91 base peak tropylium ion, 92 M+, 65, 39)
        sticks = [(rx0 + 40.0, 25.0, "39"), (rx0 + 80.0, 40.0, "65"), (rx0 + 140.0, 75.0, "91 (100%)"), (rx0 + 160.0, 50.0, "92 (M⁺)")]
        for sx, sh, slbl in sticks:
            stk = QGraphicsLineItem(sx, ms_base_y, sx, ms_base_y - sh)
            stk.setPen(QPen(QColor('#10b981'), 2.2))
            stk.setZValue(10.4)
            items.append(stk)

            stk_lbl = QGraphicsTextItem(slbl)
            stk_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
            stk_lbl.setDefaultTextColor(QColor('#10b981'))
            stk_lbl.setPos(sx - 12.0, ms_base_y - sh - 14.0)
            stk_lbl.setZValue(10.5)
            items.append(stk_lbl)

        # Readout text
        sel_name = peak_names[selected_peak_idx] if selected_peak_idx < len(peak_names) else "Hợp chất"
        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ PHÂN TÍCH GC-MS:\\n'
            f'• Hợp chất đang chọn: {sel_name} (Peak #{selected_peak_idx+1})\\n'
            f'• Tốc độ gia nhiệt: {ramp_rate:.1f} °C/phút | Lưu lượng khí He: {flow_rate:.2f} mL/phút\\n'
            f'• Ion phân tử M⁺•: m/z 92 | Ion gốc cơ sở (Base peak): m/z 91 (Tropylium C₇H₇⁺)\\n'
            f'• Độ phân giải sắc ký: R_s ≥ 1.5 (Tách rời hoàn toàn về đường nền)'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, ms_base_y + 12.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Kỹ thuật GC-MS: Sắc ký khí tách các cấu tử bay hơi theo độ phân cực và điểm sôi trên cột mao quản DB-5MS. '
            'Khối phổ ion hóa điện tử 70 eV bắn phá phân tử thành ion phân tử M⁺• và các mảnh ion đặc trưng, '
            'cho phép nhận danh chính xác cấu trúc hóa học với độ tin cậy tuyệt đối.'
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
    print("Successfully added create_gcms_diagram to ChemistryTools!")

if __name__ == "__main__":
    main()
