"""
Insert create_wave_interference_diagram into PhysicsTools in src/ui/stem_tools.py
"""

def main():
    with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if 'def create_wave_interference_diagram(' in content:
        print("create_wave_interference_diagram already exists!")
        return

    method_code = '''    @staticmethod
    def create_wave_interference_diagram(
        center: tuple[float, float],
        width: float = 480.0,
        height: float = 300.0,
        wavelength_nm: float = 632.8,
        slit_a_mm: float = 0.50,
        distance_D_m: float = 1.50,
        fringe_spacing_mm: Optional[float] = None,
        preset: str = "RED_HE_NE_LASER",
        probe_x_mm: float = 0.0,
    ) -> list[QGraphicsItem]:
        """Creates an authentic Wave Optics & Young's Double-Slit vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        i_val = fringe_spacing_mm if fringe_spacing_mm is not None else ((wavelength_nm * distance_D_m * 1e-3) / max(0.01, slit_a_mm))

        # 1. Title Banner
        banner_txt = f"GIAO THOA SÓNG ÁNH SÁNG KHE YOUNG (λ = {wavelength_nm:.1f}nm, a = {slit_a_mm:.2f}mm, i = {i_val:.3f}mm)"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        # 2. Viewport Dark Cosmic Pane
        pane_x = x_left + 10.0
        pane_y = y_top + 10.0
        pane_w = 280.0
        pane_h = 240.0

        bg_pane = QGraphicsRectItem(pane_x, pane_y, pane_w, pane_h)
        bg_pane.setBrush(QBrush(QColor("#030712")))
        bg_pane.setPen(QPen(QColor("#1e293b"), 1.2))
        bg_pane.setZValue(10.0)
        items.append(bg_pane)

        # 3. Optical Bench Elements (Laser, Slits, Screen)
        laser_box = QGraphicsRectItem(pane_x + 12.0, pane_y + 35.0, 36.0, 24.0)
        laser_box.setBrush(QBrush(QColor("#1e293b")))
        laser_box.setPen(QPen(QColor("#475569"), 1.0))
        laser_box.setZValue(10.2)
        items.append(laser_box)

        lbl_laser = QGraphicsTextItem("LASER")
        lbl_laser.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_laser.setDefaultTextColor(QColor("#ef4444"))
        lbl_laser.setPos(pane_x + 13.0, pane_y + 37.0)
        lbl_laser.setZValue(10.3)
        items.append(lbl_laser)

        # Double slit barrier
        barrier_x = pane_x + 95.0
        bar1 = QGraphicsLineItem(barrier_x, pane_y + 15.0, barrier_x, pane_y + 42.0)
        bar1.setPen(QPen(QColor("#64748b"), 2.0))
        bar1.setZValue(10.2)
        items.append(bar1)

        bar2 = QGraphicsLineItem(barrier_x, pane_y + 52.0, barrier_x, pane_y + 80.0)
        bar2.setPen(QPen(QColor("#64748b"), 2.0))
        bar2.setZValue(10.2)
        items.append(bar2)

        lbl_slits = QGraphicsTextItem("S₁, S₂")
        lbl_slits.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        lbl_slits.setDefaultTextColor(QColor("#94a3b8"))
        lbl_slits.setPos(barrier_x - 18.0, pane_y + 40.0)
        lbl_slits.setZValue(10.3)
        items.append(lbl_slits)

        # Screen line
        screen_x = pane_x + 260.0
        screen_line = QGraphicsLineItem(screen_x, pane_y + 15.0, screen_x, pane_y + 80.0)
        screen_line.setPen(QPen(QColor("#cbd5e1"), 2.5))
        screen_line.setZValue(10.2)
        items.append(screen_line)

        # Light rays from slits to screen center
        ray1 = QGraphicsLineItem(barrier_x, pane_y + 45.0, screen_x, pane_y + 47.0)
        ray1.setPen(QPen(QColor(239, 68, 68, 140), 1.0, Qt.PenStyle.DashLine))
        ray1.setZValue(10.2)
        items.append(ray1)

        ray2 = QGraphicsLineItem(barrier_x, pane_y + 49.0, screen_x, pane_y + 47.0)
        ray2.setPen(QPen(QColor(239, 68, 68, 140), 1.0, Qt.PenStyle.DashLine))
        ray2.setZValue(10.2)
        items.append(ray2)

        # 4. Photorealistic Optical Screen Strip (Bottom of viewport)
        strip_x = pane_x + 12.0
        strip_y = pane_y + 190.0
        strip_w = pane_w - 24.0
        strip_h = 32.0

        screen_bg = QGraphicsRectItem(strip_x, strip_y, strip_w, strip_h)
        screen_bg.setBrush(QBrush(QColor("#020617")))
        screen_bg.setPen(QPen(QColor("#475569"), 1.0))
        screen_bg.setZValue(10.2)
        items.append(screen_bg)

        # Render simulated fringe bars
        center_strip_x = strip_x + strip_w / 2.0
        fringe_px = 16.0  # visual spacing on stamped card
        for k in range(-7, 8):
            fx = center_strip_x + k * fringe_px
            if strip_x + 4.0 <= fx <= strip_x + strip_w - 4.0:
                # Bright fringe bar
                bar_alpha = int(255 * (1.0 / (1.0 + 0.15 * abs(k))))
                f_bar = QGraphicsRectItem(fx - 2.0, strip_y + 2.0, 4.0, strip_h - 4.0)
                f_bar.setBrush(QBrush(QColor(239, 68, 68, bar_alpha)))
                f_bar.setPen(Qt.PenStyle.NoPen)
                f_bar.setZValue(10.4)
                items.append(f_bar)

        # Central maximum indicator
        lbl_center = QGraphicsTextItem("k = 0 (Trung tâm)")
        lbl_center.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_center.setDefaultTextColor(QColor("#fbbf24"))
        lbl_center.setPos(center_strip_x - 36.0, strip_y + strip_h + 2.0)
        lbl_center.setZValue(10.5)
        items.append(lbl_center)

        # 5. Theoretical Intensity Curve I(x)
        graph_top = pane_y + 98.0
        graph_base = pane_y + 175.0
        axis_i = QGraphicsLineItem(strip_x, graph_base, strip_x + strip_w, graph_base)
        axis_i.setPen(QPen(QColor("#475569"), 1.0, Qt.PenStyle.DotLine))
        axis_i.setZValue(10.3)
        items.append(axis_i)

        curve_path = QPainterPath()
        first_c = True
        for step in range(101):
            px = strip_x + (step / 100.0) * strip_w
            dx_rel = (px - center_strip_x) / fringe_px
            # cos²(pi * x / i)
            intens = (math.cos(math.pi * dx_rel)) ** 2 * (math.sin(0.4 * dx_rel + 1e-5) / (0.4 * dx_rel + 1e-5)) ** 2
            py = graph_base - intens * (graph_base - graph_top)
            if first_c:
                curve_path.moveTo(px, py)
                first_c = False
            else:
                curve_path.lineTo(px, py)

        curve_item = QGraphicsPathItem(curve_path)
        curve_item.setPen(QPen(QColor("#ef4444"), 1.6))
        curve_item.setZValue(10.5)
        items.append(curve_item)

        # 6. Right-side Theory & Metrics Card
        card_x = pane_x + pane_w + 12.0
        card_y = pane_y
        card_w = width - pane_w - 34.0
        card_h = pane_h

        info_bg = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_bg.setBrush(QBrush(QColor(30, 41, 59, 230)))
        info_bg.setPen(QPen(QColor("#334155"), 1.0))
        info_bg.setZValue(10.0)
        items.append(info_bg)

        lines = [
            ("LÝ THUYẾT GIAO THOA SÓNG ÁNH SÁNG", QColor("#f8fafc"), True, 9),
            (f"• Bước sóng: λ = {wavelength_nm:.1f} nm", QColor("#ef4444"), False, 8),
            (f"• Khoảng cách 2 khe: a = {slit_a_mm:.2f} mm", QColor("#38bdf8"), False, 8),
            (f"• Khoảng cách đến màn: D = {distance_D_m:.2f} m", QColor("#10b981"), False, 8),
            (f"• Khoảng vân: i = (λ·D)/a = {i_val:.3f} mm", QColor("#fbbf24"), True, 8),
            ("• Vị trí vân sáng: x_k = k·i (k ∈ ℤ)", QColor("#cbd5e1"), False, 8),
            ("• Vị trí vân tối: x_k' = (k + 0.5)·i", QColor("#cbd5e1"), False, 8),
            ("• Hiệu đường đi: Δd = a·x / D = k·λ", QColor("#94a3b8"), False, 8),
            ("• Hiệu số pha: Δφ = 2π·a·x / (λ·D)", QColor("#94a3b8"), False, 8),
        ]

        text_y = card_y + 8.0
        for txt, color, is_bold, sz in lines:
            t_item = QGraphicsTextItem(txt)
            t_item.setFont(QFont("Segoe UI", sz, QFont.Weight.Bold if is_bold else QFont.Weight.Normal))
            t_item.setDefaultTextColor(color)
            t_item.setPos(card_x + 8.0, text_y)
            t_item.setZValue(10.5)
            items.append(t_item)
            text_y += 24.0

        return items
'''
    target = 'class ChemistryTools:'
    assert target in content, "target not found"
    new_content = content.replace(target, method_code + '\n\n' + target, 1)

    with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully inserted create_wave_interference_diagram into stem_tools.py!")

if __name__ == '__main__':
    main()
