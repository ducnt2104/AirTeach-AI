"""
Script to insert create_carnot_engine_diagram into PhysicsTools in src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "class ChemistryTools:"
    if target not in content:
        print("ERROR: Target class ChemistryTools not found!")
        sys.exit(1)

    if "create_carnot_engine_diagram" in content:
        print("create_carnot_engine_diagram already present!")
        return

    method_code = '''    @staticmethod
    def create_carnot_engine_diagram(
        center: tuple[float, float],
        width: float = 520.0,
        height: float = 320.0,
        th_k: float = 600.0,
        tc_k: float = 300.0,
        ratio_r: float = 2.0,
        preset: str = "STANDARD_STEAM",
    ) -> list[QGraphicsItem]:
        """Creates an authentic Thermodynamics Carnot Heat Engine & PV cycle vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        gas_r = 8.314462618
        moles_n = 1.0
        gamma = 1.40

        # Calculations
        eta = max(0.0, min(1.0, 1.0 - (tc_k / max(1.0, th_k))))
        ds = moles_n * gas_r * math.log(max(1.01, ratio_r))
        qh = th_k * ds
        qc_mag = tc_k * ds
        wnet = qh - qc_mag
        cop_ref = tc_k / max(0.01, th_k - tc_k)
        cop_hp = cop_ref + 1.0

        # State points: V in L, P in atm
        v1 = 1.0
        p1 = (moles_n * gas_r * th_k) / (v1 * 1e-3) / 101325.0
        v2 = ratio_r * v1
        p2 = (moles_n * gas_r * th_k) / (v2 * 1e-3) / 101325.0

        power = 1.0 / (gamma - 1.0)
        v3 = v2 * math.pow(th_k / max(1.0, tc_k), power)
        p3 = (moles_n * gas_r * tc_k) / (v3 * 1e-3) / 101325.0

        v4 = v3 / ratio_r
        p4 = (moles_n * gas_r * tc_k) / (v4 * 1e-3) / 101325.0

        # 1. Title Banner
        banner_txt = f"NHIỆT ĐỘNG LỰC HỌC: CHU TRÌNH CARNOT [T_H={th_k:.0f}K, T_C={tc_k:.0f}K, r={ratio_r:.1f}x] - HIỆU SUẤT η = {eta * 100.0:.1f}%"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        # 2. Viewport Dark Cosmic Pane (Left: P-V Diagram)
        pane_x = x_left + 10.0
        pane_y = y_top + 10.0
        pane_w = 260.0
        pane_h = 290.0

        bg_pane = QGraphicsRectItem(pane_x, pane_y, pane_w, pane_h)
        bg_pane.setBrush(QBrush(QColor("#030712")))
        bg_pane.setPen(QPen(QColor("#1e293b"), 1.2))
        bg_pane.setZValue(10.0)
        items.append(bg_pane)

        # Coordinate axes
        plot_x = pane_x + 36.0
        plot_y = pane_y + 24.0
        plot_w = pane_w - 50.0
        plot_h = pane_h - 70.0
        bottom_y = plot_y + plot_h

        axis_pen = QPen(QColor("#475569"), 1.0)
        x_axis = QGraphicsLineItem(plot_x, bottom_y, plot_x + plot_w, bottom_y)
        x_axis.setPen(axis_pen)
        x_axis.setZValue(10.2)
        items.append(x_axis)

        y_axis = QGraphicsLineItem(plot_x, plot_y, plot_x, bottom_y)
        y_axis.setPen(axis_pen)
        y_axis.setZValue(10.2)
        items.append(y_axis)

        # Axis labels
        x_lbl = QGraphicsTextItem("V (L)")
        x_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        x_lbl.setDefaultTextColor(QColor("#94a3b8"))
        x_lbl.setPos(plot_x + plot_w - 24.0, bottom_y + 4.0)
        x_lbl.setZValue(10.3)
        items.append(x_lbl)

        y_lbl = QGraphicsTextItem("P (atm)")
        y_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        y_lbl.setDefaultTextColor(QColor("#94a3b8"))
        y_lbl.setPos(plot_x - 32.0, plot_y - 12.0)
        y_lbl.setZValue(10.3)
        items.append(y_lbl)

        # Map function (V, P) -> (x, y)
        def to_pt(v_val: float, p_val: float) -> QPointF:
            px = plot_x + ((v_val - v1) / max(0.1, v3 - v1)) * (plot_w - 15.0)
            py = bottom_y - (p_val / max(1.0, p1)) * (plot_h - 15.0)
            return QPointF(px, py)

        # Build closed path for cycle
        cycle_path = QPainterPath()
        cycle_path.moveTo(to_pt(v1, p1))
        # 1 -> 2: Isothermal
        for s in range(1, 20):
            vs = v1 + (s / 19.0) * (v2 - v1)
            ps = (moles_n * gas_r * th_k) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_pt(vs, ps))
        # 2 -> 3: Adiabatic
        for s in range(1, 20):
            vs = v2 + (s / 19.0) * (v3 - v2)
            ts = th_k * math.pow(v2 / vs, gamma - 1.0)
            ps = (moles_n * gas_r * ts) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_pt(vs, ps))
        # 3 -> 4: Isothermal
        for s in range(1, 20):
            vs = v3 - (s / 19.0) * (v3 - v4)
            ps = (moles_n * gas_r * tc_k) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_pt(vs, ps))
        # 4 -> 1: Adiabatic
        for s in range(1, 20):
            vs = v4 - (s / 19.0) * (v4 - v1)
            ts = tc_k * math.pow(v4 / vs, gamma - 1.0)
            ps = (moles_n * gas_r * ts) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_pt(vs, ps))
        cycle_path.closeSubpath()

        cycle_item = QGraphicsPathItem(cycle_path)
        cycle_item.setBrush(QBrush(QColor(234, 88, 12, 110)))
        cycle_item.setPen(QPen(QColor("#ea580c"), 2.0))
        cycle_item.setZValue(10.4)
        items.append(cycle_item)

        # Point labels 1, 2, 3, 4
        for num, pt_v, pt_p in [("1", v1, p1), ("2", v2, p2), ("3", v3, p3), ("4", v4, p4)]:
            pos_pt = to_pt(pt_v, pt_p)
            dot = QGraphicsEllipseItem(pos_pt.x() - 3.0, pos_pt.y() - 3.0, 6.0, 6.0)
            dot.setBrush(QBrush(QColor("#ffffff")))
            dot.setPen(QPen(QColor("#ea580c"), 1.0))
            dot.setZValue(10.6)
            items.append(dot)

            t_num = QGraphicsTextItem(num)
            t_num.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            t_num.setDefaultTextColor(QColor("#f8fafc"))
            t_num.setPos(pos_pt.x() + 4.0, pos_pt.y() - 10.0)
            t_num.setZValue(10.7)
            items.append(t_num)

        # Net work badge below plot
        work_txt = QGraphicsTextItem(f"W_net = ∮ P dV = {wnet / 1000.0:.2f} kJ (Diện tích chu trình)")
        work_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        work_txt.setDefaultTextColor(QColor("#fb923c"))
        work_txt.setPos(pane_x + 10.0, pane_y + pane_h - 28.0)
        work_txt.setZValue(10.5)
        items.append(work_txt)

        # 3. Right Side: Theory & Metrics Card
        card_x = x_left + 280.0
        card_y = y_top + 10.0
        card_w = width - 290.0
        card_h = 290.0

        info_bg = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_bg.setBrush(QBrush(QColor("#0f172a")))
        info_bg.setPen(QPen(QColor("#334155"), 1.0))
        info_bg.setZValue(10.0)
        items.append(info_bg)

        lines = [
            ("NHIỆT ĐỘNG LỰC HỌC & ĐỘNG CƠ CARNOT", QColor("#f8fafc"), True, 9),
            (f"• Nhiệt độ nguồn nóng: T_H = {th_k:.1f} K ({th_k - 273.15:+.1f}°C)", QColor("#f97316"), False, 8),
            (f"• Nhiệt độ nguồn lạnh: T_C = {tc_k:.1f} K ({tc_k - 273.15:+.1f}°C)", QColor("#38bdf8"), False, 8),
            (f"• Tỷ số nén/giãn: r = V₂/V₁ = {ratio_r:.2f}", QColor("#cbd5e1"), False, 8),
            (f"• Hiệu suất cực đại: η = 1 - T_C/T_H = {eta * 100.0:.1f}%", QColor("#22c55e"), True, 8),
            (f"• Nhiệt nhận từ nguồn nóng: Q_H = {qh / 1000.0:.2f} kJ", QColor("#f97316"), False, 8),
            (f"• Nhiệt nhả cho nguồn lạnh: |Q_C| = {qc_mag / 1000.0:.2f} kJ", QColor("#38bdf8"), False, 8),
            (f"• Công cơ học thực tế: W_net = {wnet / 1000.0:.2f} kJ", QColor("#fb923c"), True, 8),
            (f"• Độ biến thiên Entropy: ΔS = {ds:.2f} J/K", QColor("#a855f7"), False, 8),
            (f"• Máy lạnh Carnot: COP_ref = {cop_ref:.2f}", QColor("#38bdf8"), False, 8),
            (f"• Bơm nhiệt Carnot: COP_pump = {cop_hp:.2f}", QColor("#e2e8f0"), False, 8),
            ("• Định luật 2: Không động cơ nào có η vượt quá Carnot!", QColor("#94a3b8"), False, 8),
        ]

        text_y = card_y + 8.0
        for txt, color, is_bold, sz in lines:
            t_item = QGraphicsTextItem(txt)
            t_item.setFont(QFont("Segoe UI", sz, QFont.Weight.Bold if is_bold else QFont.Weight.Normal))
            t_item.setDefaultTextColor(color)
            t_item.setPos(card_x + 8.0, text_y)
            t_item.setZValue(10.5)
            items.append(t_item)
            text_y += 22.0

        return items

'''

    new_content = content.replace(target, method_code + "\n" + target, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully added create_carnot_engine_diagram to stem_tools.py")

if __name__ == "__main__":
    main()
