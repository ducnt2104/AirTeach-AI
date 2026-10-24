"""
Helper script to insert create_spacetime_relativity_diagram into PhysicsTools in src/ui/stem_tools.py
"""
import math
import sys

def main():
    with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if 'def create_spacetime_relativity_diagram(' in content:
        print('create_spacetime_relativity_diagram already exists!')
        return

    method_code = '''    @staticmethod
    def create_spacetime_relativity_diagram(
        center: tuple[float, float],
        width: float = 480.0,
        height: float = 300.0,
        beta: float = 0.60,
        gamma: Optional[float] = None,
        preset: str = "LIGHT_CLOCK_GEDANKEN",
        event1: tuple[float, float] = (0.0, 0.0),
        event2: tuple[float, float] = (3.0, 4.0),
    ) -> list[QGraphicsItem]:
        """Creates an authentic Special Relativity Minkowski spacetime vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        b_val = max(0.0, min(0.99, float(beta)))
        g_val = gamma if gamma is not None else (1.0 / math.sqrt(max(0.0001, 1.0 - b_val * b_val)))

        # 1. Title Banner
        banner_txt = f"KHÔNG-THỜI GIAN MINKOWSKI (β = {b_val:.3f}, γ = {g_val:.3f})"
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

        origin_x = pane_x + pane_w / 2.0
        origin_y = pane_y + pane_h / 2.0
        scale = 18.0

        # 3. Light Cones at 45°
        w_cone = 110.0
        cone_future = QPainterPath()
        cone_future.moveTo(origin_x, origin_y)
        cone_future.lineTo(origin_x + w_cone, origin_y - w_cone)
        cone_future.lineTo(origin_x - w_cone, origin_y - w_cone)
        cone_future.closeSubpath()

        cone_future_item = QGraphicsPathItem(cone_future)
        cone_future_item.setBrush(QBrush(QColor(245, 158, 11, 35)))
        cone_future_item.setPen(Qt.PenStyle.NoPen)
        cone_future_item.setZValue(10.2)
        items.append(cone_future_item)

        cone_past = QPainterPath()
        cone_past.moveTo(origin_x, origin_y)
        cone_past.lineTo(origin_x + w_cone, origin_y + w_cone)
        cone_past.lineTo(origin_x - w_cone, origin_y + w_cone)
        cone_past.closeSubpath()

        cone_past_item = QGraphicsPathItem(cone_past)
        cone_past_item.setBrush(QBrush(QColor(245, 158, 11, 25)))
        cone_past_item.setPen(Qt.PenStyle.NoPen)
        cone_past_item.setZValue(10.2)
        items.append(cone_past_item)

        # Light Cone Boundary Rays (ct = ±x)
        ray_cone1 = QGraphicsLineItem(origin_x - w_cone, origin_y + w_cone, origin_x + w_cone, origin_y - w_cone)
        ray_cone1.setPen(QPen(QColor("#fbbf24"), 1.2, Qt.PenStyle.DashDotLine))
        ray_cone1.setZValue(10.3)
        items.append(ray_cone1)

        ray_cone2 = QGraphicsLineItem(origin_x - w_cone, origin_y - w_cone, origin_x + w_cone, origin_y + w_cone)
        ray_cone2.setPen(QPen(QColor("#fbbf24"), 1.2, Qt.PenStyle.DashDotLine))
        ray_cone2.setZValue(10.3)
        items.append(ray_cone2)

        # 4. Invariant Timelike Hyperbola ((ct)² - x² = 4)
        hyp_path = QPainterPath()
        tau = 2.0
        first_hyp = True
        for step in range(-40, 41):
            x_val = (step / 40.0) * 5.0
            ct_val = math.sqrt(tau * tau + x_val * x_val)
            hx = origin_x + x_val * scale
            hy = origin_y - ct_val * scale
            if first_hyp:
                hyp_path.moveTo(hx, hy)
                first_hyp = False
            else:
                hyp_path.lineTo(hx, hy)

        hyp_item = QGraphicsPathItem(hyp_path)
        hyp_item.setPen(QPen(QColor(56, 189, 248, 120), 1.1, Qt.PenStyle.DashLine))
        hyp_item.setZValue(10.4)
        items.append(hyp_item)

        # 5. Base S Axes (x and ct)
        axis_x = QGraphicsLineItem(pane_x + 10.0, origin_y, pane_x + pane_w - 10.0, origin_y)
        axis_x.setPen(QPen(QColor("#94a3b8"), 1.5))
        axis_x.setZValue(10.6)
        items.append(axis_x)

        axis_ct = QGraphicsLineItem(origin_x, pane_y + pane_h - 10.0, origin_x, pane_y + 10.0)
        axis_ct.setPen(QPen(QColor("#94a3b8"), 1.5))
        axis_ct.setZValue(10.6)
        items.append(axis_ct)

        lbl_x = QGraphicsTextItem("x")
        lbl_x.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl_x.setDefaultTextColor(QColor("#cbd5e1"))
        lbl_x.setPos(pane_x + pane_w - 20.0, origin_y - 18.0)
        lbl_x.setZValue(10.7)
        items.append(lbl_x)

        lbl_ct = QGraphicsTextItem("ct")
        lbl_ct.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl_ct.setDefaultTextColor(QColor("#cbd5e1"))
        lbl_ct.setPos(origin_x + 4.0, pane_y + 2.0)
        lbl_ct.setZValue(10.7)
        items.append(lbl_ct)

        # 6. Boosted S' Axes (ct' and x')
        if b_val > 0.01:
            theta = math.atan(b_val)
            L_axis = 100.0
            ct_prime = QGraphicsLineItem(
                origin_x - math.sin(theta) * L_axis,
                origin_y + math.cos(theta) * L_axis,
                origin_x + math.sin(theta) * L_axis,
                origin_y - math.cos(theta) * L_axis,
            )
            ct_prime.setPen(QPen(QColor("#10b981"), 1.8))
            ct_prime.setZValue(10.8)
            items.append(ct_prime)

            x_prime = QGraphicsLineItem(
                origin_x - math.cos(theta) * L_axis,
                origin_y + math.sin(theta) * L_axis,
                origin_x + math.cos(theta) * L_axis,
                origin_y - math.cos(theta) * L_axis,
            )
            x_prime.setPen(QPen(QColor("#10b981"), 1.8))
            x_prime.setZValue(10.8)
            items.append(x_prime)

            lbl_ctp = QGraphicsTextItem("ct'")
            lbl_ctp.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            lbl_ctp.setDefaultTextColor(QColor("#34d399"))
            lbl_ctp.setPos(origin_x + math.sin(theta) * L_axis + 4.0, origin_y - math.cos(theta) * L_axis - 6.0)
            lbl_ctp.setZValue(10.9)
            items.append(lbl_ctp)

            lbl_xp = QGraphicsTextItem("x'")
            lbl_xp.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            lbl_xp.setDefaultTextColor(QColor("#34d399"))
            lbl_xp.setPos(origin_x + math.cos(theta) * L_axis + 4.0, origin_y - math.sin(theta) * L_axis - 10.0)
            lbl_xp.setZValue(10.9)
            items.append(lbl_xp)

        # 7. Interactive Events E1 and E2
        e1_x, e1_ct = event1
        e2_x, e2_ct = event2
        p1_x = origin_x + e1_x * scale
        p1_y = origin_y - e1_ct * scale
        p2_x = origin_x + e2_x * scale
        p2_y = origin_y - e2_ct * scale

        # Spacetime interval line
        ds_sq = (e2_ct - e1_ct) ** 2 - (e2_x - e1_x) ** 2
        line_iv = QGraphicsLineItem(p1_x, p1_y, p2_x, p2_y)
        iv_color = QColor("#38bdf8") if ds_sq > 0.001 else (QColor("#ec4899") if ds_sq < -0.001 else QColor("#fbbf24"))
        line_iv.setPen(QPen(iv_color, 1.8, Qt.PenStyle.SolidLine if ds_sq > 0 else Qt.PenStyle.DashLine))
        line_iv.setZValue(11.0)
        items.append(line_iv)

        # E1 marker
        dot_e1 = QGraphicsEllipseItem(p1_x - 4.0, p1_y - 4.0, 8.0, 8.0)
        dot_e1.setBrush(QBrush(QColor("#f59e0b")))
        dot_e1.setPen(QPen(QColor("#ffffff"), 1.0))
        dot_e1.setZValue(11.2)
        items.append(dot_e1)

        # E2 marker
        dot_e2 = QGraphicsEllipseItem(p2_x - 4.0, p2_y - 4.0, 8.0, 8.0)
        dot_e2.setBrush(QBrush(QColor("#0ea5e9")))
        dot_e2.setPen(QPen(QColor("#ffffff"), 1.0))
        dot_e2.setZValue(11.2)
        items.append(dot_e2)

        lbl_e1 = QGraphicsTextItem(f"E₁({e1_x:.1f}, {e1_ct:.1f})")
        lbl_e1.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_e1.setDefaultTextColor(QColor("#fbbf24"))
        lbl_e1.setPos(p1_x + 5.0, p1_y - 14.0)
        lbl_e1.setZValue(11.3)
        items.append(lbl_e1)

        lbl_e2 = QGraphicsTextItem(f"E₂({e2_x:.1f}, {e2_ct:.1f})")
        lbl_e2.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_e2.setDefaultTextColor(QColor("#38bdf8"))
        lbl_e2.setPos(p2_x + 5.0, p2_y - 14.0)
        lbl_e2.setZValue(11.3)
        items.append(lbl_e2)

        # 8. Right-side Theory and Metrics Card
        card_x = pane_x + pane_w + 12.0
        card_y = pane_y
        card_w = width - pane_w - 34.0
        card_h = pane_h

        info_bg = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_bg.setBrush(QBrush(QColor(30, 41, 59, 230)))
        info_bg.setPen(QPen(QColor("#334155"), 1.0))
        info_bg.setZValue(10.0)
        items.append(info_bg)

        # Metric text lines
        class_str = "THỜI GIAN TÍNH (Timelike)" if ds_sq > 0.001 else ("KHÔNG GIAN TÍNH (Spacelike)" if ds_sq < -0.001 else "ÁNH SÁNG TÍNH (Lightlike)")
        proper_val = math.sqrt(ds_sq) if ds_sq > 0 else (math.sqrt(-ds_sq) if ds_sq < 0 else 0.0)
        proper_label = "Δτ (proper time)" if ds_sq > 0 else "Δσ (proper length)"

        lines = [
            ("LÝ THUYẾT & THỐNG KÊ TƯƠNG ĐỐI HẸP", QColor("#f8fafc"), True, 9),
            (f"• Vận tốc tương đối: β = v/c = {b_val:.3f}", QColor("#10b981"), False, 8),
            (f"• Hệ số Lorentz: γ = 1/√(1−β²) = {g_val:.3f}", QColor("#38bdf8"), False, 8),
            (f"• Khoảng Minkowski: Δs² = c²Δt² − Δx² = {ds_sq:.3f}", QColor("#fbbf24"), True, 8),
            (f"• Phân loại: {class_str}", QColor("#cbd5e1"), False, 8),
            (f"• Đại lượng riêng: {proper_label} = {proper_val:.3f}", QColor("#cbd5e1"), False, 8),
            ("• Dãn nở thời gian: Δt = γ·Δτ (Moving clocks run slow)", QColor("#94a3b8"), False, 8),
            ("• Co ngắn Lorentz: L = L₀/γ (Length contraction)", QColor("#94a3b8"), False, 8),
            ("• Tính tương đối của sự đồng thời (Relativity of simultaneity)", QColor("#94a3b8"), False, 8),
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
    assert target in content, 'target not found'
    new_content = content.replace(target, method_code + '\n\n' + target, 1)

    with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Successfully inserted create_spacetime_relativity_diagram into stem_tools.py!')

if __name__ == '__main__':
    main()
