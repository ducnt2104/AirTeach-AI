"""
Script to cleanly insert PhysicsTools.create_fluid_venturi_diagram into src/ui/stem_tools.py.
"""
import py_compile
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "def create_fluid_venturi_diagram(" in content:
        print("create_fluid_venturi_diagram already in stem_tools.py")
        return

    method_code = '''    @staticmethod
    def create_fluid_venturi_diagram(
        center: tuple[float, float] = (400.0, 300.0),
        preset_code: str = "VENTURI_METER",
        inlet_dia_mm: float = 100.0,
        throat_dia_mm: float = 45.0,
        inlet_speed_ms: float = 8.0,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """
        Create a styled vector fluid dynamics Venturi tube diagram with
        continuity flow equation, Bernoulli static pressure drop, and U-tube manometer.
        """
        items: list[QGraphicsItem] = []
        cx, cy = center
        diag_w = 580.0
        diag_h = 350.0
        x_left = cx - diag_w / 2.0
        y_top = cy - diag_h / 2.0

        # Physical constants & presets (Zero Mock)
        presets = {
            "VENTURI_METER": ("Ống Venturi đo lưu lượng (Venturi Meter)", 1.225, 1000.0, "Không khí (Air)", "Nước (Water)"),
            "CARBURETOR_CHOKE": ("Bộ chế hòa khí động cơ (Carburetor Choke)", 1.225, 740.0, "Không khí nạp", "Xăng (Gasoline)"),
            "AIRPLANE_WING_LIFT": ("Hiệu ứng Bernoulli cánh máy bay (Airfoil Lift)", 1.225, 1000.0, "Luồng khí", "Áp kế chênh lệch"),
            "PERFUME_ASPIRATOR": ("Bình xịt khí nén Bernoulli (Perfume Aspirator)", 1.225, 1000.0, "Không khí thổi", "Chất lỏng hút"),
            "WATER_PIPE_RESTRICTION": ("Đường ống dẫn nước qua áp kế Hg (Water Pipe)", 1000.0, 13600.0, "Nước (Water)", "Thủy ngân (Hg)"),
        }
        info = presets.get(preset_code, presets["VENTURI_METER"])
        title_str, rho_f, rho_m, f_name, m_name = info

        # Exact fluid dynamics equations
        v1 = max(0.1, float(inlet_speed_ms))
        d1 = max(20.0, float(inlet_dia_mm))
        d2 = max(10.0, min(d1, float(throat_dia_mm)))
        area_ratio = (d1 / d2) ** 2
        v2 = v1 * area_ratio
        r1_m = (d1 / 2000.0)
        q_flow = math.pi * (r1_m ** 2) * v1  # m³/s
        delta_p = 0.5 * rho_f * (v2 ** 2 - v1 ** 2)  # Pa
        delta_h_mm = (delta_p / (rho_m * 9.80665)) * 1000.0  # mm

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x_left, y_top, diag_w, diag_h)
        bg_card.setBrush(QBrush(QColor("#0f172a")))
        bg_card.setPen(QPen(pen_color or QColor("#38bdf8"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header Banner
        banner_txt = f"CƠ HỌC CHẤT LƯU: HIỆU ỨNG VENTURI & ĐỊNH LUẬT BERNOULLI — {title_str}"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#f8fafc"))
        banner.setPos(x_left + 14.0, y_top + 10.0)
        banner.setZValue(10.5)
        items.append(banner)

        # 3. Viewport Venturi Tube Canvas (Left Side)
        canvas_x = x_left + 14.0
        canvas_y = y_top + 38.0
        canvas_w = 320.0
        canvas_h = 296.0

        canvas_bg = QGraphicsRectItem(canvas_x, canvas_y, canvas_w, canvas_h)
        canvas_bg.setBrush(QBrush(QColor("#090d16")))
        canvas_bg.setPen(QPen(QColor("#1e293b"), 1.2))
        canvas_bg.setZValue(10.2)
        items.append(canvas_bg)

        # Tube geometry coordinates inside canvas
        tx0 = canvas_x + 16.0
        tx1 = canvas_x + canvas_w - 16.0
        pw = tx1 - tx0
        tube_cy = canvas_y + 85.0

        h1 = (d1 / 120.0) * 75.0
        h2 = (d2 / 120.0) * 75.0

        top_path = QPainterPath()
        bot_path = QPainterPath()

        top_path.moveTo(tx0, tube_cy - h1 / 2.0)
        bot_path.moveTo(tx0, tube_cy + h1 / 2.0)

        cx_mid = tx0 + pw * 0.5
        ctrl_w = pw * 0.22

        top_path.cubicTo(cx_mid - ctrl_w, tube_cy - h1 / 2.0, cx_mid - ctrl_w, tube_cy - h2 / 2.0, cx_mid, tube_cy - h2 / 2.0)
        top_path.cubicTo(cx_mid + ctrl_w, tube_cy - h2 / 2.0, cx_mid + ctrl_w, tube_cy - h1 / 2.0, tx1, tube_cy - h1 / 2.0)

        bot_path.cubicTo(cx_mid - ctrl_w, tube_cy + h1 / 2.0, cx_mid - ctrl_w, tube_cy + h2 / 2.0, cx_mid, tube_cy + h2 / 2.0)
        bot_path.cubicTo(cx_mid + ctrl_w, tube_cy + h2 / 2.0, cx_mid + ctrl_w, tube_cy + h1 / 2.0, tx1, tube_cy + h1 / 2.0)

        # Closed interior for fluid tint
        fill_path = QPainterPath(top_path)
        fill_path.lineTo(tx1, tube_cy + h1 / 2.0)
        reverse_bot = QPainterPath()
        reverse_bot.moveTo(tx1, tube_cy + h1 / 2.0)
        reverse_bot.cubicTo(cx_mid + ctrl_w, tube_cy + h1 / 2.0, cx_mid + ctrl_w, tube_cy + h2 / 2.0, cx_mid, tube_cy + h2 / 2.0)
        reverse_bot.cubicTo(cx_mid - ctrl_w, tube_cy + h2 / 2.0, cx_mid - ctrl_w, tube_cy + h1 / 2.0, tx0, tube_cy + h1 / 2.0)
        fill_path.connectPath(reverse_bot)
        fill_path.closeSubpath()

        fluid_fill = QGraphicsPathItem(fill_path)
        fluid_fill.setBrush(QBrush(QColor(14, 165, 233, 40)))
        fluid_fill.setPen(Qt.PenStyle.NoPen)
        fluid_fill.setZValue(10.3)
        items.append(fluid_fill)

        # Tube walls
        tube_pen = QPen(QColor("#38bdf8"), 2.2)
        top_wall = QGraphicsPathItem(top_path)
        top_wall.setPen(tube_pen)
        top_wall.setZValue(10.4)
        items.append(top_wall)

        bot_wall = QGraphicsPathItem(bot_path)
        bot_wall.setPen(tube_pen)
        bot_wall.setZValue(10.4)
        items.append(bot_wall)

        # Streamlines
        for s_idx, frac in enumerate([-0.3, 0.0, 0.3]):
            s_path = QPainterPath()
            s_path.moveTo(tx0, tube_cy + frac * h1)
            s_path.cubicTo(cx_mid - ctrl_w, tube_cy + frac * h1, cx_mid - ctrl_w, tube_cy + frac * h2, cx_mid, tube_cy + frac * h2)
            s_path.cubicTo(cx_mid + ctrl_w, tube_cy + frac * h2, cx_mid + ctrl_w, tube_cy + frac * h1, tx1, tube_cy + frac * h1)
            s_item = QGraphicsPathItem(s_path)
            s_item.setPen(QPen(QColor(56, 189, 248, 120), 1.0, Qt.PenStyle.DashLine))
            s_item.setZValue(10.35)
            items.append(s_item)

        # Flow velocity vectors
        v1_len = min(60.0, v1 * 3.5)
        v2_len = min(90.0, v2 * 3.5)

        # v1 arrow at inlet
        v1_line = QGraphicsLineItem(tx0 + 10.0, tube_cy, tx0 + 10.0 + v1_len, tube_cy)
        v1_line.setPen(QPen(QColor("#22c55e"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        v1_line.setZValue(10.6)
        items.append(v1_line)

        v1_tip = QGraphicsPolygonItem(QPolygonF([
            QPointF(tx0 + 10.0 + v1_len, tube_cy),
            QPointF(tx0 + 10.0 + v1_len - 8.0, tube_cy - 4.0),
            QPointF(tx0 + 10.0 + v1_len - 8.0, tube_cy + 4.0),
        ]))
        v1_tip.setBrush(QBrush(QColor("#22c55e")))
        v1_tip.setPen(Qt.PenStyle.NoPen)
        v1_tip.setZValue(10.7)
        items.append(v1_tip)

        v1_lbl = QGraphicsTextItem(f"v₁={v1:.1f} m/s")
        v1_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        v1_lbl.setDefaultTextColor(QColor("#22c55e"))
        v1_lbl.setPos(tx0 + 8.0, tube_cy - 20.0)
        v1_lbl.setZValue(10.8)
        items.append(v1_lbl)

        # v2 arrow at throat
        v2_line = QGraphicsLineItem(cx_mid - v2_len / 2.0, tube_cy, cx_mid + v2_len / 2.0, tube_cy)
        v2_line.setPen(QPen(QColor("#f59e0b"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        v2_line.setZValue(10.6)
        items.append(v2_line)

        v2_tip = QGraphicsPolygonItem(QPolygonF([
            QPointF(cx_mid + v2_len / 2.0, tube_cy),
            QPointF(cx_mid + v2_len / 2.0 - 8.0, tube_cy - 4.0),
            QPointF(cx_mid + v2_len / 2.0 - 8.0, tube_cy + 4.0),
        ]))
        v2_tip.setBrush(QBrush(QColor("#f59e0b")))
        v2_tip.setPen(Qt.PenStyle.NoPen)
        v2_tip.setZValue(10.7)
        items.append(v2_tip)

        v2_lbl = QGraphicsTextItem(f"v₂={v2:.1f} m/s")
        v2_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        v2_lbl.setDefaultTextColor(QColor("#f59e0b"))
        v2_lbl.setPos(cx_mid - 25.0, tube_cy - 20.0)
        v2_lbl.setZValue(10.8)
        items.append(v2_lbl)

        # U-Tube Manometer
        tap1_x = tx0 + 35.0
        tap2_x = cx_mid
        mano_top_y = tube_cy + h1 / 2.0
        mano_bot_y = canvas_y + canvas_h - 22.0
        tube_w = 12.0

        mano_path = QPainterPath()
        mano_path.moveTo(tap1_x - tube_w / 2.0, mano_top_y)
        mano_path.lineTo(tap1_x - tube_w / 2.0, mano_bot_y)
        mano_path.lineTo(tap2_x + tube_w / 2.0, mano_bot_y)
        mano_path.lineTo(tap2_x + tube_w / 2.0, tube_cy + h2 / 2.0)
        mano_path.lineTo(tap2_x - tube_w / 2.0, tube_cy + h2 / 2.0)
        mano_path.lineTo(tap2_x - tube_w / 2.0, mano_bot_y - tube_w)
        mano_path.lineTo(tap1_x + tube_w / 2.0, mano_bot_y - tube_w)
        mano_path.lineTo(tap1_x + tube_w / 2.0, mano_top_y)
        mano_path.closeSubpath()

        mano_glass = QGraphicsPathItem(mano_path)
        mano_glass.setBrush(QBrush(QColor(15, 23, 42, 180)))
        mano_glass.setPen(QPen(QColor("#64748b"), 1.2))
        mano_glass.setZValue(10.4)
        items.append(mano_glass)

        # Manometer Liquid Columns
        scale_dh = min(80.0, max(5.0, delta_h_mm * 0.15))
        base_h = 45.0
        h_left_liq = base_h
        h_right_liq = base_h + scale_dh

        left_liq = QGraphicsRectItem(tap1_x - tube_w / 2.0 + 1.0, mano_bot_y - h_left_liq, tube_w - 2.0, h_left_liq)
        left_liq.setBrush(QBrush(QColor("#f43f5e")))
        left_liq.setPen(Qt.PenStyle.NoPen)
        left_liq.setZValue(10.45)
        items.append(left_liq)

        right_liq = QGraphicsRectItem(tap2_x - tube_w / 2.0 + 1.0, mano_bot_y - h_right_liq, tube_w - 2.0, h_right_liq)
        right_liq.setBrush(QBrush(QColor("#f43f5e")))
        right_liq.setPen(Qt.PenStyle.NoPen)
        right_liq.setZValue(10.45)
        items.append(right_liq)

        bot_liq = QGraphicsRectItem(tap1_x - tube_w / 2.0 + 1.0, mano_bot_y - tube_w + 1.0, (tap2_x - tap1_x) + tube_w - 2.0, tube_w - 2.0)
        bot_liq.setBrush(QBrush(QColor("#f43f5e")))
        bot_liq.setPen(Qt.PenStyle.NoPen)
        bot_liq.setZValue(10.45)
        items.append(bot_liq)

        dh_top = mano_bot_y - h_right_liq
        dh_bot = mano_bot_y - h_left_liq
        dh_line = QGraphicsLineItem(tap2_x + tube_w + 4.0, dh_top, tap2_x + tube_w + 4.0, dh_bot)
        dh_line.setPen(QPen(QColor("#fbbf24"), 1.4, Qt.PenStyle.DashLine))
        dh_line.setZValue(10.6)
        items.append(dh_line)

        dh_lbl = QGraphicsTextItem(f"Δh = {delta_h_mm:.1f} mm")
        dh_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        dh_lbl.setDefaultTextColor(QColor("#fbbf24"))
        dh_lbl.setPos(tap2_x + tube_w + 8.0, (dh_top + dh_bot) / 2.0 - 10.0)
        dh_lbl.setZValue(10.7)
        items.append(dh_lbl)

        # 4. Analytics Card (Right Side)
        card_x = canvas_x + canvas_w + 12.0
        card_y = canvas_y
        card_w = diag_w - canvas_w - 40.0
        card_h = canvas_h

        info_card = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_card.setBrush(QBrush(QColor("#090d16")))
        info_card.setPen(QPen(QColor("#1e293b"), 1.2))
        info_card.setZValue(10.2)
        items.append(info_card)

        lines = [
            ("LÝ THUYẾT KHÍ ĐỘNG & CHẤT LƯU", QColor("#f8fafc"), True, 9),
            (f"• Dòng chất lưu: {f_name} (ρ = {rho_f:.3f} kg/m³)", QColor("#38bdf8"), False, 8),
            (f"• Chất lỏng áp kế: {m_name} (ρ = {rho_m:.0f} kg/m³)", QColor("#cbd5e1"), False, 8),
            (f"• Tiết diện đầu: D₁ = {d1:.1f} mm  (v₁ = {v1:.1f} m/s)", QColor("#22c55e"), True, 8),
            (f"• Tiết diện hẹp: D₂ = {d2:.1f} mm  (v₂ = {v2:.1f} m/s)", QColor("#f59e0b"), True, 8),
            (f"• Tỷ số gia tốc vận tốc: v₂/v₁ = {area_ratio:.2f}×", QColor("#38bdf8"), False, 8),
            (f"• Lưu lượng khối tích: Q = {q_flow * 1000.0:.2f} L/s", QColor("#e2e8f0"), True, 8),
            (f"• Độ giảm áp suất tĩnh: ΔP = {delta_p:.1f} Pa", QColor("#f43f5e"), True, 8),
            (f"• Độ chênh cột áp kế: Δh = {delta_h_mm:.1f} mm", QColor("#fbbf24"), True, 8),
            ("1. Phương trình liên tục: A₁·v₁ = A₂·v₂ = Q", QColor("#94a3b8"), False, 7),
            ("2. Định luật Bernoulli: P₁ + ½ρv₁² = P₂ + ½ρv₂²", QColor("#94a3b8"), False, 7),
            ("   => ΔP = P₁ - P₂ = ½·ρ·(v₂² - v₁²)", QColor("#94a3b8"), False, 7),
            ("3. Áp kế chữ U: Δh = ΔP / (ρ_m · g)", QColor("#94a3b8"), False, 7),
        ]

        text_y = card_y + 8.0
        for txt, col, is_bld, sz in lines:
            t_itm = QGraphicsTextItem(txt)
            t_itm.setFont(QFont("Segoe UI", sz, QFont.Weight.Bold if is_bld else QFont.Weight.Normal))
            t_itm.setDefaultTextColor(col)
            t_itm.setPos(card_x + 8.0, text_y)
            t_itm.setZValue(10.5)
            items.append(t_itm)
            text_y += 21.0

        return items
'''

    target = "class ChemistryTools:"
    assert target in content, "Target class ChemistryTools: not found!"

    new_content = content.replace(target, method_code + "\n\n" + target, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    py_compile.compile(path, doraise=True)
    print("Successfully inserted PhysicsTools.create_fluid_venturi_diagram and compiled!")

if __name__ == "__main__":
    main()
