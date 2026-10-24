"""
Add create_gas_chamber_diagram and create_titration_curve_diagram to ChemistryTools.
"""

EXTRA_CHEM_METHODS = '''
    @staticmethod
    def create_gas_chamber_diagram(
        center: tuple[float, float],
        width: float = 380.0,
        height: float = 230.0,
        piston_fraction: float = 0.70,
        temperature: float = 300.0,
        pressure_kpa: float = 101.3,
        gas_type: str = "HELIUM",
        particle_count: int = 20,
    ) -> list[QGraphicsItem]:
        """Creates thermodynamic gas chamber vector diagram with cylinder, piston, and particles."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        # Title banner
        banner_txt = f"KHÍ LÝ TƯỞNG: {gas_type} | P = {pressure_kpa:.1f} kPa | T = {temperature:.0f} K"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        # Background
        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#f8fafc")))
        bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        bg.setZValue(10.0)
        items.append(bg)

        # Cylinder chamber
        c_x = x_left + 20.0
        c_y = y_top + 20.0
        c_w = 180.0
        c_h = 140.0
        chamber = QGraphicsRectItem(c_x, c_y, c_w, c_h)
        chamber.setBrush(QBrush(QColor("#0f172a")))
        chamber.setPen(QPen(QColor("#38bdf8"), 2.0))
        chamber.setZValue(10.5)
        items.append(chamber)

        # Piston position
        piston_h = 14.0
        piston_y = c_y + (1.0 - piston_fraction) * (c_h - piston_h)
        piston = QGraphicsRectItem(c_x + 2.0, piston_y, c_w - 4.0, piston_h)
        piston.setBrush(QBrush(QColor("#94a3b8")))
        piston.setPen(QPen(QColor("#cbd5e1"), 1.2))
        piston.setZValue(11.0)
        items.append(piston)

        # Piston rod & handle
        rod = QGraphicsLineItem(c_x + c_w / 2.0, c_y - 15.0, c_x + c_w / 2.0, piston_y)
        rod.setPen(QPen(QColor("#64748b"), 3.0))
        rod.setZValue(10.8)
        items.append(rod)

        handle = QGraphicsLineItem(c_x + c_w / 2.0 - 20.0, c_y - 15.0, c_x + c_w / 2.0 + 20.0, c_y - 15.0)
        handle.setPen(QPen(QColor("#475569"), 4.0))
        handle.setZValue(10.9)
        items.append(handle)

        # Gas particles inside chamber
        import math
        np_count = max(10, min(particle_count, 30))
        for p in range(np_count):
            frac_x = (p * 0.173 + 0.1) % 0.8 + 0.1
            frac_y = (p * 0.231 + 0.1) % 0.8 + 0.1
            px = c_x + 8.0 + frac_x * (c_w - 16.0)
            py = piston_y + piston_h + 6.0 + frac_y * (c_y + c_h - piston_y - piston_h - 12.0)
            dot = QGraphicsEllipseItem(px - 3.0, py - 3.0, 6.0, 6.0)
            dot.setBrush(QBrush(QColor("#38bdf8" if gas_type == "HELIUM" else "#f59e0b")))
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setZValue(11.5)
            items.append(dot)

        # Pressure gauge meter
        gauge_cx = x_left + 260.0
        gauge_cy = y_top + 60.0
        gauge_r = 35.0
        g_circle = QGraphicsEllipseItem(gauge_cx - gauge_r, gauge_cy - gauge_r, gauge_r * 2, gauge_r * 2)
        g_circle.setBrush(QBrush(QColor("#ffffff")))
        g_circle.setPen(QPen(QColor("#0284c7"), 1.8))
        g_circle.setZValue(11.0)
        items.append(g_circle)

        # Gauge needle
        g_angle = (pressure_kpa / 300.0) * math.pi - math.pi / 2.0
        needle = QGraphicsLineItem(gauge_cx, gauge_cy, gauge_cx + 26.0 * math.cos(g_angle), gauge_cy + 26.0 * math.sin(g_angle))
        needle.setPen(QPen(QColor("#ef4444"), 2.0))
        needle.setZValue(11.5)
        items.append(needle)

        # Gauge text
        p_txt = QGraphicsTextItem(f"{pressure_kpa:.1f} kPa")
        p_txt.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        p_txt.setDefaultTextColor(QColor("#0f172a"))
        p_txt.setPos(gauge_cx - 24.0, gauge_cy + 10.0)
        p_txt.setZValue(11.6)
        items.append(p_txt)

        # Equation note
        eq_y = y_top + height - 50.0
        eq_txt = (
            f"Phương Trình Trạng Thái Khí Lý Tưởng: P · V = n · R · T\\n"
            f"Phân Bố Maxwell-Boltzmann: f(v) = 4π · (m/(2πkT))^(3/2) · v² · e^(-mv²/(2kT))"
        )
        eq_item = QGraphicsTextItem(eq_txt)
        eq_item.setFont(QFont("Consolas", 8))
        eq_item.setDefaultTextColor(QColor("#334155"))
        eq_item.setPos(x_left + 15.0, eq_y)
        eq_item.setZValue(11.0)
        items.append(eq_item)

        return items

    @staticmethod
    def create_titration_curve_diagram(
        center: tuple[float, float],
        width: float = 420.0,
        height: float = 260.0,
        acid_type: str = "WEAK_ACID",
        v_titrant: float = 25.0,
        v_eq: float = 25.0,
        c_acid: float = 0.10,
        c_base: float = 0.10,
    ) -> list[QGraphicsItem]:
        """Creates laboratory acid-base titration curve and setup diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        banner_txt = f"CHUẨN ĐỘ AXIT - BAZƠ: {acid_type} | V_eq = {v_eq:.1f} mL"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        # Background
        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#f8fafc")))
        bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        bg.setZValue(10.0)
        items.append(bg)

        # Burette Glass Tube (Left)
        b_x = x_left + 35.0
        b_y = y_top + 20.0
        b_w = 16.0
        b_h = 90.0
        burette = QGraphicsRectItem(b_x, b_y, b_w, b_h)
        burette.setBrush(QBrush(QColor("#e0f2fe")))
        burette.setPen(QPen(QColor("#0284c7"), 1.5))
        burette.setZValue(10.5)
        items.append(burette)

        # Titrant fluid inside burette
        fluid_h = max(5.0, b_h * (1.0 - v_titrant / (v_eq * 2.0)))
        fluid = QGraphicsRectItem(b_x + 1.0, b_y + (b_h - fluid_h), b_w - 2.0, fluid_h)
        fluid.setBrush(QBrush(QColor("#38bdf8")))
        fluid.setPen(Qt.PenStyle.NoPen)
        fluid.setZValue(10.6)
        items.append(fluid)

        # Burette stopcock valve
        valve = QGraphicsRectItem(b_x - 5.0, b_y + b_h, b_w + 10.0, 6.0)
        valve.setBrush(QBrush(QColor("#0369a1")))
        valve.setPen(Qt.PenStyle.NoPen)
        valve.setZValue(10.7)
        items.append(valve)

        # Erlenmeyer Flask below
        f_path = QPainterPath()
        f_top = b_y + b_h + 16.0
        f_path.moveTo(b_x + 2.0, f_top)
        f_path.lineTo(b_x - 18.0, f_top + 45.0)
        f_path.lineTo(b_x + b_w + 18.0, f_top + 45.0)
        f_path.lineTo(b_x + b_w - 2.0, f_top)
        f_path.closeSubpath()
        flask = QGraphicsPathItem(f_path)
        flask_col = QColor("#f43f5e") if v_titrant >= v_eq else QColor("#ffffff")
        flask.setBrush(QBrush(flask_col))
        flask.setPen(QPen(QColor("#0284c7"), 1.5))
        flask.setZValue(10.5)
        items.append(flask)

        # pH Curve Graph (Right)
        g_x = x_left + 110.0
        g_y = y_top + 20.0
        g_w = width - 125.0
        g_h = 135.0
        g_bg = QGraphicsRectItem(g_x, g_y, g_w, g_h)
        g_bg.setBrush(QBrush(QColor("#ffffff")))
        g_bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        g_bg.setZValue(10.5)
        items.append(g_bg)

        # pH axes lines
        ax_h = QGraphicsLineItem(g_x, g_y + g_h - 15.0, g_x + g_w, g_y + g_h - 15.0)
        ax_h.setPen(QPen(QColor("#64748b"), 1.0))
        ax_h.setZValue(10.6)
        items.append(ax_h)

        # Sigmoidal pH curve
        curve = QPainterPath()
        steps = 50
        import math
        for s in range(steps + 1):
            v = (s / float(steps)) * (v_eq * 2.0)
            px = g_x + (s / float(steps)) * (g_w - 20.0) + 10.0
            # Sigmoidal pH model
            diff = v - v_eq
            sigm = 1.0 / (1.0 + math.exp(-0.8 * diff))
            ph_val = 3.0 + 8.0 * sigm
            py = g_y + g_h - 15.0 - (ph_val / 14.0) * (g_h - 30.0)
            if s == 0:
                curve.moveTo(px, py)
            else:
                curve.lineTo(px, py)

        c_item = QGraphicsPathItem(curve)
        c_item.setPen(QPen(QColor("#2563eb"), 2.0))
        c_item.setZValue(11.0)
        items.append(c_item)

        # Equivalence point marker E(V_eq, pH_eq)
        eq_x = g_x + 0.5 * (g_w - 20.0) + 10.0
        eq_y_pos = g_y + g_h - 15.0 - (7.0 / 14.0) * (g_h - 30.0)
        eq_dot = QGraphicsEllipseItem(eq_x - 4.0, eq_y_pos - 4.0, 8.0, 8.0)
        eq_dot.setBrush(QBrush(QColor("#ef4444")))
        eq_dot.setPen(QPen(QColor("#ffffff"), 1.5))
        eq_dot.setZValue(11.5)
        items.append(eq_dot)

        # Equations note
        note_y = y_top + height - 60.0
        note_txt = (
            f"Phương Trình Henderson-Hasselbalch: pH = pKa + log10([A⁻] / [HA])\\n"
            f"Điểm Nửa Tương Đương: pH = pKa | Điểm Tương Đương: dpH/dV đạt cực đại"
        )
        note_item = QGraphicsTextItem(note_txt)
        note_item.setFont(QFont("Consolas", 8))
        note_item.setDefaultTextColor(QColor("#334155"))
        note_item.setPos(x_left + 15.0, note_y)
        note_item.setZValue(11.0)
        items.append(note_item)

        return items
'''

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '    @staticmethod\n    def create_reaction_kinetics_diagram('
idx = text.find(target)
assert idx != -1, "Target reaction kinetics diagram not found"

text = text[:idx] + EXTRA_CHEM_METHODS + '\n' + text[idx:]

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully added create_gas_chamber_diagram and create_titration_curve_diagram!")
