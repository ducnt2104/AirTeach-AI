"""
Script to add missing STEM methods to stem_tools.py:
  - MathTools: create_geometric_triangle, create_angle_arc, create_circle_with_radius
  - BiologyTools: create_dna_double_helix
  - ChemistryTools: create_molecular_geometry, create_organic_isomer_diagram
  - InformaticsTools: get_flowchart_pins, snap_to_nearest_pin, create_manhattan_wire, create_interactive_logic_gate
"""
import sys

def add_math_tools():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    math_methods = '''    @staticmethod
    def create_geometric_triangle(
        v0: tuple[float, float],
        v1: tuple[float, float],
        v2: tuple[float, float],
        labels: tuple[str, str, str] = ("A", "B", "C"),
        color: QColor = QColor("#1a73e8"),
        fill_color: QColor = QColor("#e8f0fe"),
        show_right_angle: bool = False,
        right_angle_vertex_idx: int = 0,
    ) -> list[QGraphicsItem]:
        """Creates a 2D geometric triangle with vertex labels and optional right-angle symbol."""
        items: list[QGraphicsItem] = []
        poly = QPolygonF([QPointF(*v0), QPointF(*v1), QPointF(*v2)])
        tri_item = QGraphicsPolygonItem(poly)
        tri_item.setPen(QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        tri_item.setBrush(QBrush(fill_color))
        tri_item.setZValue(11.0)
        items.append(tri_item)

        # Labels
        vertices = [v0, v1, v2]
        centroid_x = sum(v[0] for v in vertices) / 3.0
        centroid_y = sum(v[1] for v in vertices) / 3.0
        lbl_font = QFont("Segoe UI", 11, QFont.Weight.Bold)

        for i, (vx, vy) in enumerate(vertices):
            lbl_text = labels[i] if i < len(labels) else f"V{i}"
            dx = vx - centroid_x
            dy = vy - centroid_y
            dist = math.hypot(dx, dy) or 1.0
            lx = vx + (dx / dist) * 16.0 - 8.0
            ly = vy + (dy / dist) * 16.0 - 12.0
            t_item = QGraphicsTextItem(lbl_text)
            t_item.setFont(lbl_font)
            t_item.setDefaultTextColor(color)
            t_item.setPos(lx, ly)
            t_item.setZValue(12.5)
            items.append(t_item)

        # Right-angle marker
        if show_right_angle and 0 <= right_angle_vertex_idx < 3:
            v_apex = vertices[right_angle_vertex_idx]
            v_prev = vertices[(right_angle_vertex_idx - 1) % 3]
            v_next = vertices[(right_angle_vertex_idx + 1) % 3]
            d1x, d1y = v_prev[0] - v_apex[0], v_prev[1] - v_apex[1]
            d2x, d2y = v_next[0] - v_apex[0], v_next[1] - v_apex[1]
            l1, l2 = math.hypot(d1x, d1y) or 1.0, math.hypot(d2x, d2y) or 1.0
            u1x, u1y = d1x / l1, d1y / l1
            u2x, u2y = d2x / l2, d2y / l2
            s_sz = 14.0
            p_a = (v_apex[0] + u1x * s_sz, v_apex[1] + u1y * s_sz)
            p_c = (v_apex[0] + u2x * s_sz, v_apex[1] + u2y * s_sz)
            p_b = (p_a[0] + u2x * s_sz, p_a[1] + u2y * s_sz)
            sq_path = QPainterPath()
            sq_path.moveTo(*p_a)
            sq_path.lineTo(*p_b)
            sq_path.lineTo(*p_c)
            sq_item = QGraphicsPathItem(sq_path)
            sq_item.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine))
            sq_item.setZValue(12.0)
            items.append(sq_item)

        return items

    @staticmethod
    def create_angle_arc(
        vertex: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        radius: float = 40.0,
        label: str = "",
        color: QColor = QColor("#22c55e"),
    ) -> list[QGraphicsItem]:
        """Create a geometric angle arc with ray lines and angle label."""
        items: list[QGraphicsItem] = []
        vx, vy = vertex
        x1, y1 = p1
        x2, y2 = p2

        a1 = math.atan2(y1 - vy, x1 - vx)
        a2 = math.atan2(y2 - vy, x2 - vx)
        diff = a2 - a1
        while diff < 0:
            diff += 2 * math.pi
        while diff >= 2 * math.pi:
            diff -= 2 * math.pi

        arc_path = QPainterPath()
        start_deg = -math.degrees(a1)
        span_deg = -math.degrees(diff)
        arc_rect = QRectF(vx - radius, vy - radius, 2 * radius, 2 * radius)
        arc_path.arcMoveTo(arc_rect, start_deg)
        arc_path.arcTo(arc_rect, start_deg, span_deg)

        arc_item = QGraphicsPathItem(arc_path)
        arc_item.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine))
        arc_item.setBrush(Qt.BrushStyle.NoBrush)
        arc_item.setZValue(11.5)
        items.append(arc_item)

        if label:
            mid_a = a1 + diff / 2.0
            lx = vx + (radius + 16.0) * math.cos(mid_a)
            ly = vy + (radius + 16.0) * math.sin(mid_a)
            lbl = QGraphicsTextItem(label)
            lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            lbl.setDefaultTextColor(color)
            lbl.setPos(lx - 12.0, ly - 12.0)
            lbl.setZValue(12.0)
            items.append(lbl)

        return items

    @staticmethod
    def create_circle_with_radius(
        center: tuple[float, float],
        radius: float = 80.0,
        label: str = "R",
        color: QColor = QColor("#1a73e8"),
        fill_color: QColor = QColor("#e8f0fe"),
    ) -> list[QGraphicsItem]:
        """Creates a circle with radius dimension line and center marker."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        circ = QGraphicsEllipseItem(cx - radius, cy - radius, 2 * radius, 2 * radius)
        circ.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine))
        circ.setBrush(QBrush(fill_color))
        circ.setZValue(10.0)
        items.append(circ)

        # Center dot
        dot = QGraphicsEllipseItem(cx - 3.0, cy - 3.0, 6.0, 6.0)
        dot.setPen(Qt.PenStyle.NoPen)
        dot.setBrush(QBrush(color))
        dot.setZValue(12.0)
        items.append(dot)

        # Radius line (horizontal right)
        r_line = QGraphicsLineItem(cx, cy, cx + radius, cy)
        r_line.setPen(QPen(color, 1.8, Qt.PenStyle.DashLine))
        r_line.setZValue(11.0)
        items.append(r_line)

        # Label
        lbl = QGraphicsTextItem(label)
        lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl.setDefaultTextColor(color)
        lbl.setPos(cx + radius * 0.45, cy - 20.0)
        lbl.setZValue(12.0)
        items.append(lbl)

        return items
'''
    target = "class PhysicsTools:"
    if "create_geometric_triangle" not in code and target in code:
        code = code.replace(target, math_methods + "\n\n" + target, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("Added MathTools methods!")


def add_biology_dna():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    dna_code = '''    @staticmethod
    def create_dna_double_helix(
        start: tuple[float, float],
        end: tuple[float, float],
        turns: float = 2.0,
        amplitude: float = 30.0,
        rungs: int = 12,
        color1: QColor = QColor("#2563eb"),
        color2: QColor = QColor("#dc2626"),
        show_labels: bool = True,
    ) -> list[QGraphicsItem]:
        """Create vector DNA double helix diagram with 2 continuous backbone strands, 24 half-rungs, and 5'/3' labels."""
        items: list[QGraphicsItem] = []
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1.0:
            return items

        angle = math.atan2(dy, dx)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        num_samples = 60
        strand1_path = QPainterPath()
        strand2_path = QPainterPath()

        for i in range(num_samples + 1):
            t = i / num_samples
            s_dist = t * length
            theta = 2.0 * math.pi * turns * t
            offset1 = amplitude * math.sin(theta)
            offset2 = -offset1

            p1x = x0 + s_dist * cos_a - offset1 * sin_a
            p1y = y0 + s_dist * sin_a + offset1 * cos_a
            p2x = x0 + s_dist * cos_a - offset2 * sin_a
            p2y = y0 + s_dist * sin_a + offset2 * cos_a

            if i == 0:
                strand1_path.moveTo(p1x, p1y)
                strand2_path.moveTo(p2x, p2y)
            else:
                strand1_path.lineTo(p1x, p1y)
                strand2_path.lineTo(p2x, p2y)

        s1_item = QGraphicsPathItem(strand1_path)
        s1_item.setPen(QPen(color1, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        s1_item.setZValue(12.0)
        items.append(s1_item)

        s2_item = QGraphicsPathItem(strand2_path)
        s2_item.setPen(QPen(color2, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        s2_item.setZValue(12.0)
        items.append(s2_item)

        if rungs > 0:
            rung_pen1 = QPen(color1, 2.0, Qt.PenStyle.SolidLine)
            rung_pen2 = QPen(color2, 2.0, Qt.PenStyle.SolidLine)

            for r_idx in range(rungs):
                t = (r_idx + 0.5) / rungs
                s_dist = t * length
                theta = 2.0 * math.pi * turns * t
                offset1 = amplitude * math.sin(theta)
                offset2 = -offset1

                p1x = x0 + s_dist * cos_a - offset1 * sin_a
                p1y = y0 + s_dist * sin_a + offset1 * cos_a
                p2x = x0 + s_dist * cos_a - offset2 * sin_a
                p2y = y0 + s_dist * sin_a + offset2 * cos_a

                mid_x = (p1x + p2x) / 2.0
                mid_y = (p1y + p2y) / 2.0
                rx, ry = p2x - p1x, p2y - p1y
                rlen = math.hypot(rx, ry) or 1.0
                rux, ruy = rx / rlen, ry / rlen

                gap = 6.0
                h1_end_x = mid_x - rux * (gap / 2.0)
                h1_end_y = mid_y - ruy * (gap / 2.0)
                h2_start_x = mid_x + rux * (gap / 2.0)
                h2_start_y = mid_y + ruy * (gap / 2.0)

                seg1 = QGraphicsLineItem(p1x, p1y, h1_end_x, h1_end_y)
                seg1.setPen(rung_pen1)
                seg1.setZValue(11.5)
                items.append(seg1)

                seg2 = QGraphicsLineItem(h2_start_x, h2_start_y, p2x, p2y)
                seg2.setPen(rung_pen2)
                seg2.setZValue(11.5)
                items.append(seg2)

        if show_labels:
            for txt_val, lx, ly in [("5'", x0 - 24.0, y0 - 15.0), ("3'", x0 - 24.0, y0 + 5.0),
                                    ("3'", x1 + 10.0, y1 - 15.0), ("5'", x1 + 10.0, y1 + 5.0)]:
                t_lbl = QGraphicsTextItem(txt_val)
                t_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                t_lbl.setDefaultTextColor(QColor("#f8fafc"))
                t_lbl.setPos(lx, ly)
                t_lbl.setZValue(12.5)
                items.append(t_lbl)

        return items
'''
    target = "    @staticmethod\n    def create_punnett_square("
    if "create_dna_double_helix" not in code and target in code:
        code = code.replace(target, dna_code + "\n" + target, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("Added BiologyTools.create_dna_double_helix!")


def add_chem_methods():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    mol_geom_code = '''    @staticmethod
    def create_molecular_geometry(
        center: tuple[float, float],
        geometry_type: str = "tetrahedral",
        scale: float = 80.0,
        title: str = "",
        show_bonds: bool = True,
        show_labels: bool = True,
    ) -> list[QGraphicsItem]:
        """Create 3D VSEPR molecular geometry vector diagram with realistic chemical elements."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        g_lower = geometry_type.lower()

        if "bent" in g_lower or "chữ v" in g_lower:
            c_sym = "O"
            l_sym = "H"
            def_title = "Mô hình Góc / Chữ V (H₂O)"
            ligand_offsets = [(-scale * 0.707, scale * 0.707), (scale * 0.707, scale * 0.707)]
        elif "linear" in g_lower or "thẳng" in g_lower:
            c_sym = "C"
            l_sym = "O"
            def_title = "Mô hình Đường thẳng (CO₂)"
            ligand_offsets = [(-scale, 0.0), (scale, 0.0)]
        elif "planar" in g_lower or "tam giác phẳng" in g_lower:
            c_sym = "B"
            l_sym = "F"
            def_title = "Mô hình Tam giác phẳng (BF₃)"
            ligand_offsets = [
                (0.0, -scale),
                (-scale * 0.866, scale * 0.5),
                (scale * 0.866, scale * 0.5),
            ]
        elif "pyramidal" in g_lower or "chóp" in g_lower:
            c_sym = "N"
            l_sym = "H"
            def_title = "Mô hình Chóp tam giác (NH₃)"
            ligand_offsets = [
                (-scale * 0.70, scale * 0.6),
                (scale * 0.70, scale * 0.6),
                (0.0, scale * 0.85),
            ]
        elif "octahedral" in g_lower or "bát diện" in g_lower:
            c_sym = "S"
            l_sym = "F"
            def_title = "Mô hình Bát diện đều (SF₆)"
            ligand_offsets = [
                (0.0, -scale),
                (0.0, scale),
                (-scale, 0.0),
                (scale, 0.0),
                (-scale * 0.55, -scale * 0.45),
                (scale * 0.55, scale * 0.45),
            ]
        else:  # tetrahedral
            c_sym = "C"
            l_sym = "H"
            def_title = "Mô hình Tứ diện đều (CH₄)"
            ligand_offsets = [
                (0.0, -scale * 0.95),
                (-scale * 0.82, scale * 0.55),
                (scale * 0.82, scale * 0.55),
                (0.0, scale * 0.75),
            ]

        # Title
        t_text = title if title else def_title
        t_item = QGraphicsTextItem(t_text)
        t_item.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        t_item.setDefaultTextColor(QColor("#202124"))
        t_item.setPos(cx - 100.0, cy - scale - 38.0)
        t_item.setZValue(12.0)
        items.append(t_item)

        # Bonds
        if show_bonds:
            pen_bond = QPen(QColor("#5f6368"), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            for lx, ly in ligand_offsets:
                b_line = QGraphicsLineItem(cx, cy, cx + lx, cy + ly)
                b_line.setPen(pen_bond)
                b_line.setZValue(10.5)
                items.append(b_line)

        # Center atom
        c_radius = 22.0
        c_atom = QGraphicsEllipseItem(cx - c_radius, cy - c_radius, 2 * c_radius, 2 * c_radius)
        c_atom.setBrush(QBrush(QColor("#334155")))
        c_atom.setPen(QPen(QColor("#ffffff"), 2.0))
        c_atom.setZValue(12.0)
        items.append(c_atom)

        if show_labels:
            c_lbl = QGraphicsTextItem(c_sym)
            c_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            c_lbl.setDefaultTextColor(QColor("#ffffff"))
            c_lbl.setPos(cx - 8.0, cy - 12.0)
            c_lbl.setZValue(12.5)
            items.append(c_lbl)

        # Ligand atoms
        l_radius = 16.0
        for lx, ly in ligand_offsets:
            px, py = cx + lx, cy + ly
            l_atom = QGraphicsEllipseItem(px - l_radius, py - l_radius, 2 * l_radius, 2 * l_radius)
            l_atom.setBrush(QBrush(QColor("#cbd5e1")))
            l_atom.setPen(QPen(QColor("#0f172a"), 1.5))
            l_atom.setZValue(11.5)
            items.append(l_atom)

            if show_labels:
                l_lbl = QGraphicsTextItem(l_sym)
                l_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                l_lbl.setDefaultTextColor(QColor("#0f172a"))
                l_lbl.setPos(px - 7.0, py - 11.0)
                l_lbl.setZValue(12.0)
                items.append(l_lbl)

        return items
'''
    target = "class BiologyTools:"
    if "create_molecular_geometry" not in code and target in code:
        code = code.replace(target, mol_geom_code + "\n\n" + target, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("Added ChemistryTools.create_molecular_geometry!")


def add_info_wire():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    wire_code = '''    @staticmethod
    def get_flowchart_pins(
        center: tuple[float, float],
        width: float = 140.0,
        height: float = 70.0,
    ) -> dict[str, tuple[float, float]]:
        """Return 4 cardinal connection pin coordinates (top, bottom, left, right)."""
        cx, cy = center
        w2, h2 = width / 2.0, height / 2.0
        return {
            "top": (cx, cy - h2),
            "bottom": (cx, cy + h2),
            "left": (cx - w2, cy),
            "right": (cx + w2, cy),
        }

    @staticmethod
    def snap_to_nearest_pin(
        query_point: tuple[float, float],
        pins: dict[str, tuple[float, float]],
        tolerance: float = 28.0,
    ) -> Optional[tuple[str, tuple[float, float]]]:
        """Find the nearest cardinal pin within tolerance distance."""
        qx, qy = query_point
        best_name = None
        best_dist = float("inf")
        best_pos = None

        for name, (px, py) in pins.items():
            dist = math.hypot(qx - px, qy - py)
            if dist < best_dist and dist <= tolerance:
                best_dist = dist
                best_name = name
                best_pos = (px, py)

        if best_name is not None and best_pos is not None:
            return (best_name, best_pos)
        return None

    @staticmethod
    def create_manhattan_wire(
        start: tuple[float, float],
        end: tuple[float, float],
        color: QColor = QColor("#10b981"),
        line_width: float = 2.4,
        has_junction_dot: bool = False,
    ) -> list[QGraphicsItem]:
        """Creates an orthogonal 90-degree Manhattan circuit wire between two connection pins."""
        items: list[QGraphicsItem] = []
        x0, y0 = start
        x1, y1 = end
        pen = QPen(color, line_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

        path = QPainterPath()
        path.moveTo(x0, y0)
        mid_x = (x0 + x1) / 2.0
        path.lineTo(mid_x, y0)
        path.lineTo(mid_x, y1)
        path.lineTo(x1, y1)

        wire_item = QGraphicsPathItem(path)
        wire_item.setPen(pen)
        wire_item.setZValue(11.0)
        items.append(wire_item)

        if has_junction_dot:
            dot_radius = line_width * 1.8
            dot = QGraphicsEllipseItem(mid_x - dot_radius, y0 - dot_radius, dot_radius * 2, dot_radius * 2)
            dot.setBrush(QBrush(color))
            dot.setPen(Qt.PenStyle.NoPen)
            dot.setZValue(11.5)
            items.append(dot)

        return items

    @staticmethod
    def create_interactive_logic_gate(
        gate_type: str = "AND",
        center: tuple[float, float] = (960.0, 540.0),
        width: float = 130.0,
        height: float = 75.0,
        input_labels: tuple[str, ...] = ("A", "B"),
        output_label: str = "Y",
    ):
        """Create and return an InteractiveLogicGateItem."""
        from src.ui.interactive_logic_gate import InteractiveLogicGateItem
        return InteractiveLogicGateItem(
            gate_type=gate_type,
            center=center,
            width=width,
            height=height,
            input_labels=input_labels,
            output_label=output_label,
        )
'''
    target = "class StemToolDialog(QDialog):"
    if "create_interactive_logic_gate" not in code and target in code:
        code = code.replace(target, wire_code + "\n\n" + target, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("Added InformaticsTools wire and interactive gate methods!")


if __name__ == "__main__":
    add_math_tools()
    add_biology_dna()
    add_chem_methods()
    add_info_wire()
    print("All additions completed!")
