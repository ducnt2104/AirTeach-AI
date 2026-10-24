"""
Script to add InformaticsTools.create_automata_diagram to src/ui/stem_tools.py.
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = """        return items


# ─── STEM Subject Dialog for Interactive Tool Parameter Entry ─────────────────"""

    method_code = '''        return items

    @staticmethod
    def create_automata_diagram(
        center: tuple[float, float],
        width: float = 540.0,
        height: float = 320.0,
        preset: str = "EVEN_ZEROS",
        input_string: str = "1001100",
    ) -> list[QGraphicsItem]:
        """
        Creates a permanent vector diagram of a Finite State Machine / Automaton (DFA)
        complete with state transition graph, tape reading head, and language acceptance verdict.
        """
        items: list[QGraphicsItem] = []
        cx, cy = center
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#0a0f1d")))
        bg.setPen(QPen(QColor("#1e293b"), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header
        header = QGraphicsTextItem("🤖 MÁY TRẠNG THÁI HỮU HẠN & DFA / NFA (AUTOMATA THEORY)")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setDefaultTextColor(QColor("#38bdf8"))
        header.setPos(x_left + 14.0, y_top + 10.0)
        header.setZValue(11.0)
        items.append(header)

        # 3. Model Presets Data
        preset_info = {
            "EVEN_ZEROS": {
                "title": "Số bit 0 là Chẵn: L = {w ∈ {0,1}* | |w|₀ chẵn}",
                "states": [("q₀ (Chẵn)", 100.0, 110.0, True, True), ("q₁ (Lẻ)", 260.0, 110.0, False, False)],
                "transitions": [
                    ("q₀", "0", "q₁", -22.0),
                    ("q₁", "0", "q₀", -22.0),
                    ("q₀", "1", "q₀", 0.0),
                    ("q₁", "1", "q₁", 0.0),
                ],
            },
            "MOD_3": {
                "title": "Chia hết cho 3: L = {w ∈ {0,1}* | val(w) mod 3 = 0}",
                "states": [
                    ("q₀ (r=0)", 80.0, 110.0, True, True),
                    ("q₁ (r=1)", 200.0, 65.0, False, False),
                    ("q₂ (r=2)", 200.0, 155.0, False, False),
                ],
                "transitions": [
                    ("q₀", "0", "q₀", 0.0),
                    ("q₀", "1", "q₁", 0.0),
                    ("q₁", "0", "q₂", 0.0),
                    ("q₁", "1", "q₀", -16.0),
                    ("q₂", "0", "q₁", 16.0),
                    ("q₂", "1", "q₂", 0.0),
                ],
            },
            "ENDS_WITH_ABB": {
                "title": "Kết thúc bằng 'abb': L = {w ∈ {a,b}* | w kết thúc 'abb'}",
                "states": [
                    ("q₀", 60.0, 110.0, True, False),
                    ("q₁ (a)", 135.0, 110.0, False, False),
                    ("q₂ (ab)", 210.0, 110.0, False, False),
                    ("q₃ (abb)", 285.0, 110.0, False, True),
                ],
                "transitions": [
                    ("q₀", "a", "q₁", 0.0),
                    ("q₀", "b", "q₀", 0.0),
                    ("q₁", "a", "q₁", 0.0),
                    ("q₁", "b", "q₂", 0.0),
                    ("q₂", "a", "q₁", -22.0),
                    ("q₂", "b", "q₃", 0.0),
                    ("q₃", "a", "q₁", -40.0),
                    ("q₃", "b", "q₀", -55.0),
                ],
            },
            "ODD_PARITY": {
                "title": "Kiểm tra Parity Lẻ: L = {w ∈ {0,1}* | |w|₁ lẻ}",
                "states": [("q₀ (Chẵn 1)", 100.0, 110.0, True, False), ("q₁ (Lẻ 1)", 260.0, 110.0, False, True)],
                "transitions": [
                    ("q₀", "1", "q₁", -22.0),
                    ("q₁", "1", "q₀", -22.0),
                    ("q₀", "0", "q₀", 0.0),
                    ("q₁", "0", "q₁", 0.0),
                ],
            },
        }

        cfg = preset_info.get(preset, preset_info["EVEN_ZEROS"])

        # Subtitle
        sub = QGraphicsTextItem(cfg["title"])
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x_left + 16.0, y_top + 34.0)
        sub.setZValue(11.0)
        items.append(sub)

        # Graph Panel
        gw = 360.0
        gh = 175.0
        gx = x_left + 14.0
        gy = y_top + 58.0
        g_panel = QGraphicsRectItem(gx, gy, gw, gh)
        g_panel.setBrush(QBrush(QColor("#0d1322")))
        g_panel.setPen(QPen(QColor("#1e293b"), 1.2))
        g_panel.setZValue(10.5)
        items.append(g_panel)

        # Draw States
        state_coords = {}
        for sname, sx, sy, is_start, is_acc in cfg["states"]:
            px = gx + sx
            py = gy + sy
            sid = sname.split()[0]
            state_coords[sid] = (px, py)
            radius = 20.0

            # Start Arrow
            if is_start:
                s_line = QGraphicsLineItem(px - radius - 20.0, py, px - radius - 2.0, py)
                s_line.setPen(QPen(QColor("#f59e0b"), 1.8))
                s_line.setZValue(11.2)
                items.append(s_line)

            # Node Circle
            s_circ = QGraphicsEllipseItem(px - radius, py - radius, radius * 2.0, radius * 2.0)
            s_circ.setBrush(QBrush(QColor("#1e293b")))
            s_circ.setPen(QPen(QColor("#64748b"), 1.4))
            s_circ.setZValue(11.0)
            items.append(s_circ)

            # Accept State Inner Circle
            if is_acc:
                acc_circ = QGraphicsEllipseItem(px - radius + 3.5, py - radius + 3.5, (radius - 3.5) * 2.0, (radius - 3.5) * 2.0)
                acc_circ.setBrush(Qt.BrushStyle.NoBrush)
                acc_circ.setPen(QPen(QColor("#64748b"), 1.2))
                acc_circ.setZValue(11.2)
                items.append(acc_circ)

            # Node Label
            lbl = QGraphicsTextItem(sid)
            lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            lbl.setDefaultTextColor(QColor("#f8fafc"))
            lbl.setPos(px - 12.0, py - 12.0)
            lbl.setZValue(11.4)
            items.append(lbl)

        # Draw Transitions
        for src, sym, tgt, curv in cfg["transitions"]:
            if src in state_coords and tgt in state_coords:
                spx, spy = state_coords[src]
                tpx, tpy = state_coords[tgt]
                radius = 20.0

                if src == tgt:
                    # Self-loop
                    lr = 13.0
                    lpath = QPainterPath()
                    lpath.arcMoveTo(spx - lr, spy - radius - lr * 2.0 + 3.0, lr * 2.0, lr * 2.0, -40)
                    lpath.arcTo(spx - lr, spy - radius - lr * 2.0 + 3.0, lr * 2.0, lr * 2.0, -40, 260)
                    lp_item = QGraphicsPathItem(lpath)
                    lp_item.setPen(QPen(QColor("#64748b"), 1.2))
                    lp_item.setZValue(10.8)
                    items.append(lp_item)

                    ltxt = QGraphicsTextItem(sym)
                    ltxt.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                    ltxt.setDefaultTextColor(QColor("#cbd5e1"))
                    ltxt.setPos(spx - 6.0, spy - radius - lr * 2.0 - 10.0)
                    ltxt.setZValue(11.0)
                    items.append(ltxt)
                else:
                    dx = tpx - spx
                    dy = tpy - spy
                    dist = math.hypot(dx, dy)
                    if dist > 1e-3:
                        ux = dx / dist
                        uy = dy / dist
                        nx = -uy
                        ny = ux

                        start_p = QPointF(spx + ux * radius, spy + uy * radius)
                        end_p = QPointF(tpx - ux * radius, tpy - uy * radius)
                        ctrl_p = QPointF((spx + tpx) / 2.0 + nx * curv, (spy + tpy) / 2.0 + ny * curv)

                        epath = QPainterPath()
                        epath.moveTo(start_p)
                        epath.quadTo(ctrl_p, end_p)

                        ep_item = QGraphicsPathItem(epath)
                        ep_item.setPen(QPen(QColor("#64748b"), 1.2))
                        ep_item.setZValue(10.8)
                        items.append(ep_item)

                        # Label
                        mid_x = 0.25 * start_p.x() + 0.5 * ctrl_p.x() + 0.25 * end_p.x() + nx * 6.0
                        mid_y = 0.25 * start_p.y() + 0.5 * ctrl_p.y() + 0.25 * end_p.y() + ny * 6.0
                        etxt = QGraphicsTextItem(sym)
                        etxt.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                        etxt.setDefaultTextColor(QColor("#cbd5e1"))
                        etxt.setPos(mid_x - 6.0, mid_y - 8.0)
                        etxt.setZValue(11.0)
                        items.append(etxt)

        # Right Panel: 5-Tuple Formal Definition
        rw = 140.0
        rh = 175.0
        rx = x_left + width - rw - 14.0
        ry = y_top + 58.0
        r_panel = QGraphicsRectItem(rx, ry, rw, rh)
        r_panel.setBrush(QBrush(QColor("#0d1322")))
        r_panel.setPen(QPen(QColor("#1e293b"), 1.2))
        r_panel.setZValue(10.5)
        items.append(r_panel)

        rtitle = QGraphicsTextItem("ĐỊNH NGHĨA 5-TUPLE")
        rtitle.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        rtitle.setDefaultTextColor(QColor("#38bdf8"))
        rtitle.setPos(rx + 8.0, ry + 8.0)
        rtitle.setZValue(11.0)
        items.append(rtitle)

        formal_lines = [
            "M = ⟨Q, Σ, δ, q₀, F⟩",
            f"• |Q| = {len(cfg['states'])} trạng thái",
            "• Σ = {0, 1}" if "0" in cfg["title"] or "Parity" in cfg["title"] or "mod" in cfg["title"] else "• Σ = {a, b}",
            "• Trạng thái đầu: q₀",
            "• Trạng thái kết thúc: F",
        ]
        fy = ry + 28.0
        for fl in formal_lines:
            fitem = QGraphicsTextItem(fl)
            fitem.setFont(QFont("Consolas", 7))
            fitem.setDefaultTextColor(QColor("#cbd5e1"))
            fitem.setPos(rx + 8.0, fy)
            fitem.setZValue(11.0)
            items.append(fitem)
            fy += 16.0

        # Bottom Tape Bar
        tape_y = y_top + height - 68.0
        tbg = QGraphicsRectItem(x_left + 14.0, tape_y, width - 28.0, 48.0)
        tbg.setBrush(QBrush(QColor("#0d1322")))
        tbg.setPen(QPen(QColor("#1e293b"), 1.2))
        tbg.setZValue(10.5)
        items.append(tbg)

        ttitle = QGraphicsTextItem(f"BĂNG NHẬP (INPUT TAPE): \\"{input_string}\\"")
        ttitle.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        ttitle.setDefaultTextColor(QColor("#94a3b8"))
        ttitle.setPos(x_left + 22.0, tape_y + 4.0)
        ttitle.setZValue(11.0)
        items.append(ttitle)

        cell_w = 22.0
        cell_h = 24.0
        for i, ch in enumerate(input_string[:16]):
            cx_cell = x_left + 24.0 + i * (cell_w + 2.0)
            cy_cell = tape_y + 20.0
            cbox = QGraphicsRectItem(cx_cell, cy_cell, cell_w, cell_h)
            cbox.setBrush(QBrush(QColor("#1e293b")))
            cbox.setPen(QPen(QColor("#334155"), 0.8))
            cbox.setZValue(11.0)
            items.append(cbox)

            ctxt = QGraphicsTextItem(ch)
            ctxt.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            ctxt.setDefaultTextColor(QColor("#f8fafc"))
            ctxt.setPos(cx_cell + 3.0, cy_cell + 1.0)
            ctxt.setZValue(11.2)
            items.append(ctxt)

        return items


# ─── STEM Subject Dialog for Interactive Tool Parameter Entry ─────────────────'''

    if target not in content:
        print("ERROR: target not found!")
        sys.exit(1)

    content = content.replace(target, method_code, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully added InformaticsTools.create_automata_diagram!")

if __name__ == "__main__":
    main()
