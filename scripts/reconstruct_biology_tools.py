"""
Script to reconstruct BiologyTools in src/ui/stem_tools.py with:
- create_callout_pin
- create_dna_double_helix
- create_punnett_square
- create_punnett_genetics_diagram
- create_trophic_pyramid
"""
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    start_str = "    @staticmethod\n    def create_dna_double_helix("
    end_str = "    @staticmethod\n    def create_trophic_pyramid("

    if start_str not in content or end_str not in content:
        print("ERROR: start_str or end_str not found")
        sys.exit(1)

    idx_start = content.index(start_str)
    idx_end = content.index(end_str)

    new_methods = '''    @staticmethod
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
        """
        Create vector DNA double helix diagram with 2 continuous backbone strands,
        split half-rungs with hydrogen bond gap (2 * rungs segments), and 5'/3' terminal labels.
        """
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

        # Strand 1 item
        s1_item = QGraphicsPathItem(strand1_path)
        s1_item.setPen(QPen(color1, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        s1_item.setZValue(12.0)
        items.append(s1_item)

        # Strand 2 item
        s2_item = QGraphicsPathItem(strand2_path)
        s2_item.setPen(QPen(color2, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        s2_item.setZValue(12.0)
        items.append(s2_item)

        # Rungs (hydrogen bonds)
        base_pairs = [("A", "T"), ("G", "C"), ("T", "A"), ("C", "G")]
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

                # Split rung in half for hydrogen bonds
                mid_x = (p1x + p2x) / 2.0
                mid_y = (p1y + p2y) / 2.0
                rx = p2x - p1x
                ry = p2y - p1y
                rlen = math.hypot(rx, ry)
                if rlen > 1e-4:
                    rux = rx / rlen
                    ruy = ry / rlen
                else:
                    rux, ruy = 1.0, 0.0

                gap = 6.0
                h1_end_x = mid_x - rux * (gap / 2.0)
                h1_end_y = mid_y - ruy * (gap / 2.0)
                h2_start_x = mid_x + rux * (gap / 2.0)
                h2_start_y = mid_y + ruy * (gap / 2.0)

                # Segment 1
                seg1 = QGraphicsLineItem(p1x, p1y, h1_end_x, h1_end_y)
                seg1.setPen(rung_pen1)
                seg1.setZValue(11.5)
                items.append(seg1)

                # Segment 2
                seg2 = QGraphicsLineItem(h2_start_x, h2_start_y, p2x, p2y)
                seg2.setPen(rung_pen2)
                seg2.setZValue(11.5)
                items.append(seg2)

        # Terminal labels: 5' and 3' at start and end
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

    @staticmethod
    def create_punnett_square(
        center: tuple[float, float],
        cell_size: float = 60.0,
        male_gametes: tuple[str, ...] = ("B", "b"),
        female_gametes: tuple[str, ...] = ("B", "b"),
        title: str = "Phép lai: Bb × Bb",
        color: QColor = QColor("#202124"),
    ) -> list[QGraphicsItem]:
        """
        Create Mendelian genetics 2x2 Punnett square table with gametes and offspring genotypes.
        """
        items: list[QGraphicsItem] = []
        cx, cy = center
        s = cell_size

        # Grid starts at (cx - 1.5*s, cy - 1.5*s) to accommodate 3x3 layout (headers + 2x2 cells)
        grid_x = cx - 1.5 * s
        grid_y = cy - 1.5 * s
        grid_w = 3 * s
        grid_h = 3 * s

        # Outer card background
        card_bg = QGraphicsRectItem(grid_x - 12.0, grid_y - 36.0, grid_w + 24.0, grid_h + 48.0)
        card_bg.setPen(QPen(QColor("#334155"), 1.5))
        card_bg.setBrush(QBrush(QColor("#0f172a")))
        card_bg.setZValue(10.0)
        items.append(card_bg)

        # Title text
        title_item = QGraphicsTextItem(title)
        title_item.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_item.setDefaultTextColor(QColor("#f8fafc"))
        title_item.setPos(grid_x, grid_y - 32.0)
        title_item.setZValue(11.0)
        items.append(title_item)

        # Header: Top-Left cell indicates female (row) vs male (column)
        tl_cell = QGraphicsRectItem(grid_x, grid_y, s, s)
        tl_cell.setPen(QPen(QColor("#475569"), 1.2))
        tl_cell.setBrush(QBrush(QColor("#1e293b")))
        tl_cell.setZValue(11.0)
        items.append(tl_cell)

        txt_hdr = QGraphicsTextItem("♀ \\ ♂")
        txt_hdr.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        txt_hdr.setDefaultTextColor(QColor("#94a3b8"))
        txt_hdr.setPos(grid_x + 4.0, grid_y + s / 2.0 - 12.0)
        txt_hdr.setZValue(11.5)
        items.append(txt_hdr)

        # Male Gametes (Columns 1 and 2)
        for col_idx, g_male in enumerate(male_gametes[:2]):
            col_x = grid_x + (col_idx + 1) * s
            txt = QGraphicsTextItem(g_male)
            txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            txt.setDefaultTextColor(QColor("#60a5fa"))
            txt.setPos(col_x + s / 2.0 - 8.0, grid_y + s / 2.0 - 12.0)
            txt.setZValue(11.5)
            items.append(txt)

        # Female Gametes (Rows 1 and 2)
        for row_idx, g_fem in enumerate(female_gametes[:2]):
            row_y = grid_y + (row_idx + 1) * s
            txt = QGraphicsTextItem(g_fem)
            txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            txt.setDefaultTextColor(QColor("#f9a8d4"))
            txt.setPos(grid_x + 8.0, row_y + s / 2.0 - 12.0)
            txt.setZValue(11.5)
            items.append(txt)

        # 2x2 Offspring Cells
        for r_i, g_f in enumerate(female_gametes[:2]):
            for c_i, g_m in enumerate(male_gametes[:2]):
                cx_cell = grid_x + (c_i + 1) * s
                cy_cell = grid_y + (r_i + 1) * s

                cell_rect = QGraphicsRectItem(cx_cell, cy_cell, s, s)
                cell_rect.setPen(QPen(QColor("#475569"), 1.2))
                cell_rect.setBrush(QBrush(QColor("#0f172a")))
                cell_rect.setZValue(11.0)
                items.append(cell_rect)

                # Genotype: Capital first if hetero
                alleles = sorted([g_f, g_m], key=lambda a: (a.islower(), a))
                geno = "".join(alleles)
                gt_txt = QGraphicsTextItem(geno)
                gt_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                gt_txt.setDefaultTextColor(QColor("#f8fafc"))
                gt_txt.setPos(cx_cell + s / 2.0 - 12.0, cy_cell + s / 2.0 - 12.0)
                gt_txt.setZValue(12.0)
                items.append(gt_txt)

        return items

    @staticmethod
    def create_punnett_genetics_diagram(
        center: tuple[float, float],
        width: float = 540.0,
        height: float = 320.0,
        preset: str = "MENDEL_PEAS_MONOHYBRID",
    ) -> list[QGraphicsItem]:
        """
        Creates a permanent vector diagram of Mendelian genetics and Punnett square,
        including 2x2 or 4x4 gamete grid, genotypic/phenotypic ratios, and Chi-Square goodness-of-fit stats.
        """
        items: list[QGraphicsItem] = []
        cx, cy = center
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#09101d")))
        bg.setPen(QPen(QColor("#1e293b"), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header
        header = QGraphicsTextItem("🧬 DI TRUYỀN HỌC MENDEL & KHUNG PUNNETT (GENETICS)")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setDefaultTextColor(QColor("#4ade80"))
        header.setPos(x_left + 14.0, y_top + 10.0)
        header.setZValue(11.0)
        items.append(header)

        # Preset info
        presets_data = {
            "MENDEL_PEAS_MONOHYBRID": {
                "title": "Lai 1 tính trạng: P: Aa × Aa (3 Hạt Vàng : 1 Hạt Xanh)",
                "mg": ["A", "a"],
                "fg": ["A", "a"],
                "cells": [("AA", "Vàng", "#eab308"), ("Aa", "Vàng", "#eab308"), ("Aa", "Vàng", "#eab308"), ("aa", "Xanh", "#22c55e")],
                "pheno_txt": "3 Vàng (75%) : 1 Xanh (25%)",
                "geno_txt": "1 AA : 2 Aa : 1 aa (25% : 50% : 25%)",
                "chi_txt": "χ² = 0.083 (df=1, p > 0.05) ✔ Chấp nhận Mendel",
            },
            "MENDEL_PEAS_DIHYBRID": {
                "title": "Lai 2 tính trạng: P: AaBb × AaBb (9:3:3:1)",
                "mg": ["AB", "Ab", "aB", "ab"],
                "fg": ["AB", "Ab", "aB", "ab"],
                "cells": [
                    ("AABB", "V-T", "#eab308"), ("AABb", "V-T", "#eab308"), ("AaBB", "V-T", "#eab308"), ("AaBb", "V-T", "#eab308"),
                    ("AABb", "V-T", "#eab308"), ("AAbb", "V-N", "#ca8a04"), ("AaBb", "V-T", "#eab308"), ("Aabb", "V-N", "#ca8a04"),
                    ("AaBB", "V-T", "#eab308"), ("AaBb", "V-T", "#eab308"), ("aaBB", "X-T", "#22c55e"), ("aaBb", "X-T", "#22c55e"),
                    ("AaBb", "V-T", "#eab308"), ("Aabb", "V-N", "#ca8a04"), ("aaBb", "X-T", "#22c55e"), ("aabb", "X-N", "#15803d"),
                ],
                "pheno_txt": "9 Vàng-Trơn : 3 Vàng-Nhăn : 3 Xanh-Trơn : 1 Xanh-Nhăn",
                "geno_txt": "16 tổ hợp (9 kiểu gen độc lập)",
                "chi_txt": "χ² = 0.470 (df=3, p > 0.05) ✔ Chấp nhận Mendel",
            },
            "BLOOD_TYPE_ABO": {
                "title": "Nhóm máu ABO: P: Iᴬ i × Iᴮ i (Đồng trội & Đa alen)",
                "mg": ["Iᴬ", "i"],
                "fg": ["Iᴮ", "i"],
                "cells": [("IᴬIᴮ", "Nhóm AB", "#ec4899"), ("Iᴮi", "Nhóm B", "#f59e0b"), ("Iᴬi", "Nhóm A", "#38bdf8"), ("ii", "Nhóm O", "#10b981")],
                "pheno_txt": "1 AB : 1 A : 1 B : 1 O (25% mỗi loại)",
                "geno_txt": "1 IᴬIᴮ : 1 Iᴬi : 1 Iᴮi : 1 ii",
                "chi_txt": "χ² = 0.320 (df=3, p > 0.05) ✔ Phù hợp",
            },
            "SEX_LINKED_COLORBLIND": {
                "title": "Liên kết giới tính X: P: Xᶜ Xᶜ × Xᶜ Y (Bệnh mù màu)",
                "mg": ["Xᶜ", "Y"],
                "fg": ["Xᶜ", "Xᶜ"],
                "cells": [("XᶜXᶜ", "Gái thường", "#38bdf8"), ("XᶜY", "Trai thường", "#0284c7"), ("XᶜXᶜ", "Gái mang gen", "#a855f7"), ("XᶜY", "Trai mù màu", "#ef4444")],
                "pheno_txt": "1 Gái thường : 1 Gái mang gen : 1 Trai thường : 1 Trai bệnh",
                "geno_txt": "Phân ly 1:1:1:1 theo giới tính",
                "chi_txt": "χ² = 0.160 (df=3, p > 0.05) ✔ Phù hợp",
            },
        }

        cfg = presets_data.get(preset, presets_data["MENDEL_PEAS_MONOHYBRID"])

        # Subtitle
        sub = QGraphicsTextItem(cfg["title"])
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x_left + 16.0, y_top + 34.0)
        sub.setZValue(11.0)
        items.append(sub)

        # 3. Grid Pane (Left)
        gw = 290.0
        gh = 240.0
        gx = x_left + 14.0
        gy = y_top + 58.0
        g_rect = QGraphicsRectItem(gx, gy, gw, gh)
        g_rect.setBrush(QBrush(QColor("#0d1322")))
        g_rect.setPen(QPen(QColor("#1e293b"), 1.2))
        g_rect.setZValue(10.5)
        items.append(g_rect)

        # Draw Cells
        mg_list = cfg["mg"]
        fg_list = cfg["fg"]
        n_c = len(mg_list)
        n_r = len(fg_list)

        hw = 36.0
        hh = 24.0
        cw = (gw - hw - 14.0) / n_c
        ch = (gh - hh - 14.0) / n_r

        # Male Gametes (Top)
        for c_idx, mg in enumerate(mg_list):
            mx = gx + hw + 8.0 + c_idx * cw
            my = gy + 6.0
            mtxt = QGraphicsTextItem(f"♂ {mg}")
            mtxt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            mtxt.setDefaultTextColor(QColor("#38bdf8"))
            mtxt.setPos(mx + cw / 2.0 - 12.0, my)
            mtxt.setZValue(11.2)
            items.append(mtxt)

        # Female Gametes (Left)
        for r_idx, fg_val in enumerate(fg_list):
            fx = gx + 6.0
            fy = gy + hh + 6.0 + r_idx * ch
            ftxt = QGraphicsTextItem(f"♀ {fg_val}")
            ftxt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            ftxt.setDefaultTextColor(QColor("#ec4899"))
            ftxt.setPos(fx, fy + ch / 2.0 - 10.0)
            ftxt.setZValue(11.2)
            items.append(ftxt)

        # Grid Offspring Cells
        cell_idx = 0
        for r_idx in range(n_r):
            for c_idx in range(n_c):
                if cell_idx < len(cfg["cells"]):
                    geno, pheno, col_hex = cfg["cells"][cell_idx]
                else:
                    geno, pheno, col_hex = "Aa", "F₁", "#eab308"
                cell_idx += 1

                bx = gx + hw + 8.0 + c_idx * cw
                by = gy + hh + 6.0 + r_idx * ch

                cbox = QGraphicsRectItem(bx + 1.0, by + 1.0, cw - 2.0, ch - 2.0)
                badge_c = QColor(col_hex)
                cbox.setBrush(QBrush(QColor(badge_c.red(), badge_c.green(), badge_c.blue(), 30)))
                cbox.setPen(QPen(badge_c, 1.0))
                cbox.setZValue(11.0)
                items.append(cbox)

                gt = QGraphicsTextItem(geno)
                gt.setFont(QFont("Consolas", 8 if n_r <= 2 else 7, QFont.Weight.Bold))
                gt.setDefaultTextColor(QColor("#f8fafc"))
                gt.setPos(bx + cw / 2.0 - 12.0, by + (ch / 2.0 - 16.0 if n_r <= 2 else ch / 2.0 - 12.0))
                gt.setZValue(11.4)
                items.append(gt)

                pt = QGraphicsTextItem(pheno)
                pt.setFont(QFont("Segoe UI", 7 if n_r <= 2 else 6))
                pt.setDefaultTextColor(badge_c.lighter(120))
                pt.setPos(bx + cw / 2.0 - 12.0, by + ch / 2.0)
                pt.setZValue(11.4)
                items.append(pt)

        # 4. Right Statistics Panel
        rw = width - gw - 38.0
        rh = gh
        rx = x_left + gw + 24.0
        ry = gy
        r_rect = QGraphicsRectItem(rx, ry, rw, rh)
        r_rect.setBrush(QBrush(QColor("#0d1322")))
        r_rect.setPen(QPen(QColor("#1e293b"), 1.2))
        r_rect.setZValue(10.5)
        items.append(r_rect)

        rtitle = QGraphicsTextItem("TỈ LỆ PHÂN LY & KIỂM ĐỊNH χ²")
        rtitle.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        rtitle.setDefaultTextColor(QColor("#4ade80"))
        rtitle.setPos(rx + 8.0, ry + 8.0)
        rtitle.setZValue(11.0)
        items.append(rtitle)

        stat_lines = [
            ("TỈ LỆ KIỂU HÌNH:", QColor("#eab308"), True),
            (cfg["pheno_txt"], QColor("#f8fafc"), False),
            ("", QColor("#ffffff"), False),
            ("TỈ LỆ KIỂU GEN:", QColor("#38bdf8"), True),
            (cfg["geno_txt"], QColor("#cbd5e1"), False),
            ("", QColor("#ffffff"), False),
            ("KIỂM ĐỊNH CHI-BÌNH PHƯƠNG:", QColor("#4ade80"), True),
            (cfg["chi_txt"], QColor("#22c55e"), False),
        ]

        curr_sy = ry + 28.0
        for s_text, col, is_b in stat_lines:
            if s_text:
                sit = QGraphicsTextItem(s_text)
                sit.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_b else QFont.Weight.Normal))
                sit.setDefaultTextColor(col)
                sit.setPos(rx + 8.0, curr_sy)
                sit.setZValue(11.0)
                items.append(sit)
                curr_sy += 16.0
            else:
                curr_sy += 4.0

        return items

    '''

    new_content = content[:idx_start] + new_methods + content[idx_end:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully reconstructed BiologyTools!")

if __name__ == "__main__":
    main()
