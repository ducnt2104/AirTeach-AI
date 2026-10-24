with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = "        angle = math.atan2(dy, dx)"
end_marker = """    @staticmethod
    def create_trophic_pyramid("""

if start_marker in text and end_marker in text:
    part1 = text[:text.index(start_marker)]
    part2 = text[text.index(end_marker):]

    new_method = """
    @staticmethod
    def create_punnett_genetics_diagram(
        center: tuple[float, float],
        width: float = 540.0,
        height: float = 320.0,
        preset: str = "MENDEL_PEAS_MONOHYBRID",
    ) -> list[QGraphicsItem]:
        \"\"\"
        Creates a permanent vector diagram of Mendelian genetics and Punnett square,
        including 2x2 or 4x4 gamete grid, genotypic/phenotypic ratios, and Chi-Square goodness-of-fit stats.
        \"\"\"
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

"""
    with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
        f.write(part1 + new_method + part2)
    print("Successfully cleanly replaced dead DNA chunk with create_punnett_genetics_diagram!")
else:
    print("Markers not found!")
