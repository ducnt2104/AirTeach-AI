"""
Script to add InformaticsTools.create_sorting_diagram to src/ui/stem_tools.py.
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
    def create_sorting_diagram(
        center: tuple[float, float],
        width: float = 540.0,
        height: float = 320.0,
        algorithm: str = "BUBBLE",
        array: Optional[list[int]] = None,
        comparisons: int = 0,
        swaps: int = 0,
    ) -> list[QGraphicsItem]:
        """
        Creates a permanent vector diagram of a Sorting Algorithm state,
        including bar chart visualization, Big-O theoretical complexity table,
        and real-time operation counters.
        """
        items: list[QGraphicsItem] = []
        cx, cy = center
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        if array is None:
            array = [15, 45, 20, 80, 55, 30, 95, 10, 65, 40]

        # 1. Main Background Card
        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#090e1a")))
        bg.setPen(QPen(QColor("#1e293b"), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header
        algo_names = {
            "BUBBLE": "Bubble Sort (Sắp xếp Nổi bọt)",
            "SELECTION": "Selection Sort (Sắp xếp Chọn trực tiếp)",
            "INSERTION": "Insertion Sort (Sắp xếp Chèn trực tiếp)",
            "MERGE": "Merge Sort (Sắp xếp Trộn chia để trị)",
            "QUICK": "Quick Sort (Sắp xếp Phân đoạn nhanh)",
            "HEAP": "Heap Sort (Sắp xếp Vun đống cực đại)",
        }
        title_str = algo_names.get(algorithm, f"Sorting Algorithm ({algorithm})")
        header = QGraphicsTextItem(f"📊 {title_str.upper()} & ĐỘ PHỨC TẠP BIG-O")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setDefaultTextColor(QColor("#38bdf8"))
        header.setPos(x_left + 14.0, y_top + 10.0)
        header.setZValue(11.0)
        items.append(header)

        # 3. Bar Chart Pane (Left)
        cw = 330.0
        ch = 250.0
        c_rect = QGraphicsRectItem(x_left + 14.0, y_top + 46.0, cw, ch)
        c_rect.setBrush(QBrush(QColor("#0d1322")))
        c_rect.setPen(QPen(QColor("#1e293b"), 1.2))
        c_rect.setZValue(10.5)
        items.append(c_rect)

        # Draw Bars
        max_v = max(max(array), 1)
        n = len(array)
        avail_w = cw - 24.0
        bar_w = max(4.0, avail_w / n - 3.0)
        avail_h = ch - 50.0

        for i, val in enumerate(array):
            bx = x_left + 26.0 + i * (bar_w + 3.0)
            bh = (val / max_v) * avail_h
            by = y_top + 46.0 + ch - 26.0 - bh

            bar_item = QGraphicsRectItem(bx, by, bar_w, bh)
            bar_item.setBrush(QBrush(QColor("#0284c7")))
            bar_item.setPen(QPen(QColor("#38bdf8"), 1.0))
            bar_item.setZValue(11.0)
            items.append(bar_item)

            if n <= 16:
                v_txt = QGraphicsTextItem(str(val))
                v_txt.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
                v_txt.setDefaultTextColor(QColor("#f8fafc"))
                v_txt.setPos(bx - 3.0, by - 14.0)
                v_txt.setZValue(11.2)
                items.append(v_txt)

        # Chart Footer Legend
        legend_txt = QGraphicsTextItem(f"Mảng n = {n} phần tử | Thao tác so sánh (C) = {comparisons} | Hoán đổi (S) = {swaps}")
        legend_txt.setFont(QFont("Segoe UI", 7))
        legend_txt.setDefaultTextColor(QColor("#94a3b8"))
        legend_txt.setPos(x_left + 22.0, y_top + 46.0 + ch - 20.0)
        legend_txt.setZValue(11.0)
        items.append(legend_txt)

        # 4. Big-O Complexity Table (Right)
        rw = width - cw - 42.0
        rh = ch
        rx = x_left + cw + 28.0
        ry = y_top + 46.0
        r_rect = QGraphicsRectItem(rx, ry, rw, rh)
        r_rect.setBrush(QBrush(QColor("#0d1322")))
        r_rect.setPen(QPen(QColor("#1e293b"), 1.2))
        r_rect.setZValue(10.5)
        items.append(r_rect)

        rtitle = QGraphicsTextItem("BẢNG PHÂN TÍCH BIG-O")
        rtitle.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        rtitle.setDefaultTextColor(QColor("#38bdf8"))
        rtitle.setPos(rx + 8.0, ry + 8.0)
        rtitle.setZValue(11.0)
        items.append(rtitle)

        big_o_info = {
            "BUBBLE": [("Best", "O(n)"), ("Average", "O(n²)"), ("Worst", "O(n²)"), ("Space", "O(1)"), ("Stable", "Có")],
            "SELECTION": [("Best", "O(n²)"), ("Average", "O(n²)"), ("Worst", "O(n²)"), ("Space", "O(1)"), ("Stable", "Không")],
            "INSERTION": [("Best", "O(n)"), ("Average", "O(n²)"), ("Worst", "O(n²)"), ("Space", "O(1)"), ("Stable", "Có")],
            "MERGE": [("Best", "O(n log n)"), ("Average", "O(n log n)"), ("Worst", "O(n log n)"), ("Space", "O(n)"), ("Stable", "Có")],
            "QUICK": [("Best", "O(n log n)"), ("Average", "O(n log n)"), ("Worst", "O(n²)"), ("Space", "O(log n)"), ("Stable", "Không")],
            "HEAP": [("Best", "O(n log n)"), ("Average", "O(n log n)"), ("Worst", "O(n log n)"), ("Space", "O(1)"), ("Stable", "Không")],
        }

        specs = big_o_info.get(algorithm, big_o_info["BUBBLE"])
        ty = ry + 32.0
        for s_lbl, s_val in specs:
            lit = QGraphicsTextItem(f"• {s_lbl}:")
            lit.setFont(QFont("Segoe UI", 7))
            lit.setDefaultTextColor(QColor("#cbd5e1"))
            lit.setPos(rx + 8.0, ty)
            lit.setZValue(11.0)
            items.append(lit)

            vit = QGraphicsTextItem(s_val)
            vit.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            vit.setDefaultTextColor(QColor("#facc15") if "O(n²)" in s_val else QColor("#4ade80") if "log" in s_val or s_val == "O(n)" else QColor("#f8fafc"))
            vit.setPos(rx + rw - 75.0, ty)
            vit.setZValue(11.0)
            items.append(vit)

            ty += 20.0

        # Divider
        div_line = QGraphicsLineItem(rx + 8.0, ty + 4.0, rx + rw - 8.0, ty + 4.0)
        div_line.setPen(QPen(QColor("#1e293b"), 1.0))
        div_line.setZValue(11.0)
        items.append(div_line)
        ty += 12.0

        # Operation counts in right panel
        otxt = QGraphicsTextItem(f"So sánh: {comparisons}\nHoán đổi: {swaps}")
        otxt.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        otxt.setDefaultTextColor(QColor("#38bdf8"))
        otxt.setPos(rx + 10.0, ty)
        otxt.setZValue(11.0)
        items.append(otxt)

        return items


# ─── STEM Subject Dialog for Interactive Tool Parameter Entry ─────────────────'''

    if target not in content:
        print("ERROR: target not found!")
        sys.exit(1)

    content = content.replace(target, method_code, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully added InformaticsTools.create_sorting_diagram!")

if __name__ == "__main__":
    main()
