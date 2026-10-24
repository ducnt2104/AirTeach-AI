"""
Script to cleanly insert InformaticsTools.create_binary_tree_diagram into src/ui/stem_tools.py.
"""
import py_compile
import sys

def main():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "def create_binary_tree_diagram(" in content:
        print("create_binary_tree_diagram already in stem_tools.py")
        return

    method_code = '''    @staticmethod
    def create_binary_tree_diagram(
        center: tuple[float, float] = (400.0, 300.0),
        preset_code: str = "BST_BALANCED_SEARCH",
        traversal_mode: str = "IN_ORDER",
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """
        Create a styled vector binary tree diagram with node hierarchy,
        traversal sequence, and Huffman prefix codes / search complexity analytics.
        """
        items: list[QGraphicsItem] = []
        cx, cy = center
        diag_w = 580.0
        diag_h = 350.0
        x_left = cx - diag_w / 2.0
        y_top = cy - diag_h / 2.0

        from src.ui.interactive_binary_tree import PRESETS, TreeNode

        preset = PRESETS.get(preset_code, PRESETS["BST_BALANCED_SEARCH"])
        title_str = preset.title_vi

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x_left, y_top, diag_w, diag_h)
        bg_card.setBrush(QBrush(QColor("#0f172a")))
        bg_card.setPen(QPen(pen_color or QColor("#10b981"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header Banner
        banner_txt = f"CẤU TRÚC DỮ LIỆU: CÂY NHỊ PHÂN & MÃ HÓA — {title_str}"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#f8fafc"))
        banner.setPos(x_left + 14.0, y_top + 10.0)
        banner.setZValue(10.5)
        items.append(banner)

        # 3. Tree Viewport Canvas (Left Side)
        canvas_x = x_left + 14.0
        canvas_y = y_top + 38.0
        canvas_w = 320.0
        canvas_h = 296.0

        canvas_bg = QGraphicsRectItem(canvas_x, canvas_y, canvas_w, canvas_h)
        canvas_bg.setBrush(QBrush(QColor("#090d16")))
        canvas_bg.setPen(QPen(QColor("#1e293b"), 1.2))
        canvas_bg.setZValue(10.2)
        items.append(canvas_bg)

        # Build tree representation
        from src.ui.interactive_binary_tree import InteractiveBinaryTreeItem
        temp_sim = InteractiveBinaryTreeItem()
        temp_sim.load_preset(preset_code)

        # Render Nodes and Branches into canvas
        nodes = temp_sim._all_nodes
        if nodes:
            # Map temp_sim coordinates into this diagram canvas
            orig_c = temp_sim._btn_rects["canvas"]
            orig_w = orig_c.width()
            orig_h = orig_c.height()

            scale_x = (canvas_w - 32.0) / orig_w
            scale_y = (canvas_h - 32.0) / orig_h

            def _map_pt(ox: float, oy: float) -> tuple[float, float]:
                rel_x = (ox - orig_c.left()) * scale_x
                rel_y = (oy - orig_c.top()) * scale_y
                return (canvas_x + 16.0 + rel_x, canvas_y + 16.0 + rel_y)

            # Draw branch edges
            for n in nodes:
                if n.left:
                    p1 = _map_pt(n.x, n.y)
                    p2 = _map_pt(n.left.x, n.left.y)
                    edge = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                    edge.setPen(QPen(QColor("#475569"), 1.4))
                    edge.setZValue(10.3)
                    items.append(edge)

                    if preset.tree_type == "HUFFMAN":
                        lbl_0 = QGraphicsTextItem("0")
                        lbl_0.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
                        lbl_0.setDefaultTextColor(QColor("#34d399"))
                        lbl_0.setPos((p1[0] + p2[0]) / 2.0 - 6.0, (p1[1] + p2[1]) / 2.0 - 8.0)
                        lbl_0.setZValue(10.4)
                        items.append(lbl_0)

                if n.right:
                    p1 = _map_pt(n.x, n.y)
                    p2 = _map_pt(n.right.x, n.right.y)
                    edge = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                    edge.setPen(QPen(QColor("#475569"), 1.4))
                    edge.setZValue(10.3)
                    items.append(edge)

                    if preset.tree_type == "HUFFMAN":
                        lbl_1 = QGraphicsTextItem("1")
                        lbl_1.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
                        lbl_1.setDefaultTextColor(QColor("#38bdf8"))
                        lbl_1.setPos((p1[0] + p2[0]) / 2.0 - 6.0, (p1[1] + p2[1]) / 2.0 - 8.0)
                        lbl_1.setZValue(10.4)
                        items.append(lbl_1)

            # Draw nodes
            r_node = 14.0
            for n in nodes:
                nx, ny = _map_pt(n.x, n.y)
                circ = QGraphicsEllipseItem(nx - r_node, ny - r_node, 2 * r_node, 2 * r_node)
                if n.is_operator:
                    circ.setBrush(QBrush(QColor("#9333ea")))
                    circ.setPen(QPen(QColor("#d8b4fe"), 1.4))
                else:
                    circ.setBrush(QBrush(QColor("#1e293b")))
                    circ.setPen(QPen(QColor("#10b981"), 1.4))
                circ.setZValue(10.5)
                items.append(circ)

                txt = QGraphicsTextItem(n.val or str(n.key))
                txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                txt.setDefaultTextColor(QColor("#ffffff"))
                txt.setPos(nx - 8.0, ny - 10.0)
                txt.setZValue(10.6)
                items.append(txt)

        # 4. Analytics & Algorithm Theory Card (Right Side)
        card_x = canvas_x + canvas_w + 12.0
        card_y = canvas_y
        card_w = diag_w - canvas_w - 40.0
        card_h = canvas_h

        info_card = QGraphicsRectItem(card_x, card_y, card_w, card_h)
        info_card.setBrush(QBrush(QColor("#090d16")))
        info_card.setPen(QPen(QColor("#1e293b"), 1.2))
        info_card.setZValue(10.2)
        items.append(info_card)

        # Collect traversal order
        seq: list[TreeNode] = []
        if temp_sim._root:
            if traversal_mode == "PRE_ORDER":
                temp_sim._collect_preorder(temp_sim._root, seq)
            elif traversal_mode == "POST_ORDER":
                temp_sim._collect_postorder(temp_sim._root, seq)
            elif traversal_mode == "LEVEL_ORDER":
                temp_sim._collect_levelorder(temp_sim._root, seq)
            else:
                temp_sim._collect_inorder(temp_sim._root, seq)

        seq_str = " ➔ ".join(n.val for n in seq) if seq else "Trống"
        if len(seq_str) > 34:
            seq_str = seq_str[:31] + "..."

        is_avl = temp_sim.is_avl_balanced
        lines = [
            ("LÝ THUYẾT CÂY & ĐỘ PHỨC TẠP", QColor("#f8fafc"), True, 9),
            (f"• Số nút: N = {temp_sim.node_count}", QColor("#38bdf8"), False, 8),
            (f"• Chiều cao cây: h = {temp_sim.tree_height}", QColor("#cbd5e1"), False, 8),
            (f"• Cân bằng AVL: {'Đạt ✔' if is_avl else 'Không ✘'}", QColor("#22c55e" if is_avl else "#f59e0b"), True, 8),
            (f"• Thứ tự duyệt: {traversal_mode}", QColor("#34d399"), True, 8),
            (f"  {seq_str}", QColor("#e2e8f0"), False, 7),
        ]

        if preset.tree_type == "HUFFMAN":
            hm = temp_sim.calculate_huffman_metrics()
            lines.extend([
                ("MÃ HÓA HUFFMAN TỐI ƯU:", QColor("#fbbf24"), True, 8),
                (f"• Entropy H(X): {hm['entropy']:.3f} bit/ký tự", QColor("#cbd5e1"), False, 7),
                (f"• Chiều dài L̄: {hm['avg_len']:.3f} bit", QColor("#cbd5e1"), False, 7),
                (f"• Hiệu suất η: {hm['efficiency']:.1f}%", QColor("#34d399"), True, 7),
                (f"• Tỷ lệ nén: {hm['compression_ratio']:.1f}% vs ASCII", QColor("#38bdf8"), True, 7),
            ])
        else:
            lines.extend([
                ("TÌM KIẾM NHỊ PHÂN (BST):", QColor("#fbbf24"), True, 8),
                ("• Bất biến: Key(L) < Key(Node) ≤ Key(R)", QColor("#94a3b8"), False, 7),
                ("• Duyệt In-Order (LNR) luôn tăng dần", QColor("#34d399"), False, 7),
                (f"• Thời gian: {'O(log n) tối ưu' if is_avl else 'O(n) suy biến'}", QColor("#e2e8f0"), True, 7),
            ])

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

    target = "class StemToolDialog(QDialog):"
    assert target in content, "Target class StemToolDialog(QDialog): not found!"

    new_content = content.replace(target, method_code + "\n\n" + target, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    py_compile.compile(path, doraise=True)
    print("Successfully inserted InformaticsTools.create_binary_tree_diagram and compiled!")

if __name__ == "__main__":
    main()
