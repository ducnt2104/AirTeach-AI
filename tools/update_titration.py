"""
Update ChemistryTools.create_titration_curve_diagram in stem_tools.py to add vertical axis and half-equivalence marker.
"""

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        # pH axes lines
        ax_h = QGraphicsLineItem(g_x, g_y + g_h - 15.0, g_x + g_w, g_y + g_h - 15.0)
        ax_h.setPen(QPen(QColor("#64748b"), 1.0))
        ax_h.setZValue(10.6)
        items.append(ax_h)'''

replacement = '''        # pH axes lines
        ax_h = QGraphicsLineItem(g_x, g_y + g_h - 15.0, g_x + g_w, g_y + g_h - 15.0)
        ax_h.setPen(QPen(QColor("#64748b"), 1.0))
        ax_h.setZValue(10.6)
        items.append(ax_h)

        ax_v = QGraphicsLineItem(g_x, g_y + 10.0, g_x, g_y + g_h - 15.0)
        ax_v.setPen(QPen(QColor("#64748b"), 1.0))
        ax_v.setZValue(10.6)
        items.append(ax_v)

        # Half-equivalence point marker pH = pKa
        half_x = g_x + 0.25 * (g_w - 20.0) + 10.0
        half_y_pos = g_y + g_h - 15.0 - (4.76 / 14.0) * (g_h - 30.0)
        half_dot = QGraphicsEllipseItem(half_x - 3.5, half_y_pos - 3.5, 7.0, 7.0)
        half_dot.setBrush(QBrush(QColor("#10b981")))
        half_dot.setPen(QPen(QColor("#ffffff"), 1.0))
        half_dot.setZValue(11.5)
        items.append(half_dot)'''

assert target in text, "Target string not found"
text = text.replace(target, replacement, 1)

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated titration diagram successfully!")
