path = "src/ui/stem_tools.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

target = '''        txt_hdr = QGraphicsTextItem("♀ \\\\ ♂")
        txt_hdr.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        txt_hdr.setDefaultTextColor(QColor("#94a3b8"))
        txt_hdr.setPos(grid_x + 4.0, grid_y + s / 2.0 - 12.0)
        txt_hdr.setZValue(11.5)
        items.append(txt_hdr)'''

replacement = '''        female_sym = QGraphicsTextItem("♀")
        female_sym.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        female_sym.setDefaultTextColor(QColor("#f9a8d4"))
        female_sym.setPos(grid_x + 6.0, grid_y + s / 2.0 - 12.0)
        female_sym.setZValue(11.5)
        items.append(female_sym)

        male_sym = QGraphicsTextItem("♂")
        male_sym.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        male_sym.setDefaultTextColor(QColor("#60a5fa"))
        male_sym.setPos(grid_x + 24.0, grid_y + s / 2.0 - 12.0)
        male_sym.setZValue(11.5)
        items.append(male_sym)'''

if target in text:
    text = text.replace(target, replacement, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully split txt_hdr into individual female and male symbol items!")
else:
    print("Target not found!")
