with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = 'txt_hdr = QGraphicsTextItem("♀ \\\\ ♂")'
assert target in text, f"Target {target} not found"

replacement = '''txt_hdr = QGraphicsTextItem("♀ \\\\ ♂")
        male_sym = QGraphicsTextItem("♂")
        male_sym.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        male_sym.setDefaultTextColor(QColor("#60a5fa"))
        male_sym.setPos(grid_x + s * 1.5 - 6.0, grid_y - 28.0)
        male_sym.setZValue(11.0)
        items.append(male_sym)

        fem_sym = QGraphicsTextItem("♀")
        fem_sym.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        fem_sym.setDefaultTextColor(QColor("#f472b6"))
        fem_sym.setPos(grid_x - 22.0, grid_y + s * 1.5 - 12.0)
        fem_sym.setZValue(11.0)
        items.append(fem_sym)'''

text = text.replace(target, replacement, 1)

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(text)

import py_compile
py_compile.compile('src/ui/stem_tools.py', doraise=True)
print("Successfully updated punnett gender symbols and compiled!")
