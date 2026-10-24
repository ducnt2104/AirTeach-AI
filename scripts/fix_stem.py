with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

broken = 'otxt = QGraphicsTextItem(f"So sánh: {comparisons}\nHoán đổi: {swaps}")'
fixed = 'otxt = QGraphicsTextItem(f"So sánh: {comparisons}\\nHoán đổi: {swaps}")'

if broken in text:
    text = text.replace(broken, fixed, 1)
    with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Fixed stem_tools.py with LF")
else:
    broken_crlf = 'otxt = QGraphicsTextItem(f"So sánh: {comparisons}\r\nHoán đổi: {swaps}")'
    if broken_crlf in text:
        text = text.replace(broken_crlf, fixed, 1)
        with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Fixed stem_tools.py with CRLF")
    else:
        print("Broken string not found")
