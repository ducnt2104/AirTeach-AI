with open("src/ui/stem_tools.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == "'''" or line.strip() == '"""':
        # only skip if it's standalone and outside docstring
        pass
    new_lines.append(line)

# Let's inspect line 3456 specifically
for i, line in enumerate(lines):
    if "'''" in line and i > 3400 and i < 3500:
        lines[i] = ""

with open("src/ui/stem_tools.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
print("Cleaned line in stem_tools.py")
