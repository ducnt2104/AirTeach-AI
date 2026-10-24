with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('        @staticmethod\n    def create_trophic_pyramid(', '    @staticmethod\n    def create_trophic_pyramid(')
text = text.replace('"♀ \ ♂"', '"♀ \\\\ ♂"')

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed indentation and escape sequence in stem_tools.py")
