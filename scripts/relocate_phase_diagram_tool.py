import re

with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract method from MathTools
m = re.search(r'\n    @staticmethod\n    def create_phase_diagram_diagram\(.*?\n        return items\n', text, re.DOTALL)
if not m:
    print('Failed to find create_phase_diagram_diagram in text!')
    exit(1)

phase_method = m.group(0)
# Remove from MathTools
text_without = text[:m.start()] + text[m.end():]

# Find insertion spot in ChemistryTools (before BiologyTools)
bio_idx = text_without.find('class BiologyTools:')
if bio_idx == -1:
    print('Failed to find class BiologyTools!')
    exit(1)

# Find last return items before BiologyTools
ret_idx = text_without[:bio_idx].rfind('        return items\n')
insert_pos = ret_idx + len('        return items\n')

new_text = text_without[:insert_pos] + phase_method + text_without[insert_pos:]

with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Successfully relocated create_phase_diagram_diagram to ChemistryTools!')
