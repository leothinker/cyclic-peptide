TARGETS = [
    'src/cpd/repair_policy.py',
    'src/cpd/mass_constrained_search.py',
    'src/cpd/ocr_rewrite.py',
    'src/cpd/repair_audit.py',
    'main.py',
]
for t in TARGETS:
    try:
        with open(t, encoding='utf-8') as fh:
            content = fh.read()
    except FileNotFoundError:
        print(f'\n========== {t} (not found) ==========\n')
        continue
    print(f'\n========== {t} ({len(content)} chars) ==========\n')
    print(content)
