import re
path = 'phase2_mass_gate_audit.py'
src = open(path, encoding='utf-8').read()
pattern = re.compile(r"    print\('\\n--- Paired comparison ---'\).*?print\('  yes->no \(regression\)   : ' \+ str\(flip2\)", re.S)
m = pattern.search(src)
if not m:
    print('block not matched; abort')
    raise SystemExit(1)
N = chr(10)
lines = [
    "    print('\\\\n--- Paired comparison (FILLED only) ---')",
    "    sb_filled = paired_df[(paired_df['candidate_source'] == 'smiles_best') & (paired_df['formula'] != '')]",
    "    rs_filled = paired_df[(paired_df['candidate_source'] == 'repaired_smiles') & (paired_df['formula'] != '')]",
    "    print('  cmpds with smiles_best filled     : ' + str(sb_filled['cmpd_id'].nunique()))",
    "    print('  cmpds with repaired_smiles filled : ' + str(rs_filled['cmpd_id'].nunique()))",
    "    paired_ids = sorted(set(sb_filled['cmpd_id']) & set(rs_filled['cmpd_id']))",
    "    print('  cmpds with BOTH filled             : ' + str(len(paired_ids)))",
    "    if paired_ids:",
    "        sb_tol = sb_filled.set_index('cmpd_id')['within_tolerance']",
    "        rs_tol = rs_filled.set_index('cmpd_id')['within_tolerance']",
    "        from collections import Counter",
    "        ct = Counter()",
    "        for cid in paired_ids:",
    "            ct[(sb_tol.get(cid, ''), rs_tol.get(cid, ''))] += 1",
    "        print('  tolerance transition (sb, rs):')",
    "        for k, v in sorted(ct.items()):",
    "            print('    ' + str(k) + ' : ' + str(v))",
    "",
]
replacement = N.join(lines)
new_src = src[:m.start()] + replacement + src[m.end():]
open(path, 'w', encoding='utf-8').write(new_src)
print('patched OK, size delta:', len(new_src) - len(src))
