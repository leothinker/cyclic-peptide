# Read-only Phase-2 audit script. Writes outputs to data/audit/phase2/.
from __future__ import annotations
import csv
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from cpd.ms_validation import validate_against_ms
ROOT = Path(__file__).resolve().parents[1]
DIAG_CSV = ROOT / 'data/audit/mass_gate_diagnostic.csv'
COMBINED_CSV = ROOT / 'data/processed/compounds_combined.csv'
OUT_DIR = ROOT / 'data/audit/phase2'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PAIRED = OUT_DIR / 'paired_comparison.csv'
OUT_TERMINAL = OUT_DIR / 'terminal_bridge_only.csv'
OUT_POSITIVE = OUT_DIR / 'positive_tests.csv'
OUT_SUMMARY = OUT_DIR / 'phase1_corrected_summary.txt'
CMPD_1036_REFERENCE = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)"
    "C(=O)N3[C@@H](CC3)C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N"
    "[C@@H](CCC5=CC(F)=C(C(F)=C5)C(F)(F)F)"
    "C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)"
    "C(=O)N(C)[C@@H](C8CCCC8)"
    "C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
    "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
)
def _to_float(x):
    if x in (None, '', 'nan'):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
def phase1_corrected_summary(df_diag):
    sb = df_diag[df_diag['candidate_source'] == 'smiles_best'].copy()
    rs = df_diag[df_diag['candidate_source'] == 'repaired_smiles'].copy()
    for sub in (sb, rs):
        sub['delta_da_num'] = pd.to_numeric(sub['delta_da'], errors='coerce')
        sub['error_ppm_num'] = pd.to_numeric(sub['error_ppm'], errors='coerce')
        sub['tolerance_ppm_num'] = pd.to_numeric(sub['tolerance_ppm'], errors='coerce')
    sb_mismatch = sb[sb['verdict'] == 'mismatch'].copy()
    rs_mismatch = rs[rs['verdict'] == 'mismatch'].copy()
    sb_mismatch['abs_delta'] = sb_mismatch['delta_da_num'].abs()
    sb_mismatch['abs_ppm'] = sb_mismatch['error_ppm_num'].abs()
    rs_mismatch['abs_delta'] = rs_mismatch['delta_da_num'].abs()
    rs_mismatch['abs_ppm'] = rs_mismatch['error_ppm_num'].abs()
    lines = []
    lines.append('=== Phase 1 corrected (audit review note) ===')
    lines.append('Bucket totals must equal candidate totals:')
    lines.append('  smiles_best      : ' + str(len(sb)) + ' rows = '
        + str(int((sb['verdict'] == 'no_candidate').sum())) + ' '
        + 'no_candidate + ' + str(int((sb['verdict'] == 'mismatch').sum())) + ' mismatch')
    lines.append('  repaired_smiles  : ' + str(len(rs)) + ' rows = '
        + str(int((rs['verdict'] == 'no_candidate').sum())) + ' '
        + 'no_candidate + ' + str(int((rs['verdict'] == 'mismatch').sum())) + ' mismatch')
    lines.append('')
    lines.append('Note: previous report quoted the SIGNED median of smiles_best '
        + 'delta_da (~+1.87 Da). The absolute-value median tells us how far off '
        + 'the molecule is, so the headline figure is now:')
    lines.append('')
    lines.append('--- |delta_da| summary ---')
    for label, sub in [('smiles_best', sb_mismatch), ('repaired_smiles', rs_mismatch)]:
        if len(sub) == 0:
            lines.append('  ' + label + ': no mismatches')
            continue
        q = sub['abs_delta'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        qstr = ', '.join('p' + str(int(k * 100)) + '=' + ('%.2f' % v) for k, v in q.items())
        lines.append('  ' + label + ': n=' + str(len(sub))
            + ' min=' + ('%.3f' % sub['abs_delta'].min())
            + ' median=' + ('%.3f' % sub['abs_delta'].median())
            + ' max=' + ('%.3f' % sub['abs_delta'].max())
            + ' mean=' + ('%.3f' % sub['abs_delta'].mean()))
        lines.append('    quantiles: ' + qstr)
    lines.append('')
    lines.append('--- "|delta_da| < 1 Da" count (real near-misses) ---')
    near_sb = int((sb_mismatch['abs_delta'] < 1.0).sum())
    near_rs = int((rs_mismatch['abs_delta'] < 1.0).sum())
    lines.append('  smiles_best      : ' + str(near_sb))
    lines.append('  repaired_smiles  : ' + str(near_rs))
    lines.append('')
    lines.append('--- count inside the patent-precise ppm tolerance ---')
    in_tol_sb = int(((sb_mismatch['abs_ppm'] <= sb_mismatch['tolerance_ppm_num']).fillna(False)).sum())
    in_tol_rs = int(((rs_mismatch['abs_ppm'] <= rs_mismatch['tolerance_ppm_num']).fillna(False)).sum())
    lines.append('  smiles_best      : ' + str(in_tol_sb))
    lines.append('  repaired_smiles  : ' + str(in_tol_rs))
    lines.append('')
    lines.append('--- 5 cmpd_ids closest to tolerance boundary ---')
    sb_for_close = sb_mismatch.copy()
    sb_for_close['ppm_gap'] = (sb_for_close['abs_ppm'] - sb_for_close['tolerance_ppm_num']).abs()
    top5 = sb_for_close.nsmallest(5, 'ppm_gap')[['cmpd_id', 'abs_ppm', 'tolerance_ppm_num', 'ppm_gap', 'abs_delta']]
    lines.append(top5.to_string(index=False))
    return '\n'.join(lines)
def paired_compare(df_combined):
    out = []
    df = df_combined.copy()
    df['sb'] = df['smiles_best'].fillna('').astype(str).str.strip()
    df['rs'] = df['repaired_smiles'].fillna('').astype(str).str.strip()
    df['sb'] = df['sb'].where(df['sb'].str.lower() != 'nan', '')
    df['rs'] = df['rs'].where(df['rs'].str.lower() != 'nan', '')
    for _, row in df.iterrows():
        cid = int(row['cmpd_id'])
        for col, label in [('sb', 'smiles_best'), ('rs', 'repaired_smiles')]:
            smi = row[col]
            if not smi:
                out.append({'cmpd_id': cid, 'candidate_source': label,
                    'formula': '', 'observed_mass': '', 'expected_mass': '',
                    'delta_da': '', 'abs_error_ppm': '',
                    'within_tolerance': '', 'parsed': ''})
                continue
            mz = _to_float(row['ms_mz'])
            res = validate_against_ms(smi, mz, row['ms_polarity'])
            obs = res['observed_mass']
            exp = res['expected_mass']
            delta = (float(obs) - float(exp)) if (obs and exp) else None
            tol = res.get('tolerance_ppm')
            abs_ppm = abs(res['error_ppm']) if res['error_ppm'] is not None else None
            in_tol = (abs_ppm is not None and tol is not None and abs_ppm <= tol)
            out.append({'cmpd_id': cid, 'candidate_source': label,
                'formula': res['formula'] or '',
                'observed_mass': obs if obs is not None else '',
                'expected_mass': exp if exp is not None else '',
                'delta_da': round(delta, 3) if delta is not None else '',
                'abs_error_ppm': abs_ppm if abs_ppm is not None else '',
                'within_tolerance': 'yes' if in_tol else 'no',
                'parsed': 'yes' if Chem.MolFromSmiles(smi) else 'no'})
    return out
def terminal_bridge_subset(df_combined):
    sub = df_combined[df_combined['repair_method'] == 'terminal_bridge'].copy()
    out = []
    for _, row in sub.iterrows():
        cid = int(row['cmpd_id'])
        rs = str(row.get('repaired_smiles') or '').strip()
        sd = str(row.get('smiles_decimer') or '').strip()
        mz = _to_float(row['ms_mz'])
        pol = row['ms_polarity']
        for side_name, smi in [('repaired_smiles', rs), ('smiles_decimer', sd)]:
            if not smi or smi.lower() == 'nan':
                out.append({'cmpd_id': cid, 'side': side_name,
                    'formula': '', 'observed_mass': '', 'expected_mass': '',
                    'delta_da': '', 'abs_error_ppm': '', 'within_tolerance': '',
                    'parsed': ''})
                continue
            res = validate_against_ms(smi, mz, pol)
            obs = res['observed_mass']; exp = res['expected_mass']
            delta = (float(obs) - float(exp)) if (obs and exp) else None
            tol = res.get('tolerance_ppm')
            abs_ppm = abs(res['error_ppm']) if res['error_ppm'] is not None else None
            in_tol = (abs_ppm is not None and tol is not None and abs_ppm <= tol)
            out.append({'cmpd_id': cid, 'side': side_name,
                'formula': res['formula'] or '',
                'observed_mass': obs if obs is not None else '',
                'expected_mass': exp if exp is not None else '',
                'delta_da': round(delta, 3) if delta is not None else '',
                'abs_error_ppm': abs_ppm if abs_ppm is not None else '',
                'within_tolerance': 'yes' if in_tol else 'no',
                'parsed': 'yes' if Chem.MolFromSmiles(smi) else 'no'})
    return out
def positive_mass_gate_tests():
    cases = [
        ('glycine [M+H]+', 'NCC(=O)O', 76.0, '[M+H]+'),
        ('caffeine [M+H]+', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 195.1, '[M+H]+'),
        ('aspirin [M+H]+', 'CC(=O)Oc1ccccc1C(=O)O', 181.0, '[M+H]+'),
        ('glucose [M+Na]+', 'OCC1OC(O)C(O)C(O)C1O', 203.05, '[M+Na]+'),
        ('formic_acid [M-H]-', 'OC=O', 45.0, '[M-H]-'),
        ('cmpd_1036_reference [M+H]+', CMPD_1036_REFERENCE, 1610.9, '[M+H]+'),
    ]
    out = []
    for name, smi, mz, pol in cases:
        r = validate_against_ms(smi, mz, pol)
        delta = ((r['observed_mass'] or 0) - (r['expected_mass'] or 0))
        out.append({'case': name, 'mz': mz, 'polarity': pol,
            'observed_mass': r['observed_mass'], 'expected_mass': r['expected_mass'],
            'delta_da': round(delta, 4), 'error_ppm': r['error_ppm'],
            'tolerance_ppm': r.get('tolerance_ppm'),
            'verdict': r['verdict'], 'formula': r['formula']})
    return out
def main():
    df_diag = pd.read_csv(DIAG_CSV)
    df_combined = pd.read_csv(COMBINED_CSV)
    summary = phase1_corrected_summary(df_diag)
    OUT_SUMMARY.write_text(summary, encoding='utf-8')
    print(summary + '\n')
    paired = paired_compare(df_combined)
    with OUT_PAIRED.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired[0].keys()))
        w.writeheader()
        for r in paired:
            w.writerow(r)
    paired_df = pd.DataFrame(paired)
    print('wrote ' + str(OUT_PAIRED) + '  (' + str(len(paired_df)) + ' rows)')
    print('\\n--- Paired comparison (FILLED only) ---')
    sb_filled = paired_df[(paired_df['candidate_source'] == 'smiles_best') & (paired_df['formula'] != '')]
    rs_filled = paired_df[(paired_df['candidate_source'] == 'repaired_smiles') & (paired_df['formula'] != '')]
    print('  cmpds with smiles_best filled     : ' + str(sb_filled['cmpd_id'].nunique()))
    print('  cmpds with repaired_smiles filled : ' + str(rs_filled['cmpd_id'].nunique()))
    paired_ids = sorted(set(sb_filled['cmpd_id']) & set(rs_filled['cmpd_id']))
    print('  cmpds with BOTH filled             : ' + str(len(paired_ids)))
    if paired_ids:
        sb_tol = sb_filled.set_index('cmpd_id')['within_tolerance']
        rs_tol = rs_filled.set_index('cmpd_id')['within_tolerance']
        from collections import Counter
        ct = Counter()
        for cid in paired_ids:
            ct[(sb_tol.get(cid, ''), rs_tol.get(cid, ''))] += 1
        print('  tolerance transition (sb, rs):')
        for k, v in sorted(ct.items()):
            print('    ' + str(k) + ' : ' + str(v))
    tb = terminal_bridge_subset(df_combined)
    with OUT_TERMINAL.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(tb[0].keys()))
        w.writeheader()
        for r in tb:
            w.writerow(r)
    print('\nwrote ' + str(OUT_TERMINAL) + '  (' + str(len(tb)) + ' rows)')
    rs_pairs = {r['cmpd_id']: float(r['delta_da']) for r in tb if r['side'] == 'repaired_smiles' and r['delta_da'] != ''}
    sd_pairs = {r['cmpd_id']: float(r['delta_da']) for r in tb if r['side'] == 'smiles_decimer' and r['delta_da'] != ''}
    common = sorted(set(rs_pairs) & set(sd_pairs))
    if common:
        shifts = [rs_pairs[c] - sd_pairs[c] for c in common]
        print('--- terminal_bridge only ---')
        print('  cmpds with both deltas        : ' + str(len(common)))
        print('  repair-mass minus decimer-mass Da: '
            + 'median=' + ('%+.3f' % statistics.median(shifts)) + ' '
            + 'mean=' + ('%+.3f' % (sum(shifts)/len(shifts))) + '  '
            + '(positive = repair added mass)')
    pos = positive_mass_gate_tests()
    with OUT_POSITIVE.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pos[0].keys()))
        w.writeheader()
        for r in pos:
            w.writerow(r)
    print('\nwrote ' + str(OUT_POSITIVE))
    print('\n--- Positive mass-gate tests ---')
    for r in pos:
        line = ('  ' + str(r['case']).ljust(34) + ' verdict=' + str(r['verdict']).ljust(8)
            + ' delta=' + ('%+.4f' % r['delta_da']) + ' Da  '
            + 'err=' + str(r['error_ppm']) + ' ppm  '
            + 'tol=' + str(r['tolerance_ppm']) + ' ppm')
        print(line)
if __name__ == '__main__':
    main()
