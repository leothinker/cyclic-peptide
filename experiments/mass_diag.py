"""Read-only per-compound mass-gate diagnostic.

For every row we know the patent-reported m/z and polarity, plus whatever
candidate sits in smiles_best or repaired_smiles. We re-run
validate_against_ms ourselves, log per-row verdict, observed/expected
masses, formula, ppm error, and the size of the discrepancy in Da, and
classify the failure mode so the cause can be grouped.

Outputs:
  data/audit/mass_gate_diagnostic.csv  -- one row per cmpd_id
  data/audit/mass_gate_summary.txt     -- bucket counts
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

from cpd.ms_validation import validate_against_ms

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / 'data/processed/compounds_combined.csv'
OUT_DIR = ROOT / 'data/audit'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'mass_gate_diagnostic.csv'
OUT_TXT = OUT_DIR / 'mass_gate_summary.txt'

# Mass deltas that frequently appear as "exactly this many Da off" because
# DECIMER recognised most of a structure but dropped a single structural
# feature. Bucket assignment is purely diagnostic.
DELTA_HINT = [
    (-120, 'CF3 drop (~69 Da)'),
    (-114, 'CF3 / CH3OH (~114 Da)'),
    (-86,  'C5H10N / branched side chain (~86 Da)'),
    (-66,  'CF2 / ring (~66 Da)'),
    (-44,  'CO2 / missing carbonyl (~44 Da)'),
    (-28,  'CO / ethylene (~28 Da)'),
    (-18,  'H2O loss (~18 Da)'),
]


def closest_hint(delta_da: float) -> str:
    if abs(delta_da) > 200:
        return '>200 Da off (large)'
    best = None
    for d, label in DELTA_HINT:
        score = abs(delta_da - d)
        if best is None or score < best[0]:
            best = (score, label)
    return best[1] if best else 'unclassified'


def main() -> None:
    df = pd.read_csv(CSV_IN)
    out_rows = []
    smiles_best_buckets: dict[str, int] = {}
    repaired_buckets: dict[str, int] = {}
    for _, row in df.iterrows():
        cid = int(row['cmpd_id'])
        mz_raw = row.get('ms_mz', '')
        polarity = (row.get('ms_polarity') or '').strip()
        try:
            mz = float(mz_raw) if mz_raw not in ('', None) else None
        except (TypeError, ValueError):
            mz = None

        for cand_col, label in [('smiles_best', 'smiles_best'),
                                ('repaired_smiles', 'repaired_smiles')]:
            raw_smi = row.get(cand_col)
            if raw_smi is None:
                raw_smi = ''
            smi = str(raw_smi).strip()
            if not smi or smi.lower() == 'nan':
                out_rows.append({
                    'cmpd_id': cid,
                    'candidate_source': label,
                    'smiles': '',
                    'parsed': '',
                    'expected_mass': '',
                    'observed_mass': '',
                    'delta_da': '',
                    'error_ppm': '',
                    'tolerance_ppm': '',
                    'verdict': 'no_candidate',
                    'formula': '',
                    'polarity': polarity,
                    'ms_mz': mz_raw,
                    'failure_hint': '',
                })
                continue
            res = validate_against_ms(smi, mz, polarity)
            obs = res['observed_mass']
            exp = res['expected_mass']
            try:
                delta_da = (float(obs) - float(exp)) if (obs and exp) else None
            except (TypeError, ValueError):
                delta_da = None
            out_rows.append({
                'cmpd_id': cid,
                'candidate_source': label,
                'smiles': smi[:120],
                'parsed': 'yes' if Chem.MolFromSmiles(smi) is not None else 'no',
                'expected_mass': exp if exp is not None else '',
                'observed_mass': obs if obs is not None else '',
                'delta_da': round(delta_da, 3) if delta_da is not None else '',
                'error_ppm': res['error_ppm'] if res['error_ppm'] is not None else '',
                'tolerance_ppm': res['tolerance_ppm'] if 'tolerance_ppm' in res else '',
                'verdict': res['verdict'],
                'formula': res['formula'] or '',
                'polarity': polarity,
                'ms_mz': mz_raw,
                'failure_hint': closest_hint(delta_da) if delta_da is not None else '',
            })
            if label == 'smiles_best':
                smiles_best_buckets[res['verdict']] = smiles_best_buckets.get(res['verdict'], 0) + 1
            else:
                repaired_buckets[res['verdict']] = repaired_buckets.get(res['verdict'], 0) + 1

    fields = list(out_rows[0].keys())
    with OUT_CSV.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    def collect(rows, name):
        items = [r for r in rows if r['candidate_source'] == name
                 and r['verdict'] == 'mismatch' and r['delta_da'] != '']
        return [float(r['delta_da']) for r in items], items

    sb_deltas, sb_items = collect(out_rows, 'smiles_best')
    rs_deltas, rs_items = collect(out_rows, 'repaired_smiles')

    def stats(name, vals):
        if not vals:
            return f'{name}: no values'
        s = sorted(vals)
        return (f'{name}: n={len(vals)} '
                f'min={s[0]:+.3f} median={s[len(s)//2]:+.3f} '
                f'max={s[-1]:+.3f} mean={sum(s)/len(s):+.3f}')

    def bucket(vals, lo, hi):
        return sum(1 for v in vals if lo <= v < hi)

    lines = []
    lines.append('=== per-row verdict buckets (mass gate) ===')
    lines.append(f'smiles_best      : {smiles_best_buckets}')
    lines.append(f'repaired_smiles  : {repaired_buckets}')
    lines.append('')
    lines.append('=== smiles_best delta Da distribution ===')
    lines.append(stats('delta_Da', sb_deltas))
    for lo, hi in [(-1000, -100), (-100, -50), (-50, -20), (-20, -5),
                   (-5, -1), (-1, 1), (1, 5), (5, 20), (20, 50),
                   (50, 100), (100, 1000)]:
        lines.append(f'  [{lo:+5.0f},{hi:+5.0f}) Da : {bucket(sb_deltas, lo, hi)}')
    lines.append('')
    lines.append('=== repaired_smiles delta Da distribution ===')
    lines.append(stats('delta_Da', rs_deltas))
    for lo, hi in [(-1000, -100), (-100, -50), (-50, -20), (-20, -5),
                   (-5, -1), (-1, 1), (1, 5), (5, 20), (20, 50),
                   (50, 100), (100, 1000)]:
        lines.append(f'  [{lo:+5.0f},{hi:+5.0f}) Da : {bucket(rs_deltas, lo, hi)}')
    lines.append('')
    lines.append('=== smiles_best failure_hint counts ===')
    for k, v in Counter(r['failure_hint'] for r in sb_items).most_common():
        lines.append(f'  {k:40s} {v}')
    lines.append('')
    lines.append('=== repaired_smiles failure_hint counts ===')
    for k, v in Counter(r['failure_hint'] for r in rs_items).most_common():
        lines.append(f'  {k:40s} {v}')
    lines.append('')
    lines.append(f'wrote {OUT_CSV}')
    lines.append(f'wrote {OUT_TXT}')
    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
