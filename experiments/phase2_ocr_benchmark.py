# Read-only Phase-2 OCR benchmark for cmpd 1036, plus honest coverage report.
# Does not modify smiles_best, repaired_smiles, or any other CSV column.
# Outputs go to data/audit/phase2/.
from __future__ import annotations
import difflib, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from cpd.ms_validation import validate_against_ms
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'data/processed/compounds_combined.csv'
OUT_DIR = ROOT / 'data/audit/phase2'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_1036 = OUT_DIR / 'ocr_benchmark_cmpd_1036.csv'
OUT_OTHER = OUT_DIR / 'ocr_benchmark_other.txt'
REF_1036 = (
    'CC[C@H]' + '(C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)'
    'C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)'
    'C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)'
    '[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)'
    'C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C'
)
def cer(a, b):
    sm = difflib.SequenceMatcher(a=a, b=b)
    return sum(max(i2-i1, j2-j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal')
def _to_float(x):
    if x in (None, '', 'nan'):
        return None
    try: return float(x)
    except Exception: return None
def main():
    with CSV_PATH.open(newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    r1036 = next(r for r in rows if r['cmpd_id'] == '1036')
    ocr = (r1036['smiles'] or '').strip()
    mz = _to_float(r1036['ms_mz']); pol = r1036['ms_polarity']
    res_ref = validate_against_ms(REF_1036, mz, pol)
    res_ocr = validate_against_ms(ocr, mz, pol) if ocr else {'verdict': 'empty'}
    edit = cer(REF_1036, ocr) if ocr else None
    cer_score = (edit / len(REF_1036)) if edit is not None else None
    with OUT_1036.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh); w.writerow(['cmpd_id','metric','value'])
        w.writerow(['1036','reference_length_chars', len(REF_1036)])
        w.writerow(['1036','ocr_length_chars', len(ocr) if ocr else 0])
        w.writerow(['1036','edit_ops', edit if edit is not None else ''])
        w.writerow(['1036','character_error_rate', round(cer_score, 4) if cer_score is not None else ''])
        w.writerow(['1036','ms_mz', mz])
        w.writerow(['1036','ms_polarity', pol])
        w.writerow(['1036','reference_formula', res_ref.get('formula') or ''])
        w.writerow(['1036','reference_observed_mass', res_ref.get('observed_mass')])
        ref_delta = (res_ref['observed_mass'] or 0) - (res_ref['expected_mass'] or 0)
        w.writerow(['1036','reference_delta_da', round(ref_delta, 4)])
        w.writerow(['1036','reference_error_ppm', res_ref.get('error_ppm')])
        w.writerow(['1036','reference_tolerance_ppm', res_ref.get('tolerance_ppm')])
        w.writerow(['1036','reference_verdict', res_ref.get('verdict')])
        w.writerow(['1036','ocr_rdkit_parse', 'yes' if Chem.MolFromSmiles(ocr) else 'no'])
        w.writerow(['1036','ocr_verdict', res_ocr.get('verdict', 'no_ocr_text')])
        w.writerow(['1036','ocr_observed_mass', res_ocr.get('observed_mass')])
    print('wrote', OUT_1036)
    print('--- cmpd 1036 ---')
    print('reference chars :', len(REF_1036))
    print('OCR chars       :', len(ocr) if ocr else 0)
    print('edit ops        :', edit)
    print('CER             :', round(cer_score,4) if cer_score is not None else 'n/a')
    print('reference mass gate verdict :', res_ref.get('verdict'),
          '(delta', round(ref_delta,3), 'Da,', res_ref.get('error_ppm'),'ppm, tol',
          res_ref.get('tolerance_ppm'),'ppm)')
    print('OCR RDKit parse  :', 'yes' if Chem.MolFromSmiles(ocr) else 'no')
    print('OCR mass verdict :', res_ocr.get('verdict', 'no_ocr_text'))
    other = []
    for r in rows:
        cid = r['cmpd_id']
        if cid == '1036': continue
        smi = (r.get('smiles') or '').strip()
        if not smi or smi.lower() == 'nan': continue
        mz2 = _to_float(r.get('ms_mz')); pol2 = r.get('ms_polarity')
        v = validate_against_ms(smi, mz2, pol2)
        other.append({'cmpd_id': cid, 'ocr_chars': len(smi),
            'rdkit_parse': 'yes' if Chem.MolFromSmiles(smi) else 'no',
            'verdict': v.get('verdict'),
            'observed_mass': v.get('observed_mass'),
            'expected_mass': v.get('expected_mass'),
            'error_ppm': v.get('error_ppm')})
    with OUT_OTHER.open('w', encoding='utf-8') as fh:
        fh.write('cmpd_id,ocr_chars,rdkit_parse,verdict,observed_mass,expected_mass,error_ppm' + chr(10))
        for o in other:
            fh.write(','.join(str(o[k]) for k in ['cmpd_id','ocr_chars','rdkit_parse','verdict','observed_mass','expected_mass','error_ppm']) + chr(10))
    py = sum(1 for o in other if o['rdkit_parse'] == 'yes')
    nm = sum(1 for o in other if o['verdict'] == 'mismatch')
    lines = []
    lines.append('--- other cmpds with non-empty OCR smiles column ---')
    lines.append('count         : ' + str(len(other)))
    lines.append('RDKit parse   : ' + str(py) + ' / ' + str(len(other)))
    lines.append('mass mismatch : ' + str(nm) + ' / ' + str(len(other)))
    lines.append('')
    lines.append('NOTE: no hand-verified references for these cmpds; CER')
    lines.append('cannot be computed. To benchmark more cmpds, chemist-')
    lines.append('verified references are required.')
    txt = chr(10).join(lines)
    OUT_OTHER.with_suffix('.md').write_text(txt, encoding='utf-8')
    print(txt)
    print('wrote', OUT_OTHER)
if __name__ == '__main__':
    main()
