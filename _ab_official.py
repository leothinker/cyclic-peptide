"""A/B test: CH rec vs EN rec on the same golden table image."""
import sys
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")

IMG = Path("data/raw/WO2025162428/golden_tables/PCTCN2025075389-ftappb-I100280.jpg")

from cpd.parsers.table_extractor import RapidOcrTableExtractor
from cpd.chem import validate_and_enrich


def run(rec_lang: str) -> dict:
    ext = RapidOcrTableExtractor(rec_lang=rec_lang)
    img = cv2.imread(str(IMG))
    cols, _, columns = ext._detect_columns(img), None, None
    cmpds, assays, columns = ext.extract_records(IMG, cells_dir=Path("data/_ab_tmp3"))
    n = 0
    valid = 0
    sample = []
    for c in cmpds:
        if not c.smiles:
            continue
        n += 1
        chem = validate_and_enrich(c.smiles)
        if chem is not None:
            valid += 1
        if len(sample) < 3:
            sample.append((c.cmpd_id, c.smiles))
    return {"n_cmpds": len(cmpds), "n_with_smiles": n, "n_valid": valid,
            "sample": sample}


print("=" * 60)
print(f"Image: {IMG.name}")
print("=" * 60)

for lang in ("ch", "en"):
    print(f"\n>>> rec_lang={lang!r}")
    res = run(lang)
    print(f"  cmpds:        {res['n_cmpds']}")
    print(f"  with SMILES:  {res['n_with_smiles']}")
    print(f"  RDKit valid:  {res['n_valid']}")
    for cid, smi in res["sample"]:
        print(f"  sample: {cid}  {smi[:80]}{'...' if len(smi) > 80 else ''}")
