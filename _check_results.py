"""Detailed stats on compounds.csv vs decimer results."""
import json
import csv
from pathlib import Path

cache = json.loads(Path("data/processed/decimer_cache.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(open("data/processed/compounds_combined.csv", encoding="utf-8-sig")))

print(f"Total rows: {len(rows)}")

# OCR quality
ocr_ok = sum(1 for r in rows if r.get("canonical_smiles"))
ocr_bad = len(rows) - ocr_ok
print(f"OCR canonical_smiles present: {ocr_ok}, missing: {ocr_bad}")

# DECIMER quality
dec_ok = sum(1 for r in rows if r.get("canonical_smiles_decimer"))
dec_bad = len(rows) - dec_ok
print(f"DECIMER canonical_smiles_decimer present: {dec_ok}, missing: {dec_bad}")

# Per the CSV's smiles_agree (computed at write)
by_collider_agree = sum(1 for r in rows if r.get("smiles_agree") == "True")
print(f"CSV smiles_agree=True: {by_collider_agree}")

# More useful: how many have BOTH OCR and DECIMER canonical?
both = sum(1 for r in rows if r.get("canonical_smiles") and r.get("canonical_smiles_decimer"))
only_ocr = sum(1 for r in rows if r.get("canonical_smiles") and not r.get("canonical_smiles_decimer"))
only_dec = sum(1 for r in rows if r.get("canonical_smiles_decimer") and not r.get("canonical_smiles"))
neither = sum(1 for r in rows if not r.get("canonical_smiles") and not r.get("canonical_smiles_decimer"))
print(f"Both: {both}, only OCR: {only_ocr}, only DECIMER: {only_dec}, neither: {neither}")

# Agreement rate among rows where both exist
both_agree = sum(1 for r in rows if r.get("canonical_smiles") and r.get("canonical_smiles_decimer") and r.get("canonical_smiles") == r.get("canonical_smiles_decimer"))
print(f"Of the {both} with both: agree={both_agree}, disagree={both - both_agree}")

# Show a few examples of disagreement
print("\nFirst 3 disagreements:")
n_shown = 0
for r in rows:
    if r.get("canonical_smiles") and r.get("canonical_smiles_decimer") and r.get("canonical_smiles") != r.get("canonical_smiles_decimer"):
        print(f"  cmpd {r['cmpd_id']}:")
        print(f"    OCR canon:   {r['canonical_smiles'][:60]}")
        print(f"    DECIMER:     {r['canonical_smiles_decimer'][:60]}")
        n_shown += 1
        if n_shown >= 3:
            break

# Show examples where OCR is empty but DECIMER works
print("\nFirst 3 OCR-empty + DECIMER-OK:")
n_shown = 0
for r in rows:
    if not r.get("canonical_smiles") and r.get("canonical_smiles_decimer"):
        print(f"  cmpd {r['cmpd_id']}: DECIMER={r['canonical_smiles_decimer'][:60]}")
        n_shown += 1
        if n_shown >= 3:
            break
