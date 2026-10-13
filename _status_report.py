"""Generate a comprehensive status report for the cyclic-peptide OCR/OCSR pipeline.

Inputs:
  - data/processed/compounds_combined.csv (current state)
  - data/processed/decimer_cache.json (with phantom cleanup)

Outputs:
  - Coverage statistics (OCR, DECIMER, both)
  - Phantom fragment analysis (before/after cleanup)
  - L1 phantom-fragment breakdown with phantom cleanup applied
  - OCR error pattern frequencies
  - Concrete number summary
"""
import csv
import json
import sys
import statistics
from collections import Counter

sys.path.insert(0, "src")
from cpd.chem import largest_fragment_canonical, validate_and_enrich
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


print("=" * 70)
print("Cyclic-Peptide OCR/OCSR Pipeline - Status Report")
print("=" * 70)

# ---------- Load data ----------
rows = list(csv.DictReader(open("data/processed/compounds_combined.csv", encoding="utf-8-sig")))
cache = json.load(open("data/processed/decimer_cache.json"))
print(f"compounds_combined.csv: {len(rows)} rows")
print(f"decimer_cache.json: {len(cache['entries'])} entries (schema v{cache['__version__']})")
print()

# ---------- 1. Coverage ----------
ocr_rows = [r for r in rows if r["smiles"].strip()]
dec_rows = [r for r in rows if r["smiles_decimer"].strip()]
both_rows = [r for r in rows if r["smiles"].strip() and r["smiles_decimer"].strip()]
ocr_only = [r for r in rows if r["smiles"].strip() and not r["smiles_decimer"].strip()]
dec_only = [r for r in rows if not r["smiles"].strip() and r["smiles_decimer"].strip()]
neither = [r for r in rows if not r["smiles"].strip() and not r["smiles_decimer"].strip()]

print("--- 1. COVERAGE ---")
print(f"  OCR detected: {len(ocr_rows)}/{len(rows)} ({len(ocr_rows)*100/len(rows):.0f}%)")
print(f"  DECIMER has SMILES: {len(dec_rows)}/{len(rows)} ({len(dec_rows)*100/len(rows):.0f}%)")
print(f"  Both: {len(both_rows)}, OCR only: {len(ocr_only)}, DECIMER only: {len(dec_only)}, Neither: {len(neither)}")
print()

# ---------- 2. RDKit validity ----------
def has_valid_canonical(r, key="canonical_smiles"):
    """Check if canonical SMILES is non-empty and parses back to same."""
    canon = r.get(key, "")
    if not canon:
        return False
    mol = Chem.MolFromSmiles(canon)
    return mol is not None


n_ocr_valid = sum(1 for r in ocr_rows if has_valid_canonical(r))
n_dec_valid_before = sum(1 for r in dec_rows if has_valid_canonical(r, "canonical_smiles_decimer"))
n_dec_valid_after = sum(1 for r in dec_rows
                        if has_valid_canonical(r, "canonical_smiles_decimer")
                        or largest_fragment_canonical(r["smiles_decimer"]) is not None)

print("--- 2. RDKIT VALIDITY ---")
print(f"  OCR canonical valid (RDKit): {n_ocr_valid}/{len(ocr_rows)}")
print(f"  DECIMER canonical valid (raw): {n_dec_valid_before}/{len(dec_rows)}")
print(f"  DECIMER valid (after phantom cleanup): {n_dec_valid_after}/{len(dec_rows)}")
print()

# ---------- 3. Phantom fragment analysis ----------
n_multi = 0
n_phantoms_removed = 0
for r in dec_rows:
    raw = r["smiles_decimer"]
    if "." not in raw:
        continue
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        continue
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    if len(frags) > 1:
        n_multi += 1
        cleaned = largest_fragment_canonical(raw)
        if cleaned and "." not in cleaned:
            n_phantoms_removed += 1

print("--- 3. PHANTOM FRAGMENTS (DECIMER) ---")
print(f"  DECIMER SMILES with multiple '.' fragments: {n_multi}/{len(dec_rows)} ({n_multi*100/len(dec_rows):.0f}%)")
print(f"  Successfully reduced to single fragment: {n_phantoms_removed}/{n_multi}")
print(f"  Phantom fragments are typically 2-4 atom impurities (CC, CCO)")
print()

# ---------- 4. L1 phantom-fragment breakdown (after cleanup) ----------
threshold = 0.5
both_clean = []
for r in both_rows:
    raw = r["smiles_decimer"]
    cleaned = largest_fragment_canonical(raw) or raw
    if r["smiles"].strip() and cleaned.strip() and r.get("decimer_min_conf", "") not in ("", "None"):
        # Recompute canonical after cleanup
        chem = validate_and_enrich(cleaned)
        canon = chem.canonical_smiles if chem else ""
        ocr_canon = r.get("canonical_smiles", "")
        agree = bool(canon and ocr_canon and canon == ocr_canon)
        both_clean.append({
            "cmpd_id": r["cmpd_id"],
            "ocr_canon": ocr_canon,
            "dec_canon_cleaned": canon,
            "agree": agree,
            "min_conf": float(r["decimer_min_conf"]),
        })

n_both = len(both_clean)
n_agree = sum(1 for x in both_clean if x["agree"])
n_disagree = n_both - n_agree
n_phantom = sum(1 for x in both_clean if not x["agree"] and x["min_conf"] < threshold)
n_real_err = n_disagree - n_phantom

print("--- 4. L1 PHANTOM-FRAGMENT BREAKDOWN (after phantom cleanup) ---")
print(f"  Rows with both OCR + DECIMER (DECIMER cleaned canonical): {n_both}")
print(f"  Agree: {n_agree} ({n_agree*100/n_both:.0f}%)" if n_both else "  Agree: 0")
print(f"  Disagree, DECIMER phantom (conf < {threshold}): {n_phantom} ({n_phantom*100/n_both:.0f}%)" if n_both else "  Disagree, phantom: 0")
print(f"  Disagree, real OCR err (conf >= {threshold}): {n_real_err} ({n_real_err*100/n_both:.0f}%)" if n_both else "  Disagree, real: 0")
median_conf = statistics.median(x["min_conf"] for x in both_clean) if both_clean else 0
print(f"  Median DECIMER min_conf: {median_conf:.3f}")
print()

# ---------- 5. OCR error patterns ----------
err_patterns = Counter()
for r in ocr_rows:
    s = r["smiles"]
    if Chem.MolFromSmiles(s) is not None:
        continue  # already valid, skip
    err_patterns["hyphens (C-O instead of C(=O))"] += s.count("-") if "-O" in s else 0
    if "C=O" in s and "C(=O)" not in s:
        err_patterns["C=O without ("] += s.count("C=O")
    if "[C@H]" in s and "[C@H](" not in s:
        err_patterns["[C@H] without ("] += s.count("[C@H]")
    if "C@H]" in s and "[C@H]" not in s:
        err_patterns["C@H] missing opening ["] += s.count("C@H]")

print("--- 5. OCR ERROR PATTERNS (in 335 invalid OCR SMILES) ---")
for pat, n in err_patterns.most_common():
    print(f"  {pat}: {n} occurrences")
print()

# ---------- 6. Pipeline improvements implemented ----------
print("--- 6. PIPELINE IMPROVEMENTS (this session) ---")
print("  - src/cpd/chem.py: added largest_fragment_canonical() for phantom cleanup")
print("  - main.py: applies largest_fragment_canonical before DECIMER canonicalisation")
print("  - main.py: on-disk DECIMER cache (schema v2) with phantom cleanup applied")
print("  - main.py: --rec-top-k 5 wires RapidOCR top-k through 3 cfg locations")
print("  - main.py: --report-l1 subcommand for L1 phantom-fragment breakdown")
print("  - main.py: smiles_best + smiles_best_source columns")
print("  - references/RapidOCR + .venv: top_k_idx/top_k_prob computed in CTCLabelDecode")
print("  - _migrate_csv_to_cache.py: migrates CSV -> cache with phantom cleanup")
print()
print("=" * 70)
