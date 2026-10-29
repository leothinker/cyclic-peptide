"""Add ocr_bracket_deficit and ocr_difficulty to the CSV in place.

Does not re-run the search; just computes these two descriptive stats
from the existing OCR column. Useful for inspecting the search results
without paying for another full corpus run.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cpd.mass_constrained_search import bracket_deficit
CSV = ROOT / "data" / "processed" / "compounds_combined.csv"

def bucket(d: int) -> str:
    if d == 0:
        return "balanced"
    if d <= 2:
        return "light"
    if d <= 5:
        return "medium"
    if d <= 10:
        return "heavy"
    return "extreme"

def main() -> None:
    with CSV.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    for c in ("ocr_bracket_deficit", "ocr_difficulty"):
        if c not in fields:
            fields.append(c)
    n_by_bucket = {}
    for r in rows:
        s = (r.get("smiles") or "").strip()
        if not s:
            r["ocr_bracket_deficit"] = ""
            r["ocr_difficulty"] = "no_ocr"
            continue
        d = bracket_deficit(s)
        total = d["total"]
        r["ocr_bracket_deficit"] = str(total)
        r["ocr_difficulty"] = bucket(total)
        n_by_bucket.setdefault(r["ocr_difficulty"], 0)
        n_by_bucket[r["ocr_difficulty"]] += 1
    with CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"updated {CSV}")
    for k in sorted(n_by_bucket, key=lambda k: -n_by_bucket[k]):
        print(f"  {k:<10} {n_by_bucket[k]}")

if __name__ == "__main__":
    main()
