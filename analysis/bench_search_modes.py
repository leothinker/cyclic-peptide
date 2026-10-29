"""Compare search modes on the 4 light rows."""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cpd.mass_constrained_search import (bracket_deficit, repair_from_ocr)
CSV = ROOT / "data" / "processed" / "compounds_combined.csv"
MODES = [
    {"allow_subs": True,  "allow_dels": True,  "label": "baseline"},
    {"allow_subs": False, "allow_dels": True,  "label": "insert+dels"},
    {"allow_subs": False, "allow_dels": False, "label": "insert-only"},
]
BUDGET_S = 15.0
BUDGET_ST = 1_500_000
def main():
    with CSV.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    targets = []
    for r in rows:
        s = (r.get("smiles") or "").strip()
        if not s:
            continue
        d = bracket_deficit(s)["total"]
        if d <= 5:
            targets.append((d, r))
    targets.sort(key=lambda x: x[0])
    print("light+medium rows:", len(targets))
    print("budget per row  : %ss / %d states" % (BUDGET_S, BUDGET_ST))
    print("")
    n_answer = 0
    for d, r in targets:
        s = (r.get("smiles") or "").strip()
        mz_raw = (r.get("ms_mz") or "").strip()
        pol = (r.get("ms_polarity") or "").strip()
        try:
            mz = float(mz_raw.replace(",", ""))
        except ValueError:
            continue
        for mode in MODES:
            t0 = time.time()
            result = repair_from_ocr(
                s, mz=mz, polarity=pol,
                max_seconds=BUDGET_S, max_states=BUDGET_ST,
                allow_subs=mode["allow_subs"],
                allow_dels=mode["allow_dels"],
            )
            elapsed = time.time() - t0
            if result is None or result.get("smiles") is None:
                trace = (result or {}).get("trace", "no_answer") if isinstance(result, dict) else "no_answer"
                answer = ""
            else:
                trace = "ok"
                answer = result["smiles"][:32]
                n_answer = n_answer + 1
            states = (result or {}).get("states", 0) if isinstance(result, dict) else 0
            line = "%-8s %-12s %-8.1fs %-22s %-8d %s" % (
                r["cmpd_id"], mode["label"], elapsed, trace, states, answer)
            print(line)
    print("\ntotal answers:", n_answer)
if __name__ == "__main__":
    main()
