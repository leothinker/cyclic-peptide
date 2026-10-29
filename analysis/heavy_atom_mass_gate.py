"""Can the reported m/z pin down the element composition on its own?

Background
----------
Every repair attempt so far has been judged on the finished SMILES, which
means the search had to get the *brackets* right before it could learn
anything from the mass. That is backwards: brackets do not change which
atoms are present. ``C(=O)N``, ``C=O)N`` and ``CON`` all carry the same
heavy atoms, so the heavy-atom mass is invariant under exactly the edits
that dominate our OCR damage.

That suggests a two-stage decomposition:

1. **Atom stage** -- decide what the molecule is *made of*, using a gate
   that does not care about brackets at all.
2. **Structure stage** -- decide how those atoms are *connected*, using
   grammar and RDKit.

This script measures whether stage 1 is tractable on this corpus. For every
row it compares the heavy-atom mass implied by the OCR text against the
neutral mass implied by the patent's m/z, and reports the gap as a hydrogen
budget.

Interpretation
--------------
``H_budget = M_target - M_heavy``. If that lands near a whole number of
hydrogens, the OCR got the composition right and only the connectivity is
damaged -- the good case. If it is negative, OCR invented atoms that must be
deleted. If it is far too large, OCR lost atoms that must be recovered.

Run from the repo root::

    .venv/Scripts/python.exe analysis/heavy_atom_mass_gate.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpd.ms_validation import neutral_mass_from_mz
from cpd.repair_audit import atom_multiset

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

MONO = {
    "H": 1.0078250319,
    "C": 12.0,
    "N": 14.0030740048,
    "O": 15.9949146196,
    "F": 18.99840316273,
    "P": 30.97376151,
    "S": 31.9720711744,
    "Cl": 34.96885271,
    "Br": 78.9183376,
    "I": 126.904473,
    "Si": 27.9769265325,
    "Se": 79.9165218,
    "B": 11.0093054,
}

# A peptide's hydrogen count is bounded by its carbon count: every carbon
# carries at most 3 H, and a saturated acyclic chain averages ~2. These are
# deliberately wide -- the point is to catch "wildly wrong", not to be tight.
H_PER_C_MIN = 0.5
H_PER_C_MAX = 3.0


def heavy_mass(counts: Counter) -> float:
    """Monoisotopic mass of the heavy atoms only, unknown elements ignored."""
    total = 0.0
    for el, n in counts.items():
        if el in MONO:
            total += MONO[el] * n
    return total


def unknown_elements(counts: Counter) -> list:
    return sorted(e for e in counts if e not in MONO)


def classify(u: dict) -> str:
    if u["unknown"]:
        return "unknown_element_read"
    hb = u["h_budget"]
    if hb < 0:
        return "too_heavy_invented_atoms"
    implied_h = hb / MONO["H"]
    if not (H_PER_C_MIN * u["carbon"] <= implied_h <= H_PER_C_MAX * u["carbon"]):
        return "h_budget_implausible"
    return "composition_plausible"


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    usable = []
    for r in rows:
        smi = (r.get("smiles") or "").strip()
        mz = (r.get("ms_mz") or "").strip()
        pol = (r.get("ms_polarity") or "").strip()
        if not smi or not mz or not pol:
            continue
        try:
            mz_val = float(mz.replace(",", ""))
        except ValueError:
            continue
        target = neutral_mass_from_mz(mz_val, pol)
        if target is None:
            continue
        counts = atom_multiset(smi)
        hm = heavy_mass(counts)
        usable.append({
            "cmpd": r["cmpd_id"],
            "target": target,
            "heavy": hm,
            "h_budget": target - hm,
            "counts": counts,
            "unknown": unknown_elements(counts),
            "carbon": counts.get("C", 0),
        })

    print(f"rows with OCR SMILES + a usable m/z: {len(usable)} / {len(rows)}")
    print()

    buckets: Counter = Counter()
    near = []
    for u in usable:
        verdict = classify(u)
        buckets[verdict] += 1
        if verdict == "composition_plausible":
            implied_h = u["h_budget"] / MONO["H"]
            frac = abs(implied_h - round(implied_h))
            near.append({**u, "implied_h": implied_h, "frac": frac})

    print("verdict per row:")
    for k, v in buckets.most_common():
        print(f"  {k:30s} {v:4d}  ({v * 100 / len(usable):5.1f}%)")
    print()

    tight = [d for d in near if d["frac"] < 0.25]
    print(f"composition plausible                  : {len(near)}")
    print(f"  of those, budget on a whole H count  : {len(tight)}")
    if tight:
        tight.sort(key=lambda d: d["frac"])
        print("  tightest 12 (|implied_H - round(H)|):")
        for d in tight[:12]:
            print(f"    cmpd {d['cmpd']:>5}  C{d['carbon']:>3}  "
                  f"target {d['target']:9.3f}  heavy {d['heavy']:9.3f}  "
                  f"implied_H {d['implied_h']:7.2f}")

    print()
    unknown_rows = [u for u in usable if u["unknown"]]
    if unknown_rows:
        seen: Counter = Counter()
        for u in unknown_rows:
            for e in u["unknown"]:
                seen[e] += 1
        print(f"rows where OCR produced an element name: {len(unknown_rows)}")
        print("  ", dict(seen.most_common(12)))
        print("  'Ca' is almost always a misread '[C@@H]': a capital C followed")
        print("  by the chiral marker renders as a two-letter element name.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
