"""Does turning OCR's bogus 'Ca' back into 'C' fix the composition?

``analysis/heavy_atom_mass_gate.py`` found 33 rows where the recogniser read
a two-letter element name out of the printed SMILES. In every case it is
``Ca``: a capital ``C`` followed by the chiral marker ``@@``, which the
character class merged into calcium. Calcium is 40.08 Da against carbon's
12.00, so each misread inflates the formula by 28.08 Da.

This script re-runs the heavy-atom mass gate with ``Ca`` normalised to
``C`` and reports how many of those rows then land on a whole hydrogen
count. It is the cheapest possible test of whether a purely lexical fix is
enough, before any search is attempted.

Run from the repo root::

    .venv/Scripts/python.exe analysis/check_ca_normalisation.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from cpd.ms_validation import neutral_mass_from_mz
from cpd.repair_audit import atom_multiset
from heavy_atom_mass_gate import H_PER_C_MAX, H_PER_C_MIN, MONO, heavy_mass

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"


def implied_hydrogen(counts: Counter, target: float) -> float | None:
    hb = target - heavy_mass(counts)
    if hb < 0:
        return None
    return hb / MONO["H"]


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    ca_rows = []
    for r in rows:
        smi = (r.get("smiles") or "").strip()
        if not smi or "Ca" not in smi:
            continue
        mz = (r.get("ms_mz") or "").strip()
        pol = (r.get("ms_polarity") or "").strip()
        if not mz or not pol:
            continue
        target = neutral_mass_from_mz(float(mz.replace(",", "")), pol)
        if target is None:
            continue
        ca_rows.append((r["cmpd_id"], smi, target))

    print(f"rows containing OCR 'Ca': {len(ca_rows)}")
    print()
    print(f"{'cmpd':>6} {'nCa':>4} {'before':>10} {'after':>10}  verdict")
    print(f"{'':>6} {'':>4} {'impl_H':>10} {'impl_H':>10}")

    fixed = 0
    for cmpd, smi, target in ca_rows:
        before = atom_multiset(smi)
        n_ca = before.get("Ca", 0)
        h_before = implied_hydrogen(before, target)

        after = Counter(before)
        after["Ca"] = 0
        after["C"] = after.get("C", 0) + n_ca
        h_after = implied_hydrogen(after, target)

        if h_after is None:
            verdict = "still too heavy"
        else:
            frac = abs(h_after - round(h_after))
            carbon = after.get("C", 0)
            in_range = (H_PER_C_MIN * carbon <= h_after
                        <= H_PER_C_MAX * carbon)
            if frac < 0.25 and in_range:
                verdict = f"FIXED (H{round(h_after)}, off {frac:.3f})"
                fixed += 1
            elif in_range:
                verdict = f"plausible but off by {frac:.2f} H"
            else:
                verdict = "still implausible"

        b_txt = f"{h_before:10.2f}" if h_before is not None else "     too heavy"
        a_txt = f"{h_after:10.2f}" if h_after is not None else "     too heavy"
        print(f"{cmpd:>6} {n_ca:>4} {b_txt} {a_txt}  {verdict}")

    print()
    print(f"rows brought onto a whole hydrogen count by Ca->C: {fixed} / {len(ca_rows)}")
    print()
    print("A fix here is a *composition* fix only. It says the formula is now")
    print("consistent with the measured mass; it does not yet say the")
    print("connectivity is right. That is the next stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
