"""Inspect the *absolute* Da gap between each SMILES and the reported m/z.

Percentages hide the shape of the distribution.  This script prints the
Da deltas directly and checks two alternative explanations for a systematic
mass offset before anyone concludes that DECIMER is wrong:

1. **A missing adduct.**  If every delta sits a constant ~22 Da (sodium) or
   ~38 Da (potassium) above the reported mass, then the m/z column may be
   quoting a different ion than the label suggests.
2. **A systematic over-read.**  If deltas cluster around a few large values
   it points at DECIMER mis-reading specific substructures (halogens,
   ring-closure digits) rather than to random noise.

Run from the repo root::

    .venv/Scripts/python.exe analysis/delta_distribution.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from cpd.ms_validation import monoisotopic_mass, neutral_mass_from_mz  # noqa: E402

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

# Candidate systematic offsets worth testing.
OFFSETS = {
    "sodium +22.99": 22.98977,
    "potassium +38.96": 38.96371,
    "ammonium +18.03": 18.03382,
    "methanol +32.03": 32.02621,
    "water +18.01": 18.01056,
    "formate +45.00": 44.99820,
}


def _num(value: str | None) -> float | None:
    value = (value or "").strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    deltas: list[tuple[str, float, float, float]] = []  # cmpd, mz, obs, delta
    for row in rows:
        mz = _num(row.get("ms_mz"))
        pol = (row.get("ms_polarity") or "").strip()
        smi = (row.get("smiles_decimer") or "").strip()
        if mz is None or not smi:
            continue
        expected = neutral_mass_from_mz(mz, pol)
        mass = monoisotopic_mass(smi)
        if expected is None or mass is None:
            continue
        deltas.append(
            ((row.get("cmpd_id") or "").strip(), mz, mass, mass - expected)
        )

    print("=" * 72)
    print("ABSOLUTE MASS DELTA: DECIMER formula vs reported LC-MS")
    print("=" * 72)
    print(f"compounds compared: {len(deltas)}")
    if not deltas:
        return 1

    d = sorted(x[3] for x in deltas)
    print(
        f"delta Da: min={d[0]:+.2f}  p10={d[int(len(d) * 0.10)]:+.2f}  "
 f"median={statistics.median(d):+.2f}  "
        f"p90={d[int(len(d) * 0.90)]:+.2f}  max={d[-1]:+.2f}"
    )
    print()
    sign = Counter("over" if x > 0 else "under" for x in d)
    print(f"direction: over-mass {sign['over']}   under-mass {sign['under']}")
    print()

    print("-- how many deltas would a constant adduct offset explain? --")
    for label, off in OFFSETS.items():
        n = sum(1 for x in d if abs(abs(x) - off) < 1.5)
        print(f"   {label:22s} : {n:4d} compounds")
    n0 = sum(1 for x in d if abs(x) < 1.0)
    print(f"   {'no offset (|d|<1 Da)':22s} : {n0:4d} compounds")
    print()

    print("-- closest 15 compounds --")
    for cmpd, mz, obs, delta in sorted(deltas, key=lambda t: abs(t[3]))[:15]:
        print(f"   cmpd {cmpd:>5}  mz={mz:9.1f}  formula={obs:11.3f}  "
              f"delta={delta:+9.3f} Da")
    print()
    print("-- largest 10 over-reads --")
    for cmpd, mz, obs, delta in sorted(deltas, key=lambda t: -t[3])[:10]:
        print(f"   cmpd {cmpd:>5}  mz={mz:9.1f}  formula={obs:11.3f}  "
              f"delta={delta:+9.3f} Da")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
