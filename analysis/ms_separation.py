"""How well does each SMILES source reproduce the reported LC-MS mass?

This is the headline evaluation for the project, because the reported m/z
is the one number in the table that is *measured* rather than inferred.  A
structure whose formula does not reproduce the observed mass is the wrong
molecule, regardless of how convincing its ring diagram looks.

The script sweeps a range of tolerances and reports, for each SMILES
source, how many compounds pass.  The interesting output is the gap between
sources: it shows whether mass is a sharp discriminator or whether
everything fails at every tolerance (which would mean the m/z column itself
is unreliable).

Sources compared
----------------
* ``smiles_decimer``   -- what DECIMER read off the structure image
* ``repaired_smiles``  -- what the macrocycle heuristics produced
* ``smiles_best``      -- what the pipeline currently promotes

Run from the repo root::

    .venv/Scripts/python.exe analysis/ms_separation.py
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

SOURCES = ("smiles_decimer", "repaired_smiles", "smiles_best")
TOLERANCES = (50, 100, 200, 500, 1000, 2000, 5000, 10_000, 50_000)


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

    errors: dict[str, list[float]] = {s: [] for s in SOURCES}
    unknown = Counter()

    for row in rows:
        mz = _num(row.get("ms_mz"))
        pol = (row.get("ms_polarity") or "").strip()
        if mz is None:
            continue
        expected = neutral_mass_from_mz(mz, pol)
        if expected is None:
            unknown[pol] += 1
            continue
        for src in SOURCES:
            smi = (row.get(src) or "").strip()
            if not smi:
                continue
            mass = monoisotopic_mass(smi)
            if mass is None:
                continue
            errors[src].append((mass - expected) / expected * 1e6)

    print("=" * 72)
    print("MASS AGREEMENT BY SMILES SOURCE  (n = compounds with a usable m/z)")
    print("=" * 72)
    if unknown:
        print(
            "unrecognised polarity labels: "
            + ", ".join(f"{k!r} x{v}" for k, v in unknown.most_common(8))
        )
    print()
    header = f"{'tolerance':>13s} " + " ".join(
        f"{s.replace('smiles_', ''):>16s}" for s in SOURCES
    )
    print(header)
    print("-" * len(header))
    for tol in TOLERANCES:
        cells = []
        for src in SOURCES:
            vals = errors[src]
            n_ok = sum(1 for v in vals if abs(v) <= tol)
            cells.append(f"{n_ok:>6d}/{len(vals):<6d}")
        print(f"{tol:>10d} ppm " + " ".join(cells))
    print()

    print("ppm error distribution (|ppm|):")
    for src in SOURCES:
        vals = sorted(abs(v) for v in errors[src])
        if not vals:
            continue
        print(
            f"  {src:18s} n={len(vals):4d}  "
            f"p10={vals[int(len(vals) * 0.10)]:12.1f}  "
            f"median={statistics.median(vals):12.1f}  "
            f"p90={vals[int(len(vals) * 0.90)]:12.1f}"
        )
    print()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
