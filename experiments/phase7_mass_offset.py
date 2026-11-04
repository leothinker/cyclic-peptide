"""Where does the patent's m/z sit relative to the printed SMILES?

The mass gate rejects every candidate, and the rejections are not
random: the patent's printed ``m/z`` sits about +0.17 Da above the
monoisotopic ``[M+H]+`` of the printed SMILES.  A systematic offset
has two possible causes.

* The *reader* is systematically wrong -- every printed SMILES is
  missing (or has) some small group.  A character error that repeats
  in every compound would look exactly like this.
* The patent reports its masses under a different convention -- a
  different adduct, a charge state, an averaged mass.

This script measures the offset for every compound whose printed
SMILES parses, and reports it against each candidate convention, so
the two can be told apart.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_mass_offset.py \
        --csv data/processed/compounds_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROTON = 1.007276466
SODIUM = 22.98976928 - 0.00054858  # [M+Na]+ -> neutral sodium

# Two mass tables are compared.  The first is the monoisotopic one that
# RDKit uses.  The second is the same table with the hydrogen and the
# sulfur rounded the way a molecular-weight calculator rounds them
# (H = 1.008, S = 32.06).  The second table is what the patent's own
# m/z values turn out to agree with.
MONOISOTOPIC = {
    "C": 12.0,
    "H": 1.0078250319,
    "F": 18.99840322,
    "N": 14.0030740048,
    "O": 15.9949146196,
    "S": 31.9720711744,
    "Cl": 34.96885268,
    "Br": 78.9183376,
    "I": 126.904473,
    "P": 30.97376163,
}

ROUNDED = dict(MONOISOTOPIC)
ROUNDED["H"] = 1.008
ROUNDED["S"] = 32.06
ROUNDED["Cl"] = 35.45
ROUNDED["Br"] = 79.904
ROUNDED["I"] = 126.90
ROUNDED["P"] = 30.974


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default="data/processed/compounds_combined.csv"
    )
    parser.add_argument(
        "--out", default="data/audit/phase7/mass_offset.csv"
    )
    return parser.parse_args(argv)


def _masses(smiles: str) -> dict[str, float] | None:
    """The candidate ion masses of one SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    exact = Descriptors.ExactMolWt(molecule)
    average = Descriptors.MolWt(molecule)
    rounded = _table_mass(molecule, ROUNDED)
    mono = _table_mass(molecule, MONOISOTOPIC)
    return {
        "mono_1plus": mono + PROTON,
        "mono_2plus": (mono + 2 * PROTON) / 2,
        "rounded_1plus": rounded + PROTON,
        "rounded_2plus": (rounded + 2 * PROTON) / 2,
        "average_1plus": average + PROTON,
    }


def _table_mass(molecule, table: dict[str, float]) -> float:
    """The neutral mass of a molecule under an explicit mass table."""
    total = 0.0
    for element, count in _formula_counts(molecule).items():
        total += table.get(element, 0.0) * count
    return total


def _formula_counts(molecule) -> dict[str, int]:
    """The element counts of a molecule, hydrogens included."""
    counts: dict[str, int] = {}
    for atom in molecule.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        counts["H"] = counts.get("H", 0) + atom.GetTotalNumHs()
    return counts


_ELEMENTS = ("C", "H", "N", "O", "F", "S", "Cl", "I")


def _element_regression(records: list[dict]) -> dict | None:
    """Which element carries the offset between the patent and RDKit?

    The offset of every compound is fitted as a linear combination of
    its element counts.  A coefficient that comes out at a fraction of
    a dalton per atom means the patent computed its masses with a
    different atomic weight for that element.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a dependency
        return None

    matrix = []
    targets = []
    for record in records:
        delta = float(record["delta_mono_1plus_da"])
        if abs(delta) > 5:
            # A compound whose m/z is a 2+ ion is not an inlier.
            continue
        counts = _formula_counts_of(str(record["formula"]))
        matrix.append([counts.get(e, 0) for e in _ELEMENTS])
        targets.append(delta)
    if len(targets) < 10:
        return None
    design = np.hstack(
        [np.array(matrix, dtype=float), np.ones((len(matrix), 1))]
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, np.array(targets, dtype=float), rcond=None
    )
    residual = design @ coefficients - np.array(targets, dtype=float)
    return {
        "n": len(targets),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "shifts": {
            element: float(coefficients[index])
            for index, element in enumerate(_ELEMENTS)
        },
        "intercept": float(coefficients[-1]),
        "base": dict(MONOISOTOPIC),
    }


def _formula_counts_of(formula: str) -> dict[str, int]:
    """The element counts of a molecular formula."""
    import re

    counts: dict[str, int] = {}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if element:
            counts[element] = int(number or 1)
    return counts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = REPO / args.csv
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    records: list[dict[str, object]] = []
    for row in rows:
        smiles = (row.get("smiles") or "").strip()
        mz_text = (row.get("ms_mz") or "").strip()
        if not smiles or not mz_text:
            continue
        try:
            observed = float(mz_text.replace(",", ""))
        except ValueError:
            continue
        masses = _masses(smiles)
        if masses is None:
            continue
        record: dict[str, object] = {
            "cmpd_id": row.get("cmpd_id", ""),
            "ms_mz": observed,
            "formula": Chem.rdMolDescriptors.CalcMolFormula(
                Chem.MolFromSmiles(smiles)
            ),
        }
        for name, value in masses.items():
            delta = observed - value
            record[f"delta_{name}_da"] = delta
            record[f"delta_{name}_ppm"] = (
                delta / value * 1e6 if value else float("nan")
            )
        records.append(record)

    print(f"rows              : {len(rows)}")
    print(f"with a mass pair  : {len(records)}")
    print()
    print(f"{'convention':<18} {'median Da':>11} {'median ppm':>11} "
          f"{'min Da':>9} {'max Da':>9}")
    for name in (
        "mono_1plus",
        "mono_2plus",
        "rounded_1plus",
        "rounded_2plus",
        "average_1plus",
    ):
        deltas = [float(r[f"delta_{name}_da"]) for r in records]
        ppms = [float(r[f"delta_{name}_ppm"]) for r in records]
        if not deltas:
            continue
        print(
            f"{name:<18} {statistics.median(deltas):>11.4f} "
            f"{statistics.median(ppms):>11.1f} {min(deltas):>9.4f} "
            f"{max(deltas):>9.4f}"
        )

    print()
    for name in ("mono_1plus", "rounded_1plus", "rounded_2plus"):
        deltas = [float(r[f"delta_{name}_da"]) for r in records]
        within = sum(1 for d in deltas if abs(d) <= 0.05)
        print(f"within +-0.05 Da of {name:<16} {within}/{len(deltas)}")
        if deltas:
            spread = statistics.pstdev(deltas)
            print(
                f"   spread (sd) {spread:.4f} Da   "
                f"quartiles "
                f"{statistics.quantiles(deltas, n=4)[0]:.4f} / "
                f"{statistics.median(deltas):.4f} / "
                f"{statistics.quantiles(deltas, n=4)[2]:.4f}"
            )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        # How many compounds reproduce the patent's own m/z when the
        # mass table the patent used is assumed, and when the ion is
        # allowed to be singly or doubly charged.
        best = {"mono_1plus": 0, "rounded_1plus": 0,
                "mono_2plus": 0, "rounded_2plus": 0}
        matched = 0
        for record in records:
            options = {
                name: abs(float(record[f"delta_{name}_da"]))
                for name in best
            }
            name = min(options, key=options.get)
            best[name] += 1
            if options[name] <= 0.05:
                matched += 1
        print()
        print("closest convention per compound")
        for name, count in best.items():
            print(f"  {name:<16} {count}")
        print(
            f"  within +-0.05 Da of the closest one: {matched}/{len(records)}"
        )

    # Which element would have to carry the offset?  A character error
    # in the printed SMILES changes the mass by whole atoms, so an
    # offset of a fraction of a dalton cannot be a reading error; it
    # has to be a difference in the mass table.  The regression says
    # which atom.
    fit = _element_regression(records)
    if fit:
        print()
        print("per-element shift of the patent's m/z "
              "(least squares on the inliers)")
        print(f"  compounds used : {fit['n']}")
        print(f"  rms residual   : {fit['rms']:.4f} Da")
        for element, shift in fit["shifts"].items():
            print(
                f"  {element:<3} {shift:+.6f} Da/atom"
                f"   -> implied mass {fit['base'][element] + shift:.5f}"
            )
        print(f"  intercept      : {fit['intercept']:+.4f} Da")

        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
