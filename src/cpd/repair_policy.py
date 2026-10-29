"""Promote repaired SMILES into the deliverable CSV under an explicit policy.

The problem this replaces
-------------------------
An earlier script (``rebuild_repair.py``) copied *every* repaired SMILES
into ``smiles_best`` as long as the heuristics produced something that
looked like a macrocycle.  Because a heuristic that bolts a bond between two
distant atoms will always close *some* ring, that check is satisfied by
chemistry the model invented.  The resulting CSV therefore advertised a
recovery rate the data did not support.

The policy
----------
Two independent gates must both pass before a repaired SMILES is allowed
into ``smiles_best``:

1. **Chemistry gate** -- :func:`cpd.repair_audit.audit_repair` must return a
   tier of ``high`` or ``medium``, meaning the repair invented no atoms, lost
   no stereocentres, and produced a peptide-plausible macrocycle.
2. **Mass gate** -- :func:`cpd.ms_validation.validate_against_ms` must
   return ``match``, i.e. the formula reproduces the measured m/z within the
   precision the patent printed.

Everything that fails a gate is still recorded, but in clearly named
candidate columns so a chemist can inspect it.  Nothing is deleted: the raw
DECIMER output, the candidate repair, its tier and its mass error all remain
in the CSV.

Column contract
---------------
Two separate questions get two separate columns, because the audit showed
they can disagree and conflating them hides the real finding:

``repair_review_status`` -- about the repair heuristic
    ``not_repaired``      DECIMER already produced a macrocycle; no repair ran
    ``validated``         the repair invented no atoms and produced a
                          peptide-plausible ring
    ``experimental``      the repair needed invented atoms or truncation
    ``rejected``          the repair failed the chemistry gate outright
    ``not_attempted``     no DECIMER output to work from

``structure_confidence`` -- about the structure itself
    ``mass_verified``     the formula reproduces the reported m/z
    ``mass_mismatch``     it does not
    ``no_mass_reference`` no usable m/z for this row

``smiles_best``
    Best available structure: DECIMER's own output when its macrocycle is
    intact, otherwise a repair that passed the chemistry gate. A repair that
    only passed the chemistry gate is still a *proposal*, so
    ``structure_confidence`` is what tells a downstream user how far to
    trust it.
``smiles_candidate_repair``
    The heuristic's proposal when it did *not* pass the chemistry gate.
``repair_confidence``
    ``high`` / ``medium`` / ``low`` / ``experimental`` / ``rejected`` /
    ``none``.
``ms_check_verdict``, ``ms_error_ppm``
    Result of the mass gate against the patent's reported m/z.
``molecular_weight``, ``heavy_atom_count``
    Derived from ``smiles_best`` when available.

Run from the repo root::

    .venv/Scripts/python.exe -m cpd.repair_policy --inplace
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from rdkit import RDLogger, Chem
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

from cpd.macrocycle_repair import repair_macrocycle_with_method
from cpd.ms_validation import validate_against_ms
from cpd.render import is_macrocycle_intact
from cpd.repair_audit import audit_repair

DEFAULT_CSV = Path("data/processed/compounds_combined.csv")

NEW_COLUMNS = (
    "smiles_candidate_repair",
    "repair_confidence",
    "repair_review_status",
    "structure_confidence",
    "repair_atoms_invented",
    "repair_atoms_deleted",
    "ms_check_verdict",
    "ms_error_ppm",
)


def _num(value: str | None) -> Optional[float]:
    value = (value or "").strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mass_confidence(ms: dict) -> str:
    """Map a :func:`validate_against_ms` verdict onto ``structure_confidence``."""
    if ms.get("verdict") == "match":
        return "mass_verified"
    if ms.get("verdict") == "mismatch":
        return "mass_mismatch"
    return "no_mass_reference"


def process_rows(rows: list[dict], require_mass_gate: bool = True) -> list[dict]:
    """Apply the promotion policy to every row, returning new row dicts."""
    out: list[dict] = []
    for row in rows:
        new = dict(row)
        decimer = (row.get("smiles_decimer") or "").strip()
        mz = _num(row.get("ms_mz"))
        polarity = (row.get("ms_polarity") or "").strip()

        new.setdefault("smiles_candidate_repair", "")
        new["repair_confidence"] = "none"
        new["repair_review_status"] = "not_attempted"
        new["structure_confidence"] = "no_mass_reference"
        new["repair_atoms_invented"] = ""
        new["repair_atoms_deleted"] = ""
        new["ms_check_verdict"] = "unknown"
        new["ms_error_ppm"] = ""

        # ---- case 1: DECIMER already produced a plausible macrocycle ----
        if decimer and is_macrocycle_intact(decimer):
            ms = validate_against_ms(decimer, mz, polarity)
            new["ms_check_verdict"] = ms["verdict"]
            if ms["error_ppm"] is not None:
                new["ms_error_ppm"] = ms["error_ppm"]
            new["structure_confidence"] = _mass_confidence(ms)
            # No repair was needed; the status reflects that fact, not a
            # judgement about whether DECIMER's graph is chemically right.
            new["repair_confidence"] = "none"
            new["repair_review_status"] = "not_repaired"
            # Keep whatever smiles_best already held if DECIMER's own value
            # is what produced it; otherwise prefer the DECIMER canonical.
            if not (new.get("smiles_best") or "").strip():
                new["smiles_best"] = decimer
                new["smiles_best_source"] = "decimer"
            new["is_repaired"] = "False"
            new["repair_method"] = ""
            new["repaired_smiles"] = ""
            out.append(new)
            continue

        # ---- case 2: no DECIMER output at all ----
        if not decimer:
            new["repair_confidence"] = "none"
            new["repair_review_status"] = "not_attempted"
            new["structure_confidence"] = "no_mass_reference"
            new["is_repaired"] = "False"
            new["repair_method"] = "no_decimer_output"
            new["repaired_smiles"] = ""
            new["smiles_best"] = ""
            new["smiles_best_source"] = ""
            out.append(new)
            continue

        # ---- case 3: try the repair heuristics, then gate the result ----
        candidate, method = repair_macrocycle_with_method(decimer)
        if candidate is None:
            new["repair_confidence"] = "none"
            new["repair_review_status"] = "rejected"
            new["repair_method"] = "no_repair_found"
            new["repaired_smiles"] = ""
            new["smiles_best"] = ""
            new["smiles_best_source"] = ""
            out.append(new)
            continue

        audit = audit_repair(decimer, candidate, method or "", None)
        ms = validate_against_ms(candidate, mz, polarity)
        new["repair_confidence"] = audit.tier
        new["repair_atoms_invented"] = audit.atoms_invented
        new["repair_atoms_deleted"] = audit.atoms_deleted
        new["ms_check_verdict"] = ms["verdict"]
        if ms["error_ppm"] is not None:
            new["ms_error_ppm"] = ms["error_ppm"]
        new["repaired_smiles"] = candidate
        new["repair_method"] = method or ""
        new["is_repaired"] = "True"

        chemistry_ok = audit.validated
        mass_ok = ms["verdict"] == "match"
        new["structure_confidence"] = _mass_confidence(ms)
        if chemistry_ok and (mass_ok or not require_mass_gate):
            new["repair_review_status"] = "validated" if mass_ok else "experimental"
            new["smiles_best"] = candidate
            new["smiles_best_source"] = f"repair_{method}"
        else:
            new["repair_review_status"] = (
                "experimental"
                if chemistry_ok and not require_mass_gate
                else ("rejected" if audit.tier == "rejected" else "experimental")
            )
            new["smiles_candidate_repair"] = candidate
            # A rejected or unvalidated candidate must not be presented as
            # the production structure.
            new["smiles_best"] = ""
            new["smiles_best_source"] = ""
        out.append(new)

    # Derived columns, computed from whatever ended up in smiles_best.
    for new in out:
        best = (new.get("smiles_best") or "").strip()
        if best:
            mol = Chem.MolFromSmiles(best)
            if mol is not None:
                new["molecular_weight"] = round(Descriptors.MolWt(mol), 3)
                new["heavy_atom_count"] = mol.GetNumHeavyAtoms()
    return out


def summarise(rows: list[dict]) -> Counter:
    return Counter(r.get("repair_review_status") or "not_attempted" for r in rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="overwrite the input CSV instead of writing <name>.policy.csv",
    )
    ap.add_argument(
        "--no-mass-gate",
        action="store_true",
        help="promote on the chemistry gate alone (diagnostic only)",
    )
    args = ap.parse_args(argv)

    if not args.csv.exists():
        print(f"missing CSV: {args.csv}", file=sys.stderr)
        return 1
    with args.csv.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("input CSV is empty", file=sys.stderr)
        return 1

    fields = list(rows[0].keys())
    for col in NEW_COLUMNS:
        if col not in fields:
            fields.append(col)

    processed = process_rows(rows, require_mass_gate=not args.no_mass_gate)

    target = (
        args.csv
        if args.inplace
        else args.csv.with_suffix(".policy.csv")
    )
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in processed:
            writer.writerow({k: row.get(k, "") for k in fields})

    counts = summarise(processed)
    ms_counts = Counter(r.get("ms_check_verdict") or "unknown" for r in processed)
    conf_counts = Counter(
        r.get("structure_confidence") or "no_mass_reference" for r in processed
    )
    populated = sum(1 for r in processed if (r.get("smiles_best") or "").strip())

    print(f"rows                    : {len(processed)}")
    print(f"smiles_best populated   : {populated}")
    print("review status           :")
    for k, v in counts.most_common():
        print(f"   {k:16s} {v:4d}")
    print("structure confidence    :")
    for k, v in conf_counts.most_common():
        print(f"   {k:16s} {v:4d}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
