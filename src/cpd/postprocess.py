"""Derived columns of the combined CSV.

Why this module exists
----------------------
The combined CSV grew a third of its columns one script at a time, and
the scripts that wrote them were ad hoc: they were run by hand, they
were never committed, and the rules they applied were not written
down.  Rebuilding the CSV therefore meant remembering the order in
which four different programs had to be run, and it was impossible to
tell which of the values in a column were current and which were left
over from an earlier run of the pipeline.

This module puts the derivations in one place, with the rules spelled
out, so that a rebuild is a single reproducible step.

The three layers
----------------
``structure`` -- derived from the DECIMER output
    ``decimer_largest_ring_size``, ``decimer_macrocycle_intact`` and
    ``decimer_num_rings`` describe the ring system of the structure
    DECIMER read from the cropped image.  They are computed from the
    *canonical* DECIMER SMILES, because that is the form the pipeline
    stores and the form the mass gate uses.

``quality`` -- derived from the assay values and the structure
    ``data_quality_flags`` lists every reason a row is not fully
    trustworthy, and ``data_quality`` is the single-word summary:
    ``ok``, ``macrocycle_broken`` or ``partial``.

``text`` -- derived from the printed-SMILES OCR text
    ``ocr_bracket_deficit`` counts the brackets the recogniser lost,
    and ``ocr_difficulty`` buckets that count.  A deficit of zero
    means the printed text is bracket-balanced, which is a necessary
    condition for RDKit to parse it -- but not a sufficient one, and
    certainly not a statement that the structure is correct.

The repair layer is not here
----------------------------
:mod:`cpd.repair_policy` adds the DECIMER-side repair columns, and
:mod:`cpd.mass_constrained_search` adds the OCR-side search columns.
Both are separate because both are *speculative* and are kept
separate from the columns that describe what was actually observed.

Command line usage::

    .venv\\Scripts\\python.exe -m cpd.postprocess \\
        --csv data/processed/compounds_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Optional

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from cpd.chem import validate_and_enrich
from cpd.render import is_macrocycle_intact

DEFAULT_CSV = Path("data/processed/compounds_combined.csv")

# --- the quality layer ---------------------------------------------------

# Plausibility windows for the three numeric assay fields.  A value
# outside its window is almost always a mis-read digit rather than a
# real measurement, so the row is flagged instead of being silently
# trusted.  ``kd_nm`` has a lower bound because a binding constant of
# zero is not a measurement but a rounding artefact; ``rt_min`` and
# ``ms_mz`` only need to be non-negative.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "kd_nm": (0.0001, 100000.0),
    "rt_min": (0.0, 30.0),
    "ms_mz": (0.0, 5000.0),
}

# --- the text layer ------------------------------------------------------

# How many lost brackets still leave a recognisable string.  The
# buckets are the ones used in the Phase 2/3 audits.
DIFFICULTY_BUCKETS: tuple[tuple[int, str], ...] = (
    (0, "balanced"),
    (2, "light"),
    (5, "medium"),
    (10, "heavy"),
)
DIFFICULTY_OVERFLOW = "extreme"

STRUCTURE_COLUMNS: tuple[str, ...] = (
    "decimer_largest_ring_size",
    "decimer_macrocycle_intact",
    "decimer_num_rings",
)
QUALITY_COLUMNS: tuple[str, ...] = ("data_quality", "data_quality_flags")
TEXT_COLUMNS: tuple[str, ...] = ("ocr_bracket_deficit", "ocr_difficulty")

DERIVED_COLUMNS: tuple[str, ...] = (
    STRUCTURE_COLUMNS + QUALITY_COLUMNS + TEXT_COLUMNS
)


def _num(value: object) -> Optional[float]:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def difficulty_bucket(deficit: int) -> str:
    """Map a bracket deficit onto its difficulty bucket."""
    for limit, name in DIFFICULTY_BUCKETS:
        if deficit <= limit:
            return name
    return DIFFICULTY_OVERFLOW


def bracket_deficit(smiles: str) -> int:
    """Number of unclosed brackets in ``smiles``.

    Counts the surplus of ``(`` over ``)`` and of ``[`` over ``]``.
    A positive number is what the recogniser lost: the printed cell
    had balanced brackets and the OCR output does not.
    """
    if not smiles:
        return 0
    surplus = smiles.count("(") - smiles.count(")")
    surplus += smiles.count("[") - smiles.count("]")
    return max(0, surplus)


def structure_columns(rows: Iterable[dict]) -> list[dict]:
    """Add the ring statistics of the DECIMER structure to every row."""
    out: list[dict] = []
    for row in rows:
        new = dict(row)
        canon = (new.get("canonical_smiles_decimer") or "").strip()
        if not canon:
            new["decimer_largest_ring_size"] = ""
            new["decimer_macrocycle_intact"] = ""
            new["decimer_num_rings"] = ""
            out.append(new)
            continue
        info = validate_and_enrich(canon)
        new["decimer_largest_ring_size"] = (
            info.largest_ring_size if info else 0
        )
        new["decimer_num_rings"] = info.num_rings if info else 0
        new["decimer_macrocycle_intact"] = str(is_macrocycle_intact(canon))
        out.append(new)
    return out


def quality_columns(rows: Iterable[dict]) -> list[dict]:
    """Add the data-quality flag list and its summary word to every row."""
    out: list[dict] = []
    for row in rows:
        new = dict(row)
        flags: list[str] = []
        for field, (low, high) in NUMERIC_RANGES.items():
            value = _num(new.get(field))
            if value is None:
                continue
            if value < low:
                flags.append(
                    f"{field} below range ({value:.4f} < {low:.4f})"
                )
            elif value > high:
                flags.append(
                    f"{field} above range ({value:.4f} > {high:.4f})"
                )
        range_flags = list(flags)
        if str(new.get("decimer_macrocycle_intact")) == "False":
            flags.append("decimer_macrocycle_broken")
        new["data_quality_flags"] = "; ".join(flags)
        if range_flags:
            new["data_quality"] = "partial"
        elif flags:
            new["data_quality"] = "macrocycle_broken"
        else:
            new["data_quality"] = "ok"
        out.append(new)
    return out


def text_columns(rows: Iterable[dict]) -> list[dict]:
    """Add the bracket deficit of the printed text to every row."""
    out: list[dict] = []
    for row in rows:
        new = dict(row)
        smiles = (new.get("smiles") or "").strip()
        if not smiles:
            new["ocr_bracket_deficit"] = ""
            new["ocr_difficulty"] = "no_ocr"
            out.append(new)
            continue
        deficit = bracket_deficit(smiles)
        new["ocr_bracket_deficit"] = str(deficit)
        new["ocr_difficulty"] = difficulty_bucket(deficit)
        out.append(new)
    return out


def apply_derived_columns(rows: Iterable[dict]) -> list[dict]:
    """Compute every derived column of the combined CSV.

    The order matters: the quality layer reads the macrocycle gate that
    the structure layer computes.
    """
    return text_columns(quality_columns(structure_columns(rows)))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict], path: Path, columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"missing CSV: {args.csv}", file=sys.stderr)
        return 1
    rows = read_rows(args.csv)
    if not rows:
        print("input CSV is empty", file=sys.stderr)
        return 1

    columns = list(rows[0].keys())
    for column in DERIVED_COLUMNS:
        if column not in columns:
            columns.append(column)

    processed = apply_derived_columns(rows)
    write_rows(processed, args.csv, columns)

    from collections import Counter

    print(f"rows            : {len(processed)}")
    print("data_quality    :")
    for key, count in Counter(
        r.get("data_quality") or "ok" for r in processed
    ).most_common():
        print(f"   {key:18s} {count:4d}")
    print("ocr_difficulty  :")
    for key, count in Counter(
        r.get("ocr_difficulty") or "no_ocr" for r in processed
    ).most_common():
        print(f"   {key:18s} {count:4d}")
    print(f"updated {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

