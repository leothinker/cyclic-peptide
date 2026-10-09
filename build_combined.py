"""Quick combined-CSV builder.

Reads the already-written ``compounds.csv`` + ``assays.csv`` (no re-OCR),
optionally re-runs DECIMER over the cropped structures, and writes
``compounds_combined.csv``.

Use this after a main.py run that died at the combined-CSV step so we
don't have to pay the full RapidOCR + DECIMER cost again.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Iterable

# UTF-8 console (emoji-safe)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from cpd.decimer import is_available, predict_smiles
from cpd.chem import validate_and_enrich
from cpd.parsers.table_extractor import AssayRecord, CompoundRecord
import main  # reuse _build_combined_rows + COMBINED_COLUMNS + _write_csv


log = logging.getLogger("cpd")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _rows_to_compounds(rows: Iterable[dict]) -> list[CompoundRecord]:
    """Reconstruct CompoundRecord objects from a compounds.csv dict stream."""
    out: list[CompoundRecord] = []
    for row in rows:
        try:
            mw = float(row["molecular_weight"]) if row.get("molecular_weight") else None
        except ValueError:
            mw = None
        try:
            ha = int(row["heavy_atom_count"]) if row.get("heavy_atom_count") else None
        except ValueError:
            ha = None
        out.append(CompoundRecord(
            cmpd_id=row["cmpd_id"],
            smiles=row.get("smiles") or None,
            canonical_smiles=row.get("canonical_smiles") or None,
            molecular_weight=mw,
            heavy_atom_count=ha,
            is_valid=(row.get("is_valid") == "True") if row.get("is_valid") else None,
            structure_image=row.get("structure_image") or None,
            source_image=row.get("source_image") or None,
        ))
    return out


def _rows_to_assays(rows: Iterable[dict]) -> dict[str, AssayRecord]:
    """Reconstruct AssayRecord dict (keyed by cmpd_id) from an assays.csv stream."""
    out: dict[str, AssayRecord] = {}
    for row in rows:
        def _f(key: str) -> float | None:
            v = row.get(key)
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                return None
        out[row["cmpd_id"]] = AssayRecord(
            cmpd_id=row["cmpd_id"],
            kd_nm=_f("kd_nm"),
            rt_min=_f("rt_min"),
            ms_mz=_f("ms_mz"),
            lcms_method=row.get("lcms_method") or None,
            ms_polarity=row.get("ms_polarity") or None,
            source_image=row.get("source_image") or None,
        )
    return out


def _run_decimer(compounds: list[CompoundRecord], out_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    raw: dict[str, str] = {}
    canon: dict[str, str] = {}
    if not is_available():
        print("[!] DECIMER not installed -- smiles_decimer will be empty")
        return raw, canon
    targets = [c for c in compounds if c.structure_image]
    print(f"[2/3] running DECIMER on {len(targets)} structures ...")
    for i, c in enumerate(targets, start=1):
        smi = predict_smiles(out_dir / c.structure_image)  # type: ignore[operator]
        if not smi:
            continue
        raw[c.cmpd_id] = smi
        chem = validate_and_enrich(smi)
        if chem is not None:
            canon[c.cmpd_id] = chem.canonical_smiles
        if i % 25 == 0 or i == len(targets):
            print(f"      [{i}/{len(targets)}] parsed", end="\r")
    print()
    print(f"      got SMILES for {len(raw)} / {len(targets)} structures")
    return raw, canon


def main_() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed",
                         help="directory containing compounds.csv / assays.csv / cells/")
    parser.add_argument("--skip-decimer", action="store_true",
                         help="don't run DECIMER, just merge OCR + assays")
    args = parser.parse_args()

    proc = Path(args.processed)
    compounds_csv = proc / "compounds.csv"
    assays_csv = proc / "assays.csv"
    if not compounds_csv.exists() or not assays_csv.exists():
        sys.exit(f"need both {compounds_csv} and {assays_csv}")

    print(f"[1/3] reading {compounds_csv.name} + {assays_csv.name} ...")
    compounds = _rows_to_compounds(_read_csv(compounds_csv))
    assays = _rows_to_assays(_read_csv(assays_csv))
    print(f"      {len(compounds)} cmpd, {len(assays)} assay")

    if args.skip_decimer:
        decimer_raw, decimer_canon = {}, {}
    else:
        decimer_raw, decimer_canon = _run_decimer(compounds, proc)

    print(f"[3/3] writing compounds_combined.csv ...")
    rows = main._build_combined_rows(compounds, assays, decimer_raw, decimer_canon)
    n = main._write_csv(rows, proc / "compounds_combined.csv", main.COMBINED_COLUMNS)
    n_dec = len(decimer_raw)
    n_agree = sum(1 for r in rows if r["smiles_agree"] == "True")
    print("=" * 55)
    print(f"combined: {n} rows  |  DECIMER: {n_dec}  |  agree: {n_agree}")
    print("=" * 55)


if __name__ == "__main__":
    main_()
