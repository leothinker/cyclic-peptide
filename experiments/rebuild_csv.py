"""Rebuild the combined CSV for one patent, reusing the DECIMER columns.

The DECIMER structure pass is expensive and its output is already in
the shipped CSV.  This script re-runs the *OCR* half of the pipeline
with a chosen route and rebuilds the CSV from it, keeping the DECIMER
columns of the existing file unchanged.

That makes it possible to ship the printed-SMILES improvement without
re-running the part of the pipeline that has been shelved.

The rebuilt file has the full column schema of :mod:`cpd.schema`: the
pipeline columns, the repair layer of :mod:`cpd.repair_policy` and the
derived layer of :mod:`cpd.postprocess`.  The OCR-search layer of
:mod:`cpd.mass_constrained_search` is left empty unless ``--ocr-search``
is given, because it is slow and speculative.

Usage::

    .venv\\Scripts\\python.exe experiments/rebuild_csv.py \
        --profile WO2025162428 \
        --route validated \
        --base data/processed/compounds_combined.csv \
        --archive data/processed/archive/compounds_combined_production_route.csv \
        --out data/processed/compounds_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import main as pipeline  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.postprocess import apply_derived_columns  # noqa: E402
from cpd.repair_policy import process_rows  # noqa: E402
from cpd.schema import combined_columns  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the combined CSV for one patent.",
    )
    parser.add_argument(
        "--profile", default="WO2025162428",
        help="Patent profile (default: %(default)s).",
    )
    parser.add_argument(
        "--route", default=None,
        help="Printed-SMILES route. Defaults to the profile's own route.",
    )
    parser.add_argument(
        "--base", default=None,
        help="Existing combined CSV whose DECIMER columns are reused.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Where to write the rebuilt CSV.",
    )
    parser.add_argument(
        "--cells", default=None,
        help="Directory for the cropped structure images.",
    )
    parser.add_argument(
        "--archive", default=None,
        help="Copy the base CSV here before overwriting it.",
    )
    parser.add_argument(
        "--ocr-search", action="store_true",
        help="Also run the mass-constrained OCR repair search and fill the "
             "search columns.  Slow (several seconds per row).",
    )
    parser.add_argument(
        "--from-csv", default=None,
        help="Skip the extraction and rebuild the deliverable from an "
             "existing combined CSV.  Useful when the OCR half has "
             "already been run and only the derived layers changed.",
    )
    return parser.parse_args(argv)


def _load_decimer_columns(base: Path) -> dict[str, dict]:
    """Read the DECIMER-derived columns out of an existing CSV."""
    if not base or not base.exists():
        return {}
    with base.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["cmpd_id"]: {
            "raw": row.get("smiles_decimer", ""),
            "canon": row.get("canonical_smiles_decimer", ""),
            "min_conf": row.get("decimer_min_conf", ""),
            "low_conf": row.get("decimer_low_conf_count", ""),
            "render": row.get("decimer_render_path", ""),
        }
        for row in rows
        if row.get("cmpd_id")
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = get_profile(args.profile)
    route = args.route or profile.smiles_route

    img_dir = REPO / "data" / "raw" / args.profile / "golden_tables"
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    cells_dir = Path(args.cells) if args.cells else out_path.parent / "cells"
    if not cells_dir.is_absolute():
        cells_dir = REPO / cells_dir

    base_path = Path(args.base) if args.base else None
    if base_path and not base_path.is_absolute():
        base_path = REPO / base_path
    decimer = _load_decimer_columns(base_path) if base_path else {}

    if args.from_csv:
        source = Path(args.from_csv)
        if not source.is_absolute():
            source = REPO / source
        print(f"[1/3] reading the existing extraction from {source} ...")
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        print(f"   -> {len(rows)} rows")
    else:
        cells_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[1/3] extracting {args.profile} "
            f"(route={route}) from {img_dir} ..."
        )
        compounds, assays, _failures = pipeline._extract_all(
            img_dir,
            cells_dir,
            rec_top_k=0,
            smiles_route=route,
            profile_name=args.profile,
        )
        print(f"   -> {len(compounds)} compounds, {len(assays)} assays")

        dec_raw = {cid: v["raw"] for cid, v in decimer.items()}
        dec_canon = {cid: v["canon"] for cid, v in decimer.items()}
        dec_min = {
            cid: float(v["min_conf"])
            for cid, v in decimer.items()
            if v["min_conf"] not in ("", None)
        }
        dec_low = {
            cid: int(v["low_conf"])
            for cid, v in decimer.items()
            if v["low_conf"] not in ("", None)
        }
        dec_render = {
            cid: v["render"] for cid, v in decimer.items() if v["render"]
        }

        print("[2/3] building the combined rows ...")
        rows = pipeline._build_combined_rows(
            compounds, assays, dec_raw, dec_canon, dec_min, dec_low, dec_render,
            extra_columns=profile.extra_assay_fields,
        )
    print("   applying the repair policy ...")
    rows = process_rows(rows)
    print("   applying the derived columns ...")
    rows = apply_derived_columns(rows)

    if args.ocr_search:
        print("   running the mass-constrained OCR search (slow) ...")
        from cpd.ocr_search import run_search  # type: ignore[import-not-found]

        rows = run_search(rows)

    if args.archive and base_path and base_path.exists():
        archive = Path(args.archive)
        if not archive.is_absolute():
            archive = REPO / archive
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_path, archive)
        print(f"   archived the previous CSV to {archive}")

    print(f"[3/3] writing {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = pipeline._write_csv(
        rows, out_path, combined_columns(profile.extra_assay_fields),
    )

    n_valid = sum(1 for r in rows if r["is_valid"] == "True")
    n_canon = sum(1 for r in rows if r["canonical_smiles"])
    n_best = sum(1 for r in rows if r["smiles_best"])
    n_mass = sum(1 for r in rows if r.get("ms_check_verdict") == "match")
    n_printed = sum(1 for r in rows if (r.get("smiles") or "").strip())
    print()
    print(f"rows                : {n}")
    print(f"printed SMILES      : {n_printed}")
    print(f"OCR RDKit-valid     : {n_valid}")
    print(f"canonical SMILES    : {n_canon}")
    print(f"smiles_best present : {n_best}")
    print(f"mass gate matched   : {n_mass}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
