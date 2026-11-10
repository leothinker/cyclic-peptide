"""Run the profile-driven pipeline over WO2022251372.

This is the end-to-end check of the adapter layer: the same extractor,
the same printed-SMILES route, the same repair chain and the same
merge code that serve the original patent, driven only by the
``WO2022251372`` profile.

The output goes to ``data/audit/phase6/`` and never touches the
formal combined CSV.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_wo2022251372_extract.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.merge import merge_compound_records  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402

OUT = REPO / "data" / "audit" / "phase6"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract WO2022251372 with the profile-driven pipeline.",
    )
    parser.add_argument(
        "--patent", default="WO2022251372",
        help="Patent folder under data/raw (default: %(default)s).",
    )
    parser.add_argument(
        "--profile", default="WO2022251372",
        help="Profile to use (default: %(default)s).",
    )
    parser.add_argument(
        "--out", default=str(OUT / "wo2022251372_extraction.csv"),
        help="Where to write the extraction CSV.",
    )
    parser.add_argument(
        "--cells", default=str(OUT / "cells"),
        help="Directory for the cropped structure images.",
    )
    parser.add_argument(
        "--route", default=None,
        help="Printed-SMILES route. Defaults to the profile's own route, "
             "which is what the committed output was produced with.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = get_profile(args.profile)
    img_dir = REPO / "data" / "raw" / args.patent / "golden_tables"
    cells_dir = Path(args.cells)
    if not cells_dir.is_absolute():
        cells_dir = REPO / cells_dir
    cells_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    )

    route = args.route or profile.smiles_route
    extractor = RapidOcrTableExtractor(profile=profile, smiles_route=route)
    compounds: list = []
    assays: list = []
    prev_columns: list = []
    for image_path in images:
        page_compounds, page_assays, columns = extractor.extract_records(
            image_path, cells_dir=cells_dir, prev_columns=prev_columns,
        )
        if columns:
            prev_columns = columns
        compounds.extend(page_compounds)
        assays.extend(page_assays)
        print(
            f"  {image_path.name}: +{len(page_compounds)} cmpd, "
            f"+{len(page_assays)} assay"
        )
    compounds.extend(extractor.flush_pending())

    merged = merge_compound_records(
        compounds, prefer_structure=True,
    )
    assay_by_id = {a.cmpd_id: a for a in assays}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grade_names = [field.name for field in profile.assay_fields]
    fieldnames = [
        "cmpd_id", "smiles", "canonical_smiles", "is_valid",
        "molecular_weight", "heavy_atom_count", "structure_image",
        "source_image",
    ] + grade_names

    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cmpd_id in sorted(merged, key=lambda k: int(k) if k.isdigit() else 0):
            record = merged[cmpd_id]
            row = {
                "cmpd_id": cmpd_id,
                "smiles": record.smiles,
                "canonical_smiles": record.canonical_smiles,
                "is_valid": record.is_valid,
                "molecular_weight": record.molecular_weight,
                "heavy_atom_count": record.heavy_atom_count,
                "structure_image": record.structure_image,
                "source_image": record.source_image,
            }
            assay = assay_by_id.get(cmpd_id)
            if assay is not None:
                extra = dict(assay.extra)
                for name in grade_names:
                    row[name] = extra.get(name, "")
            else:
                for name in grade_names:
                    row[name] = ""
            writer.writerow(row)

    n_valid = sum(1 for r in merged.values() if r.is_valid)
    n_structure = sum(1 for r in merged.values() if r.structure_image)
    summary = {
        "patent": args.patent,
        "profile": args.profile,
        "route": route,
        "n_images": len(images),
        "n_compounds": len(merged),
        "n_rdkit_valid": n_valid,
        "n_with_structure": n_structure,
        "n_assays": len(assay_by_id),
    }
    (OUT / "wo2022251372_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print()
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
