"""Snapshot the extractor output for one patent, for refactor regression.

The patent-profile refactor (``src/cpd/parsers/profiles.py``) must not
change a single byte of the extracted data for the patents that are
already shipped.  This script runs the *real* extractor over every
golden table of a patent and writes a canonical JSON snapshot:

* the compound rows (id, OCR text, repaired text, RDKit flags),
* the assay rows,
* the detected column geometry per image.

Run it before and after the refactor and diff the two files.

Usage::

    .venv\\Scripts\\python.exe experiments/phase6_profile_snapshot.py \
        --patent WO2025162428 --route production --out before.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot extractor output for regression testing.",
    )
    parser.add_argument(
        "--patent", default="WO2025162428",
        help="Patent folder under data/raw (default: %(default)s).",
    )
    parser.add_argument(
        "--route", default=None,
        help="Printed-SMILES route. Defaults to the route the profile "
             "declares for the patent.",
    )
    parser.add_argument(
        "--profile", default=None,
        help="Table profile name (default: the profile registered for the patent).",
    )
    parser.add_argument(
        "--out", required=True, help="Where to write the snapshot JSON.",
    )
    parser.add_argument(
        "--rec-top-k", type=int, default=0,
        help="Per-character top-k handed to the recogniser (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    img_dir = REPO / "data" / "raw" / args.patent / "golden_tables"
    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    )

    kwargs = {"rec_top_k": args.rec_top_k}
    if args.route:
        kwargs["smiles_route"] = args.route
    if args.profile:
        from cpd.parsers.profiles import get_profile

        kwargs["profile"] = get_profile(args.profile)

    extractor = RapidOcrTableExtractor(**kwargs)

    compounds: dict[str, dict] = {}
    assays: dict[str, dict] = {}
    geometry: dict[str, list] = {}
    prev_columns = []
    for image_path in images:
        page_compounds, page_assays, columns = extractor.extract_records(
            image_path, cells_dir=None, prev_columns=prev_columns,
        )
        prev_columns = columns
        for record in page_compounds:
            compounds.setdefault(record.cmpd_id, {
                "smiles": record.smiles,
                "canonical_smiles": record.canonical_smiles,
                "molecular_weight": record.molecular_weight,
                "heavy_atom_count": record.heavy_atom_count,
                "is_valid": record.is_valid,
                "source_image": record.source_image,
                "ocr_avg_entropy": record.ocr_avg_entropy,
                "ocr_min_top1": record.ocr_min_top1,
                "ocr_low_conf_count": record.ocr_low_conf_count,
            })
        for record in page_assays:
            assays.setdefault(record.cmpd_id, dict(record.to_dict()))
        geometry[image_path.name] = [
            [c.name, c.x1, c.x2, c.is_image] for c in columns
        ]

    snapshot = {
        "patent": args.patent,
        # The route that was actually used, which is the one the
        # profile declares when ``--route`` is not given.
        "route": extractor.smiles_route,
        "n_images": len(images),
        "compounds": {k: compounds[k] for k in sorted(compounds)},
        "assays": {k: assays[k] for k in sorted(assays)},
        "geometry": {k: geometry[k] for k in sorted(geometry)},
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"{args.patent} route={extractor.smiles_route}: "
        f"{len(compounds)} compounds, {len(assays)} assays, "
        f"{len(images)} images -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
