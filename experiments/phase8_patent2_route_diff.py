"""Where do the two WO2022251372 routes differ, and why not more?

``phase8_patent2_route_compare.py`` reports that the ``padded`` and
``padded_vote`` routes give the same result on the six transcribed
compounds.  This script finds out **which reading each route produces
per table**, because the merged compound is not necessarily the
reading of Table 1a: the patent prints every compound twice, and the
merge keeps the first *valid* record it sees.

For each compound it prints the reading of Table 1, the reading of
Table 1a, and the merged result, for both routes.

Usage::

    .venv\\Scripts\\python.exe experiments/phase8_patent2_route_diff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from cpd.merge import merge_compound_records  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import (  # noqa: E402
    RapidOcrTableExtractor,
)
from wo2022251372_references import (  # noqa: E402
    REFERENCE_LINES,
    edit_distance,
)

OUT = REPO / "data" / "audit" / "phase8"
IMAGE_DIR = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"

ROUTES = ("padded", "padded_vote")


def _extract(route: str):
    profile = get_profile("WO2022251372")
    extractor = RapidOcrTableExtractor(
        rec_top_k=0,
        smiles_route=route,
        profile=profile,
    )
    cells_dir = OUT / "cells_diff"
    cells_dir.mkdir(parents=True, exist_ok=True)
    compounds = []
    prev_columns = []
    for path in sorted(IMAGE_DIR.iterdir()):
        page_compounds, _assays, columns = extractor.extract_records(
            path,
            cells_dir=cells_dir,
            prev_columns=prev_columns,
        )
        if columns:
            prev_columns = columns
        compounds.extend(page_compounds)
    compounds.extend(extractor.flush_pending())
    return compounds


def main() -> int:
    for route in ROUTES:
        compounds = _extract(route)
        merged = merge_compound_records(compounds, prefer_structure=True)
        print(f"\n=========== {route} ===========")
        for compound_id, lines in REFERENCE_LINES.items():
            reference = "".join(lines)
            readings = [
                c for c in compounds if str(c.cmpd_id) == str(compound_id)
            ]
            merged_text = merged.get(str(compound_id))
            merged_text = merged_text.smiles if merged_text else ""
            print(f"\n  compound {compound_id}")
            for record in readings:
                text = record.smiles or ""
                source = record.source_image or ""
                print(
                    f"    {source:<12} valid={record.is_valid!s:<5} "
                    f"edit={edit_distance(reference, text):<3} "
                    f"len={len(text):<4} {text[:70]!r}"
                )
            print(
                f"    {'MERGED':<12} "
                f"edit={edit_distance(reference, merged_text):<3} "
                f"len={len(merged_text):<4} {merged_text[:70]!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
