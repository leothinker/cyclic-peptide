"""Check that the delivery repository can do what it is meant to do.

The delivery repository (``drug_ocr``) is the one that was handed to
the supervisor.  It has to answer three questions, and this script
answers them from the delivery repository's *own* code and its *own*
virtual environment, so that the answer does not depend on anything
left behind in the working repository.

1. Does the delivery repository's ``cpd`` package resolve to the
   delivery repository's own ``src`` tree?
2. Does the profile of WO2025162428 select a route that reads the
   printed SMILES correctly?
3. Are the three hand-checked compounds of WO2025162428 read
   character-exactly by that route?

The script must be **run from inside the delivery repository** with
that repository's own interpreter, otherwise it measures the working
repository by accident.  It prints what it measured and writes a JSON
report next to itself.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_delivery_check.py
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

import cpd  # noqa: E402
from cpd.chem import auto_repair_smiles  # noqa: E402
from cpd.ocr.printed_smiles import ROUTES, PrintedSmilesOCR  # noqa: E402
from cpd.parsers.profiles import available_profiles, get_profile  # noqa: E402

sys.path.insert(0, str(REPO / "experiments"))
from phase7_route_verify import (  # noqa: E402
    PAGES,
    REFERENCES,
    _read_from_page,
    edit_distance,
)

OUT = REPO / "data" / "audit" / "phase9"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT / "delivery_check.json",
        help="Where to write the JSON report.",
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "repository": str(REPO),
        "cpd_package": str(Path(cpd.__file__).resolve()),
        "cpd_is_local": str(Path(cpd.__file__).resolve()).startswith(
            str(REPO.resolve())
        ),
        "profiles": list(available_profiles()),
        "routes": list(ROUTES),
    }

    profile = get_profile("WO2025162428")
    report["WO2025162428_route"] = profile.smiles_route
    report["WO2022251372_route"] = get_profile("WO2022251372").smiles_route

    import cv2

    reader = PrintedSmilesOCR(route=profile.smiles_route)
    results = {}
    for compound_id, reference in REFERENCES.items():
        page = REPO / "data" / "raw" / "WO2025162428" / "golden_tables" / PAGES[compound_id]
        text = _read_from_page(reader, page, compound_id)
        if text is None:
            results[compound_id] = {"error": f"cell not found on {page.name}"}
            continue
        # Two numbers, because they answer two different questions.
        # The raw reading is what the recogniser returned; the repaired
        # one is what the pipeline writes into the CSV, after
        # ``auto_repair_smiles``.  Quoting only one of them is how a
        # reader ends up with two different edit distances for the
        # same compound.
        repaired = auto_repair_smiles(text)
        distance = edit_distance(reference, text)
        repaired_distance = edit_distance(reference, repaired)
        results[compound_id] = {
            "edit_distance_raw": distance,
            "edit_distance_repaired": repaired_distance,
            "exact": repaired_distance == 0,
            "length": len(repaired),
            "reference_length": len(reference),
            "source": page.name,
        }
    report["references"] = results
    report["references_exact"] = sum(
        1 for value in results.values() if value.get("exact")
    )
    report["references_total"] = len(results)

    out_path = args.out if args.out.is_absolute() else REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\n-> {out_path}")
    ok = bool(report["cpd_is_local"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
