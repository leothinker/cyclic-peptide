"""End-to-end verification of the new ``padded`` OCR route.

The route was measured on WO2022251372 (Table 1a).  This script runs
the *shipped* code -- ``PrintedSmilesOCR(route="padded")`` plus the
shipped repair chain -- over the printed cells of both patents and
reports, per patent:

* bracket balance,
* RDKit parseability,
* the edit distance to the hand transcription (where one exists).

The old patent is included on purpose: the margin must not be applied
there, so this run also shows what the new route does when it is
pointed at the wrong patent.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_padded_route_verification.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles, close_unbalanced_brackets  # noqa: E402
from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402
from cpd.ocr_rewrite import rewrite_oxo  # noqa: E402

from wo2022251372_references import REFERENCE_LINES, edit_distance  # noqa: E402

OLD_CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OLD_REFERENCES = REPO / "data" / "audit" / "phase2" / "ocr_references.json"
OLD_META = OLD_CELLS / "cells_text_meta.csv"
NEW_CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase5" / "padded_route_verification"


def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _balanced(smiles: str) -> bool:
    return (
        smiles.count("(") == smiles.count(")")
        and smiles.count("[") == smiles.count("]")
    )


def _load_old_references() -> dict[str, str]:
    """Hand transcriptions of WO2025162428, keyed by compound id."""
    data = json.loads(OLD_REFERENCES.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, dict):
            text = value.get("smiles") or value.get("reference")
            if text:
                out[key] = text
    return out


def _old_compound_ids() -> dict[str, str]:
    """Map a cell file stem to its compound id."""
    mapping: dict[str, str] = {}
    if not OLD_META.exists():
        return mapping
    # The file carries a UTF-8 BOM, so read it as utf-8-sig.
    with OLD_META.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            crop = row.get("crop_path") or ""
            stem = Path(crop).stem
            cmpd_id = (row.get("cmpd_id") or "").strip()
            if stem and cmpd_id:
                mapping[stem] = cmpd_id
    return mapping


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    routes = ("production", "validated", "padded")
    readers = {name: PrintedSmilesOCR(route=name) for name in routes}
    old_refs = _load_old_references()
    old_ids = _old_compound_ids()

    rows: list[dict] = []
    for label, directory in (("old", OLD_CELLS), ("new", NEW_CELLS)):
        for path in sorted(directory.glob("*.png")):
            cell = cv2.imread(str(path))
            if cell is None:
                continue

            if label == "new":
                compound_id = int(path.stem.split("cmpd")[-1])
                reference = "".join(REFERENCE_LINES.get(compound_id, ()))
            else:
                compound_id = old_ids.get(path.stem, "")
                reference = old_refs.get(compound_id, "")

            for route in routes:
                raw = readers[route].read_cell(cell) or ""

                oxo, n_oxo = rewrite_oxo(raw)
                repaired = auto_repair_smiles(oxo)
                closed = close_unbalanced_brackets(repaired) or repaired

                rows.append(
                    {
                        "patent": label,
                        "cell": path.stem,
                        "compound_id": compound_id,
                        "route": route,
                        "has_reference": bool(reference),
                        "raw_ocr": raw,
                        "raw_balanced": _balanced(raw),
                        "raw_parses": _parses(raw),
                        "rewrite_oxo_fixes": n_oxo,
                        "after_repair": closed,
                        "after_repair_balanced": _balanced(closed),
                        "after_repair_parses": _parses(closed),
                        "edit_to_reference": (
                            edit_distance(reference, raw) if reference else ""
                        ),
                        "edit_to_reference_after_repair": (
                            edit_distance(reference, closed)
                            if reference
                            else ""
                        ),
                    }
                )

    fields = list(rows[0].keys())
    with (OUT / "padded_route_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {}
    for label in ("old", "new"):
        for route in routes:
            subset = [
                r
                for r in rows
                if r["patent"] == label and r["route"] == route
            ]
            with_ref = [r for r in subset if r["has_reference"]]
            entry = {
                "cells": len(subset),
                "balanced_raw": sum(1 for r in subset if r["raw_balanced"]),
                "parses_raw": sum(1 for r in subset if r["raw_parses"]),
                "balanced_after_repair": sum(
                    1 for r in subset if r["after_repair_balanced"]
                ),
                "parses_after_repair": sum(
                    1 for r in subset if r["after_repair_parses"]
                ),
                "cells_with_reference": len(with_ref),
                "edit_to_reference": [
                    r["edit_to_reference"] for r in with_ref
                ],
                "edit_to_reference_after_repair": [
                    r["edit_to_reference_after_repair"] for r in with_ref
                ],
            }
            if with_ref:
                entry["total_edit_raw"] = sum(entry["edit_to_reference"])
                entry["total_edit_after_repair"] = sum(
                    entry["edit_to_reference_after_repair"]
                )
            summary.setdefault(label, {})[route] = entry

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
