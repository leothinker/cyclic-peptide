"""Should the ``padded`` route upscale the band before recognising?

The 6 px margin was measured in ``phase4_wo2022251372_candidate_config``
with ``scale=1`` -- no resize at all -- and that is where the
"21/28 bracket balanced, 17/28 RDKit-parseable" headline comes from.
The shipped ``validated`` route, however, upscales 2x, and the new
``padded`` route inherited that.

This script runs both readings over all 28 Table 1a cells with the
full repair chain and reports:

* bracket balance and RDKit parseability before and after repair,
* the edit distance to the six hand transcriptions,
* the same for the old patent's five references, so the choice cannot
  silently favour one patent.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_padded_scale_ab.py
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
from cpd.ocr.printed_smiles import (  # noqa: E402
    PADDED_MARGIN_PX,
    WHITE,
    english_recogniser_only,
    ink_bands,
    text_span,
)
from cpd.ocr_rewrite import rewrite_oxo  # noqa: E402

from wo2022251372_references import REFERENCE_LINES, edit_distance  # noqa: E402

OLD_CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OLD_META = OLD_CELLS / "cells_text_meta.csv"
OLD_REFERENCES = REPO / "data" / "audit" / "phase2" / "ocr_references.json"
NEW_CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase5" / "padded_scale_ab"

PAD_TOP = 14
PAD_BOTTOM = 15


def _read(cell, engine, scale: int) -> str:
    """Recognise every band, optionally upscaled after the margin."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        band = row[:, x1:x2 + 1]
        if band.size == 0:
            continue
        band = cv2.copyMakeBorder(
            band,
            0,
            0,
            PADDED_MARGIN_PX,
            PADDED_MARGIN_PX,
            cv2.BORDER_CONSTANT,
            value=WHITE,
        )
        if scale != 1:
            band = cv2.resize(
                band,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        result = engine(band, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        text = (txts[0] or "").strip() if txts else ""
        if text:
            parts.append(text)
    return "".join(parts)


def _parses(text: str) -> bool:
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _balanced(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


def _repair(raw: str) -> str:
    oxo, _ = rewrite_oxo(raw)
    repaired = auto_repair_smiles(oxo)
    return close_unbalanced_brackets(repaired) or repaired


def _old_refs() -> dict[str, str]:
    data = json.loads(OLD_REFERENCES.read_text(encoding="utf-8"))
    out = {}
    for key, value in data.items():
        text = value if isinstance(value, str) else value.get("smiles")
        if text:
            out[key] = text
    return out


def _old_ids() -> dict[str, str]:
    mapping = {}
    with OLD_META.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cmpd_id = (row.get("cmpd_id") or "").strip()
            stem = Path(row.get("crop_path") or "").stem
            if cmpd_id and stem:
                mapping[stem] = cmpd_id
    return mapping


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)
    engine = english_recogniser_only()
    old_refs = _old_refs()
    old_ids = _old_ids()

    rows = []
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

            for scale in (1, 2):
                raw = _read(cell, engine, scale)
                fixed = _repair(raw)
                rows.append(
                    {
                        "patent": label,
                        "cell": path.stem,
                        "compound_id": compound_id,
                        "scale": scale,
                        "has_reference": bool(reference),
                        "raw_balanced": _balanced(raw),
                        "raw_parses": _parses(raw),
                        "after_repair_balanced": _balanced(fixed),
                        "after_repair_parses": _parses(fixed),
                        "edit": (
                            edit_distance(reference, raw) if reference else ""
                        ),
                        "edit_after_repair": (
                            edit_distance(reference, fixed)
                            if reference
                            else ""
                        ),
                    }
                )

    with (OUT / "padded_scale_ab.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for label in ("old", "new"):
        summary[label] = {}
        for scale in (1, 2):
            subset = [
                r
                for r in rows
                if r["patent"] == label and r["scale"] == scale
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
                "edits": [r["edit"] for r in with_ref],
                "edits_after_repair": [
                    r["edit_after_repair"] for r in with_ref
                ],
            }
            if with_ref:
                entry["total_edit"] = sum(entry["edits"])
                entry["total_edit_after_repair"] = sum(
                    entry["edits_after_repair"]
                )
            summary[label][f"scale{scale}"] = entry

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
