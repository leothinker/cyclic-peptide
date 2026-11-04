"""Can the existing repair chain rescue the WO2022251372 OCR output?

WO2025162428 and WO2022251372 are printed by two different offices with
two different fonts, so the edit model measured on the first patent
(``cpd.ocr_edit_model``: mostly lost brackets, ``(=`` collapsed into
``-``) is not guaranteed to hold for the second one.

This script therefore measures, on all 28 Table 1a compounds:

* the raw OCR text of the current production route and of the best
  configuration found by the recogniser matrix;
* what the *existing* repair chain does to that text;
* whether the repaired string parses in RDKit;
* and, for the three compounds whose printed text was transcribed by
  hand, whether the repair recovers the reference string or merely
  produces some parseable molecule.

The last point is the one that matters: "parses" is a far weaker claim
than "is the printed molecule".

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_recovery_test.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import (
    EngineType,
    ModelType,
    OCRVersion,
    TaskType,
)
from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles, close_unbalanced_brackets  # noqa: E402
from cpd.ocr.printed_smiles import ink_bands, text_span  # noqa: E402
from cpd.ocr_rewrite import rewrite_oxo  # noqa: E402
from cpd.ocr_edit_model import normalise_lexical  # noqa: E402


CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
from wo2022251372_references import (  # noqa: E402
    CELL_FILES,
    REFERENCE_LINES,
    cer,
    edit_distance,
)

OUT = REPO / "data" / "audit" / "phase4" / "wo2022251372" / "recovery_test"



PAD_TOP = 14
PAD_BOTTOM = 15






def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _read_bands(cell: np.ndarray, engine: RapidOCR, scale: int) -> str:
    """The validated route geometry with a configurable model and scale."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts: list[str] = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        variant = row[:, x1:x2 + 1]
        if variant.size == 0:
            continue
        scaled = cv2.resize(
            variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        result = engine(scaled, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        if txts and txts[0]:
            parts.append(txts[0].strip())
    return "".join(parts)


def _repair(text: str) -> tuple[str, dict]:
    """The existing repair chain, applied in its documented order."""
    notes: dict = {}
    working, lexical_notes = normalise_lexical(text)
    notes["lexical"] = lexical_notes
    working, n_oxo = rewrite_oxo(working)
    notes["rewrite_oxo_fixes"] = n_oxo
    working = auto_repair_smiles(working)
    balanced = close_unbalanced_brackets(working)
    notes["bracket_close_appended"] = balanced is not None
    if balanced:
        working = balanced
    return working, notes


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    engines = {
        "en_v5@1x": (
            RapidOCR(
                params={
                    "Rec.lang_type": "en",
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.task_type": TaskType.REC,
                    "Global.text_score": 0.0,
                    "Global.use_cls": False,
                }
            ),
            1,
        ),
        "default_v6@2x": (RapidOCR(params={}), 2),
    }

    rows: list[dict] = []
    for path in sorted(CELLS.glob("*.png")):
        compound_id = int(path.stem.split("cmpd")[-1])
        cell = cv2.imread(str(path))
        for name, (engine, scale) in engines.items():
            raw = _read_bands(cell, engine, scale)
            repaired, notes = _repair(raw)
            reference = "".join(REFERENCE_LINES.get(compound_id, ()))
            row = {
                "compound_id": compound_id,
                "route": name,
                "raw_text": raw,
                "raw_parses": _parses(raw),
                "raw_balanced": (
                    raw.count("(") == raw.count(")")
                    and raw.count("[") == raw.count("]")
                ),
                "repaired_text": repaired,
                "repaired_parses": _parses(repaired),
                "rewrite_oxo_fixes": notes["rewrite_oxo_fixes"],
                "bracket_close_appended": notes["bracket_close_appended"],
                "has_reference": bool(reference),
                "raw_edit_to_reference": (
                    edit_distance(reference, raw) if reference else ""
                ),
                "repaired_edit_to_reference": (
                    edit_distance(reference, repaired) if reference else ""
                ),
                "raw_cer_to_reference": (
                    round(cer(reference, raw), 5) if reference else ""
                ),
                "repaired_cer_to_reference": (
                    round(cer(reference, repaired), 5) if reference else ""
                ),
            }
            rows.append(row)
            marker = ""
            if reference:
                marker = (
                    f" raw_ed={row['raw_edit_to_reference']}"
                    f" rep_ed={row['repaired_edit_to_reference']}"
                )
            print(
                f"cmpd {compound_id:>2} {name:14s} "
                f"parse_raw={row['raw_parses']!s:5s} "
                f"parse_rep={row['repaired_parses']!s:5s}{marker}"
            )

    with (OUT / "recovery_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {}
    for name in engines:
        subset = [row for row in rows if row["route"] == name]
        referenced = [row for row in subset if row["has_reference"]]
        summary[name] = {
            "compounds": len(subset),
            "raw_parses": sum(row["raw_parses"] for row in subset),
            "repaired_parses": sum(row["repaired_parses"] for row in subset),
            "raw_bracket_balanced": sum(row["raw_balanced"] for row in subset),
            "referenced_compounds": len(referenced),
            "raw_edit_to_reference": [
                row["raw_edit_to_reference"] for row in referenced
            ],
            "repaired_edit_to_reference": [
                row["repaired_edit_to_reference"] for row in referenced
            ],
            "repaired_matches_reference": sum(
                row["repaired_edit_to_reference"] == 0 for row in referenced
            ),
        }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
