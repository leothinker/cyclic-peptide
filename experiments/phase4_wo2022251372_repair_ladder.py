"""Does the existing repair chain help once the OCR geometry is fixed?

The candidate configuration (English PP-OCRv4 recogniser plus a 6 px
white margin around each printed line) takes WO2022251372 from 0/28 to
17/28 RDKit-parseable.  The remaining failures are dominated by two
damage patterns that ``cpd.ocr_edit_model`` already documented for
WO2025162428:

* ``C(=O)`` printed as ``C=O)`` -- the bracket before ``=`` is lost;
* ``C(=O)`` printed as ``C-O)`` -- bracket and bond sign collapse.

``cpd.ocr_rewrite.rewrite_oxo`` rewrites the second pattern.  This
script measures what the whole chain does to the candidate OCR text,
and -- the point of the experiment -- whether the repaired string gets
*closer to the hand transcription* or merely becomes parseable.

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_repair_ladder.py
"""

from __future__ import annotations

import csv
import json
import re
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
from cpd.ocr_edit_model import normalise_lexical  # noqa: E402
from cpd.ocr_rewrite import rewrite_oxo  # noqa: E402


CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
from wo2022251372_references import (  # noqa: E402
    CELL_FILES,
    REFERENCE_LINES,
    cer,
    edit_distance,
)

OUT = REPO / "data" / "audit" / "phase4" / "wo2022251372" / "repair_ladder"



MARGIN = 6
WHITE = (255, 255, 255)
PAD_TOP = 14
PAD_BOTTOM = 15

# ``=0`` can never be valid SMILES: zero is not an element, so a double
# bond to it is always an OCR artefact for ``=O``.
EQUALS_ZERO = re.compile(r"=0(?![0-9])")

# The shipped ``rewrite_oxo`` treats every ``X=O)`` as a damaged
# ``X(=O)``.  In WO2022251372 the lactam carbonyl is printed as
# ``C1=O)``: the ring-closure digit sits between the carbon and the
# ``=``, so the shipped rule inserts a spurious ``(`` into a perfectly
# good string.  The guarded variant below only rewrites when the
# character before the ``=`` cannot be a ring-closure digit.
OXO = re.compile(r"(.)[-=](O\))")


def rewrite_oxo_guarded(text: str) -> tuple[str, int]:
    """``rewrite_oxo`` without the ring-closure-digit false positive."""
    count = 0

    def sub(match: "re.Match") -> str:
        if match.group(1) in "([" or match.group(1).isdigit():
            return match.group(0)
        nonlocal count
        count += 1
        return f"{match.group(1)}(=O)"

    return OXO.sub(sub, text), count




def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _english_v4() -> RapidOCR:
    return RapidOCR(
        params={
            "Rec.lang_type": "en",
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.0,
            "Global.use_cls": False,
        }
    )


def _recognise_only(engine: RapidOCR, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def _read(cell: np.ndarray, engine: RapidOCR) -> str:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts: list[str] = []
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
            band, 0, 0, MARGIN, MARGIN, cv2.BORDER_CONSTANT, value=WHITE
        )
        text = _recognise_only(engine, band)
        if text:
            parts.append(text)
    return "".join(parts)


def _stage(name: str, text: str, reference: str) -> dict:
    return {
        f"{name}_parses": _parses(text),
        f"{name}_edit_to_reference": (
            edit_distance(reference, text) if reference else ""
        ),
    }


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)
    engine = _english_v4()

    rows: list[dict] = []
    for path in sorted(CELLS.glob("*.png")):
        compound_id = int(path.stem.split("cmpd")[-1])
        cell = cv2.imread(str(path))
        raw = _read(cell, engine)

        step_oxo, n_oxo = rewrite_oxo(raw)
        step_guarded, n_guarded = rewrite_oxo_guarded(raw)
        step_zero, n_zero = EQUALS_ZERO.subn("=O", step_oxo)
        step_guarded_zero, n_guarded_zero = EQUALS_ZERO.subn(
            "=O", step_guarded
        )
        step_guarded_closed = (
            close_unbalanced_brackets(step_guarded_zero) or step_guarded_zero
        )
        step_auto = auto_repair_smiles(step_zero)
        step_closed = close_unbalanced_brackets(step_auto) or step_auto

        reference = "".join(REFERENCE_LINES.get(compound_id, ()))
        row = {
            "compound_id": compound_id,
            "raw": raw,
            "rewrite_oxo_fixes": n_oxo,
            "equals_zero_fixes": n_zero,
            "guarded_oxo_fixes": n_guarded,
            "guarded_equals_zero_fixes": n_guarded_zero,
            "after_rewrite_oxo": step_oxo,
            "after_equals_zero": step_zero,
            "after_auto_repair": step_auto,
            "after_bracket_close": step_closed,
            "after_guarded_oxo": step_guarded,
            "after_guarded_equals_zero": step_guarded_zero,
            "after_guarded_bracket_close": step_guarded_closed,
            "has_reference": bool(reference),
        }
        row.update(_stage("raw", raw, reference))
        row.update(_stage("after_rewrite_oxo", step_oxo, reference))
        row.update(_stage("after_equals_zero", step_zero, reference))
        row.update(_stage("after_auto_repair", step_auto, reference))
        row.update(_stage("after_bracket_close", step_closed, reference))
        row.update(_stage("after_guarded_oxo", step_guarded, reference))
        row.update(
            _stage("after_guarded_equals_zero", step_guarded_zero, reference)
        )
        row.update(
            _stage(
                "after_guarded_bracket_close", step_guarded_closed, reference
            )
        )
        rows.append(row)
        marker = ""
        if reference:
            marker = (
                f" ed_raw={row['raw_edit_to_reference']}"
                f" ed_closed={row['after_bracket_close_edit_to_reference']}"
            )
        print(
            f"cmpd {compound_id:>2} raw_parse={row['raw_parses']!s:5s} "
            f"oxo={row['rewrite_oxo_fixes']:>2} "
            f"eq0={row['equals_zero_fixes']:>2} "
            f"final_parse={row['after_bracket_close_parses']!s:5s} "
            f"guarded={row['after_guarded_bracket_close_parses']!s:5s}{marker}"
        )

    with (OUT / "repair_ladder.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    referenced = [row for row in rows if row["has_reference"]]
    stages = (
        "raw",
        "after_rewrite_oxo",
        "after_equals_zero",
        "after_auto_repair",
        "after_bracket_close",
        "after_guarded_oxo",
        "after_guarded_equals_zero",
        "after_guarded_bracket_close",
    )
    summary = {
        "compounds": len(rows),
        "parses": {
            stage: sum(row[f"{stage}_parses"] for row in rows)
            for stage in stages
        },
        "edit_to_reference": {
            stage: [
                row[f"{stage}_edit_to_reference"] for row in referenced
            ]
            for stage in stages
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
