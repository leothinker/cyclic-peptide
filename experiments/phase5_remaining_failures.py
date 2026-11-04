"""What still goes wrong on WO2022251372 after the padded route.

The padded route takes 17 of the 28 Table 1a compounds to an RDKit
parseable string.  This script looks at the 11 that do not parse and
classifies the damage, so the next step is chosen on evidence rather
than on a guess.

For every compound the printed lines are segmented with the same ink
band code the route uses, each line is recognised on its own, and the
line is compared with the matching hand-transcribed line (where one
exists).  The report is per line, because the damage is per line.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_remaining_failures.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PADDED_MARGIN_PX,
    WHITE,
    ink_bands,
    text_span,
    upscale,
)
from cpd.ocr.printed_smiles import english_recogniser_only  # noqa: E402

from wo2022251372_references import (  # noqa: E402
    CELL_FILES,
    REFERENCE_LINES,
    edit_distance,
)

CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase5" / "remaining_failures"

PAD_TOP = 14
PAD_BOTTOM = 15


def _lines(cell: np.ndarray, engine) -> list[str]:
    """Recognise every printed line of one cell separately."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    out = []
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
        result = engine(band, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        out.append((txts[0] or "").strip() if txts else "")
    return out


def _parses(text: str) -> bool:
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


# Damage signatures, most specific first.
PATTERNS = [
    ("lost_open_bracket", re.compile(r"(?<![(\[])(?<!=[A-Za-z])[A-Za-z@]\]")),
    ("unbalanced_open", re.compile(r"\(")),
    ("unbalanced_close", re.compile(r"\)")),
]


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)
    engine = english_recogniser_only()

    rows = []
    for path in sorted(CELLS.glob("*.png")):
        compound_id = int(path.stem.split("cmpd")[-1])
        cell = cv2.imread(str(path))
        got = _lines(cell, engine)
        ref = list(REFERENCE_LINES.get(compound_id, ()))
        whole = "".join(got)

        for index, text in enumerate(got):
            reference = ref[index] if index < len(ref) else ""
            distance = edit_distance(reference, text) if reference else ""
            rows.append(
                {
                    "compound_id": compound_id,
                    "line_index": index,
                    "ocr_line": text,
                    "reference_line": reference,
                    "edit_distance": distance,
                    "line_parses": _parses(text),
                    "whole_cell_parses": _parses(whole),
                    "has_reference": bool(reference),
                }
            )

    with (OUT / "line_failures.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "lines": len(rows),
        "lines_with_reference": sum(1 for r in rows if r["has_reference"]),
        "reference_lines_exact": sum(
            1
            for r in rows
            if r["has_reference"] and r["edit_distance"] == 0
        ),
        "reference_lines_with_errors": sum(
            1
            for r in rows
            if r["has_reference"] and r["edit_distance"] not in ("", 0)
        ),
    }

    # The damage that repeats, counted over the reference lines only.
    signatures: Counter = Counter()
    for row in rows:
        if not row["has_reference"] or row["edit_distance"] in ("", 0):
            continue
        got = row["ocr_line"]
        ref = row["reference_line"]
        if got.count("(") < ref.count("("):
            signatures["lost_bracket"] += 1
        elif got.count(")") < ref.count(")"):
            signatures["lost_closing_bracket"] += 1
        elif "0" in got and "0" not in ref:
            signatures["zero_for_oxygen"] += 1
        else:
            signatures["other_character_error"] += 1
    summary["error_signatures"] = dict(signatures)

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (OUT / "error_lines.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        for row in rows:
            if row["has_reference"] and row["edit_distance"] not in ("", 0):
                writer.writerow(row)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
