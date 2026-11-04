"""Evaluate the proposed OCR configuration on all 28 Table 1a compounds.

The padding matrix found that the printed line must not end exactly on
the last ink pixel: ``text_span`` returns the first and last column that
contains ink, so the final glyph of every printed line sits flush
against the crop border and the recogniser mis-reads it.  Adding a
small white margin around the band fixes that.

This script measures the effect over the whole Table 1a, not only over
the three manually transcribed compounds.  It compares three
configurations on identical inputs:

``current_validated``
    what the shipped pipeline does today: the upstream default
    recogniser, 2x upscale, no margin.
``english_v4_nomargin``
    the English recogniser without a margin -- isolates the effect of
    switching the model.
``english_v4_margin``
    the English recogniser with a 6 px white margin -- the candidate.

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_candidate_config.py
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

from cpd.ocr.printed_smiles import ink_bands, text_span  # noqa: E402

from wo2022251372_references import (  # noqa: E402
    REFERENCE_LINES,
    cer,
    edit_distance,
)

CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase4" / "wo2022251372" / "candidate_config"

MARGIN = 6
WHITE = (255, 255, 255)
PAD_TOP = 14
PAD_BOTTOM = 15


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


def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _balanced(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


def _read(cell: np.ndarray, engine: RapidOCR, scale: int, margin: int) -> str:
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
        if margin:
            band = cv2.copyMakeBorder(
                band,
                0,
                0,
                margin,
                margin,
                cv2.BORDER_CONSTANT,
                value=WHITE,
            )
        scaled = cv2.resize(
            band, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        text = _recognise_only(engine, scaled)
        if text:
            parts.append(text)
    return "".join(parts)


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    default = RapidOCR(params={})
    english = _english_v4()
    configs = {
        "current_validated": (default, 2, 0),
        "english_v4_nomargin": (english, 1, 0),
        "english_v4_margin": (english, 1, MARGIN),
    }

    rows: list[dict] = []
    for path in sorted(CELLS.glob("*.png")):
        compound_id = int(path.stem.split("cmpd")[-1])
        cell = cv2.imread(str(path))
        row: dict = {"compound_id": compound_id, "cell_path": str(path)}
        reference = (
            "".join(REFERENCE_LINES[compound_id])
            if compound_id in REFERENCE_LINES
            else ""
        )
        row["manual_reference_status"] = (
            "hand_transcribed" if reference else "not_transcribed"
        )
        row["manual_reference"] = reference
        for name, (engine, scale, margin) in configs.items():
            text = _read(cell, engine, scale, margin)
            row[f"{name}_text"] = text
            row[f"{name}_length"] = len(text)
            row[f"{name}_balanced"] = _balanced(text)
            row[f"{name}_parses"] = _parses(text)
            row[f"{name}_edit_to_reference"] = (
                edit_distance(reference, text) if reference else ""
            )
            row[f"{name}_cer_to_reference"] = (
                round(cer(reference, text), 5) if reference else ""
            )
        rows.append(row)
        print(
            f"cmpd {compound_id:>2}  "
            f"current: bal={row['current_validated_balanced']!s:5s} "
            f"parse={row['current_validated_parses']!s:5s}  "
            f"candidate: bal={row['english_v4_margin_balanced']!s:5s} "
            f"parse={row['english_v4_margin_parses']!s:5s}"
            + (
                f" ed={row['english_v4_margin_edit_to_reference']}"
                if reference
                else ""
            )
        )

    with (OUT / "candidate_config_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        name: {
            "compounds": len(rows),
            "bracket_balanced": sum(
                row[f"{name}_balanced"] for row in rows
            ),
            "rdkit_parses": sum(row[f"{name}_parses"] for row in rows),
            "hand_transcribed_compounds": sum(
                bool(row["manual_reference"]) for row in rows
            ),
            "edit_to_reference": [
                row[f"{name}_edit_to_reference"]
                for row in rows
                if row["manual_reference"]
            ],
            "cer_to_reference": [
                row[f"{name}_cer_to_reference"]
                for row in rows
                if row["manual_reference"]
            ],
        }
        for name in configs
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
