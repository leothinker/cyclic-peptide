"""Recogniser x scale matrix for the WO2022251372 printed SMILES.

The band diagnosis showed that ink-band segmentation is perfect for this
patent (30/30 printed lines recovered on the three transcribed
compounds) and that every remaining error is made by the recogniser.
This script therefore varies only the two things that can still be
varied without touching production code:

* the recogniser model (English PP-OCRv4 mobile, English PP-OCRv5
  mobile, the upstream default PP-OCRv6 small);
* the upscaling factor applied to the band before recognition.

The evaluation set is the 30 manually transcribed printed lines of
compounds 1, 2 and 3.  Every line is scored on character error rate and
on whether the concatenation of all lines of a compound parses in
RDKit -- the two levels are reported separately.

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_recognizer_matrix.py
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


CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
from wo2022251372_references import (  # noqa: E402
    CELL_FILES,
    REFERENCE_LINES,
    cer,
    edit_distance,
)

OUT = REPO / "data" / "audit" / "phase4" / "wo2022251372" / "recognizer_matrix"




SCALES = (1, 2, 3)
PAD_TOP = 14
PAD_BOTTOM = 15






def _english(version: OCRVersion) -> RapidOCR:
    return RapidOCR(
        params={
            "Rec.lang_type": "en",
            "Rec.ocr_version": version,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.0,
            "Global.use_cls": False,
        }
    )


def _build_engines() -> dict[str, RapidOCR]:
    return {
        "en_v4": _english(OCRVersion.PPOCRV4),
        "en_v5": _english(OCRVersion.PPOCRV5),
        "default_v6": RapidOCR(params={}),
    }


def _recognise_only(engine: RapidOCR, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def _parses(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)
    engines = _build_engines()

    line_rows: list[dict] = []
    compound_rows: list[dict] = []

    for compound_id, filename in CELL_FILES.items():
        cell = cv2.imread(str(CELLS / filename))
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        bands = ink_bands(gray)
        reference_lines = REFERENCE_LINES[compound_id]

        variants: dict[tuple[str, int], list[str]] = {}
        for index, (top, bottom) in enumerate(bands):
            y1 = max(0, top - PAD_TOP)
            y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
            row = cell[y1:y2, :]
            row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
            x1, x2 = text_span(row_gray)
            variant = row[:, x1:x2 + 1]
            reference = reference_lines[index] if index < len(reference_lines) else ""

            for engine_name, engine in engines.items():
                for scale in SCALES:
                    scaled = cv2.resize(
                        variant,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    text = _recognise_only(engine, scaled)
                    key = (engine_name, scale)
                    variants.setdefault(key, []).append(text)
                    distance = edit_distance(reference, text)
                    line_rows.append(
                        {
                            "compound_id": compound_id,
                            "band_index": index,
                            "recogniser": engine_name,
                            "scale": scale,
                            "reference_line": reference,
                            "ocr_text": text,
                            "edit_distance": distance,
                            "cer": round(cer(reference, text), 5),
                        }
                    )

        for (engine_name, scale), texts in variants.items():
            joined = "".join(texts)
            reference = "".join(reference_lines)
            distance = edit_distance(reference, joined)
            compound_rows.append(
                {
                    "compound_id": compound_id,
                    "recogniser": engine_name,
                    "scale": scale,
                    "exact_lines": sum(
                        1
                        for text, expected in zip(texts, reference_lines)
                        if text == expected
                    ),
                    "cell_edit_distance": distance,
                    "cell_cer": round(cer(reference, joined), 5),
                    "cell_parses": _parses(joined),
                    "cell_length": len(joined),
                    "reference_length": len(reference),
                    "ocr_text": joined,
                }
            )

    with (OUT / "line_level_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(line_rows[0].keys()))
        writer.writeheader()
        writer.writerows(line_rows)

    with (OUT / "compound_level_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compound_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compound_rows)

    summary: dict[str, dict] = {}
    for engine_name in engines:
        for scale in SCALES:
            selected_lines = [
                row
                for row in line_rows
                if row["recogniser"] == engine_name and row["scale"] == scale
            ]
            selected_compounds = [
                row
                for row in compound_rows
                if row["recogniser"] == engine_name and row["scale"] == scale
            ]
            summary[f"{engine_name}@{scale}x"] = {
                "exact_lines": sum(
                    row["edit_distance"] == 0 for row in selected_lines
                ),
                "total_lines": len(selected_lines),
                "total_line_edit": sum(
                    row["edit_distance"] for row in selected_lines
                ),
                "mean_line_cer": round(
                    sum(row["cer"] for row in selected_lines)
                    / max(len(selected_lines), 1),
                    5,
                ),
                "compound_cer": [
                    row["cell_cer"] for row in selected_compounds
                ],
                "compounds_parsing": sum(
                    row["cell_parses"] for row in selected_compounds
                ),
                "compounds_total": len(selected_compounds),
            }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for key, value in summary.items():
        print(
            f"{key:14s} exact_lines={value['exact_lines']:>2}/{value['total_lines']} "
            f"line_edit={value['total_line_edit']:>3} "
            f"mean_cer={value['mean_line_cer']:.4f} "
            f"cell_cer={value['compound_cer']} "
            f"parse={value['compounds_parsing']}/{value['compounds_total']}"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
