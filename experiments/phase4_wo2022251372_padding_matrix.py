"""Right/left margin sweep for the WO2022251372 printed-SMILES OCR.

Why this experiment exists
--------------------------
The band diagnosis showed that the ink bands are segmented perfectly
(30/30 printed lines) and that the remaining damage is made by the
recogniser.  Inspecting the actual crops showed why: ``text_span``
returns the first and last column that contains *any* ink, so the
cropped band ends exactly on the last ink pixel of the final glyph.
That glyph is therefore flush against the image border, with no margin,
and the recogniser mis-reads or drops it:

    crop with 0 px margin  ->  'H](Cc2ccc(Cl)cc2)NC(=O)'   ( '[' lost )
    crop with 4 px margin  ->  'H](Cc2ccc(Cl)cc2)NC(=O)['  ( exact )

This is the same *class* of defect as the row-start truncation measured
on WO2025162428 (a glyph at the crop boundary is damaged), but the
mechanism differs: there the pixels were missing from the detector
box, here every pixel is present and only the margin is missing.

The sweep below varies only the white margin added around the band and
the recogniser/scale pair.  It is scored on the 30 manually
transcribed printed lines of compounds 1, 2 and 3.

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_padding_matrix.py
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
    CELL_FILES,
    REFERENCE_LINES,
    cer,
    edit_distance,
)

CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase4" / "wo2022251372" / "padding_matrix"

PAD_TOP = 14
PAD_BOTTOM = 15
WHITE = (255, 255, 255)

MARGINS = (0, 2, 4, 6, 8, 10, 12, 16, 20, 24)
SCALES = (1, 2)


def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


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


def _recognise_only(engine: RapidOCR, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def _bands(cell: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    out: list[np.ndarray] = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        out.append(row[:, x1:x2 + 1])
    return out


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    engines = {
        "en_v4": _english(OCRVersion.PPOCRV4),
        "en_v5": _english(OCRVersion.PPOCRV5),
        "default_v6": RapidOCR(params={}),
    }

    # Pre-extract the bands once.
    compound_bands: dict[int, list[np.ndarray]] = {}
    for compound_id, filename in CELL_FILES.items():
        cell = cv2.imread(str(CELLS / filename))
        compound_bands[compound_id] = _bands(cell)

    line_rows: list[dict] = []
    compound_rows: list[dict] = []

    for engine_name, engine in engines.items():
        for scale in SCALES:
            for margin in MARGINS:
                joined_by_compound: dict[int, str] = {}
                for compound_id, bands in compound_bands.items():
                    reference_lines = REFERENCE_LINES[compound_id]
                    parts: list[str] = []
                    for index, band in enumerate(bands):
                        padded = band
                        if margin:
                            padded = cv2.copyMakeBorder(
                                band,
                                0,
                                0,
                                margin,
                                margin,
                                cv2.BORDER_CONSTANT,
                                value=WHITE,
                            )
                        scaled = cv2.resize(
                            padded,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                        text = _recognise_only(engine, scaled)
                        parts.append(text)
                        reference = (
                            reference_lines[index]
                            if index < len(reference_lines)
                            else ""
                        )
                        distance = edit_distance(reference, text)
                        line_rows.append(
                            {
                                "compound_id": compound_id,
                                "band_index": index,
                                "recogniser": engine_name,
                                "scale": scale,
                                "margin_px": margin,
                                "reference_line": reference,
                                "ocr_text": text,
                                "edit_distance": distance,
                                "cer": round(cer(reference, text), 5),
                            }
                        )
                    joined_by_compound[compound_id] = "".join(parts)

                for compound_id, joined in joined_by_compound.items():
                    reference = "".join(REFERENCE_LINES[compound_id])
                    distance = edit_distance(reference, joined)
                    compound_rows.append(
                        {
                            "compound_id": compound_id,
                            "recogniser": engine_name,
                            "scale": scale,
                            "margin_px": margin,
                            "cell_edit_distance": distance,
                            "cell_cer": round(cer(reference, joined), 5),
                            "cell_parses": _parses(joined),
                            "reference_length": len(reference),
                            "ocr_length": len(joined),
                            "ocr_text": joined,
                        }
                    )

    with (OUT / "line_level_padding_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(line_rows[0].keys()))
        writer.writeheader()
        writer.writerows(line_rows)
    with (OUT / "compound_level_padding_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compound_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compound_rows)

    summary: dict[str, dict] = {}
    for engine_name in engines:
        for scale in SCALES:
            for margin in MARGINS:
                lines = [
                    row
                    for row in line_rows
                    if row["recogniser"] == engine_name
                    and row["scale"] == scale
                    and row["margin_px"] == margin
                ]
                compounds = [
                    row
                    for row in compound_rows
                    if row["recogniser"] == engine_name
                    and row["scale"] == scale
                    and row["margin_px"] == margin
                ]
                key = f"{engine_name}@{scale}x+{margin}px"
                summary[key] = {
                    "exact_lines": sum(
                        row["edit_distance"] == 0 for row in lines
                    ),
                    "total_lines": len(lines),
                    "total_line_edit": sum(
                        row["edit_distance"] for row in lines
                    ),
                    "mean_line_cer": round(
                        sum(row["cer"] for row in lines) / max(len(lines), 1), 5
                    ),
                    "cell_cer": [row["cell_cer"] for row in compounds],
                    "cell_edit": [
                        row["cell_edit_distance"] for row in compounds
                    ],
                    "compounds_parsing": sum(
                        row["cell_parses"] for row in compounds
                    ),
                    "compounds_total": len(compounds),
                }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    ranked = sorted(
        summary.items(),
        key=lambda kv: (kv[1]["total_line_edit"], -kv[1]["exact_lines"]),
    )
    total_lines = next(iter(summary.values()))["total_lines"]
    print(
        f"{'config':26s} exact  line_edit  cer     cell_cer"
        f"           parse"
    )
    for key, value in ranked:
        print(
            f"{key:26s} {value['exact_lines']:>2}/{total_lines}  "
            f"{value['total_line_edit']:>5}      {value['mean_line_cer']:.4f}  "
            f"{value['cell_cer']}  {value['compounds_parsing']}/3"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
