"""Line-level diagnosis of the WO2022251372 printed-SMILES OCR failures.

The phase-4 audit showed that the existing OCR routes read almost none of
the WO2022251372 SMILES cells.  This script localises the failure to one of
three stages by comparing, band by band, the printed line against the
recognised text:

1. ink-band segmentation  -- are the printed lines split correctly?
2. line geometry           -- does the cropped band still hold the ink?
3. recognition             -- does the recogniser read the line correctly?

Three compounds are transcribed character by character from the 300 dpi
source pages and used as the reference.  No production code and no formal
CSV is touched; everything is written under
``data/audit/phase4/wo2022251372/band_diag``.

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_band_diag.py
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

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import (  # noqa: E402

    PAD_BOTTOM,
    PAD_TOP,
    ink_bands,
    text_span,
    upscale,
)
from wo2022251372_references import (  # noqa: E402
    CELL_FILES,
    REFERENCE_LINES,
    cer,
    edit_distance,
)


CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase4" / "wo2022251372" / "band_diag"

# Transcribed character by character from the 300 dpi source pages
# (data/raw/WO2022251372/golden_tables/000116.tif and 000117.tif).
# Compounds 1, 2 and 3 share every printed line except the last one.








def _recognisers() -> dict[str, RapidOCR]:
    """The recogniser variants under test."""
    english = RapidOCR(
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
    default = RapidOCR(params={})
    return {"en_v4_mobile": english, "default_v6": default}


def _recognise_only(engine: RapidOCR, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engines = _recognisers()
    rows: list[dict] = []

    for compound_id, filename in CELL_FILES.items():
        cell = cv2.imread(str(CELLS / filename))
        if cell is None:
            raise RuntimeError(f"missing cell crop {filename}")
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        bands = ink_bands(gray)
        reference_lines = REFERENCE_LINES[compound_id]

        for index, (top, bottom) in enumerate(bands):
            y1 = max(0, top - PAD_TOP)
            y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
            row = cell[y1:y2, :]
            row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
            x1, x2 = text_span(row_gray)
            variant = row[:, x1:x2 + 1]

            reference = (
                reference_lines[index] if index < len(reference_lines) else ""
            )
            record = {
                "compound_id": compound_id,
                "band_index": index,
                "band_top": top,
                "band_bottom": bottom,
                "band_height": bottom - top + 1,
                "crop_x1": x1,
                "crop_x2": x2,
                "reference_line": reference,
                "reference_present": bool(reference),
            }
            for name, engine in engines.items():
                text = _recognise_only(engine, upscale(variant))
                record[f"{name}_text"] = text
                record[f"{name}_edit"] = edit_distance(reference, text)
                record[f"{name}_cer"] = round(cer(reference, text), 5)
            rows.append(record)
            print(
                f"cmpd {compound_id} band {index:>2} "
                f"h={bottom - top + 1:>3} "
                f"en={record['en_v4_mobile_edit']:>2} "
                f"v6={record['default_v6_edit']:>2} "
                f"| {reference}"
            )

    csv_path = OUT / "band_level_ocr.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    reference_bands = [row for row in rows if row["reference_present"]]
    summary = {
        "reference_compounds": sorted(REFERENCE_LINES),
        "reference_lines": len(reference_bands),
        "detected_bands": len(rows),
        "band_count_matches_reference": all(
            len(ink_bands(cv2.cvtColor(cv2.imread(str(CELLS / f)), cv2.COLOR_BGR2GRAY)))
            == len(REFERENCE_LINES[cid])
            for cid, f in CELL_FILES.items()
        ),
        "en_v4_mobile": {
            "exact_lines": sum(
                row["en_v4_mobile_edit"] == 0 for row in reference_bands
            ),
            "total_edit": sum(
                row["en_v4_mobile_edit"] for row in reference_bands
            ),
            "mean_cer": round(
                sum(row["en_v4_mobile_cer"] for row in reference_bands)
                / max(len(reference_bands), 1),
                5,
            ),
        },
        "default_v6": {
            "exact_lines": sum(
                row["default_v6_edit"] == 0 for row in reference_bands
            ),
            "total_edit": sum(
                row["default_v6_edit"] for row in reference_bands
            ),
            "mean_cer": round(
                sum(row["default_v6_cer"] for row in reference_bands)
                / max(len(reference_bands), 1),
                5,
            ),
        },
    }
    (OUT / "band_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
