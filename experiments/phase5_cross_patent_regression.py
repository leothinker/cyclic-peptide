"""Cross-patent regression check for the printed-SMILES OCR route.

Two patents, two printing offices, two fonts:

* WO2025162428 -- the shipped corpus, sans-serif, 335 printed cells;
* WO2022251372 -- the new corpus, serif, 28 printed cells.

Before changing the shared OCR route, the candidate configurations
have to be measured on *both*, otherwise a fix for the new patent can
silently undo the validated result on the old one.

The old-patent reference strings are the hand transcriptions already
stored in ``data/audit/phase2/ocr_references.json``; the new-patent
ones live in ``experiments/wo2022251372_references.py``.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_cross_patent_regression.py
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

OLD_CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OLD_REFERENCES = REPO / "data" / "audit" / "phase2" / "ocr_references.json"
OLD_META = OLD_CELLS / "cells_text_meta.csv"
NEW_CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase5" / "cross_patent_regression"

PAD_TOP = 14
PAD_BOTTOM = 15
WHITE = (255, 255, 255)


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


def _read(cell: np.ndarray, engine: RapidOCR, scale: int, margin: int) -> str:
    """The ink-band route with a configurable recogniser, scale and margin."""
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
                band, 0, 0, margin, margin, cv2.BORDER_CONSTANT, value=WHITE
            )
        scaled = cv2.resize(
            band, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        text = _recognise_only(engine, scaled)
        if text:
            parts.append(text)
    return "".join(parts)


def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _old_samples() -> list[tuple[str, Path, str]]:
    data = json.loads(OLD_REFERENCES.read_text(encoding="utf-8"))
    index: dict[str, Path] = {}
    with OLD_META.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            crop = (row.get("crop_path") or "").strip()
            if crop:
                index[(row.get("cmpd_id") or "").strip()] = (
                    OLD_CELLS.parent / crop
                )
    samples: list[tuple[str, Path, str]] = []
    for cmpd_id, payload in data.items():
        path = index.get(str(cmpd_id))
        if path is None:
            path = OLD_CELLS / f"I100297_{cmpd_id}_text.png"
        if not path.exists():
            continue
        samples.append((str(cmpd_id), path, payload["smiles"]))
    return samples


def _new_samples() -> list[tuple[str, Path, str]]:
    samples: list[tuple[str, Path, str]] = []
    for compound_id, lines in REFERENCE_LINES.items():
        path = NEW_CELLS / f"000116_cmpd{compound_id}.png"
        if not path.exists():
            matches = sorted(NEW_CELLS.glob(f"*_cmpd{compound_id}.png"))
            if not matches:
                continue
            path = matches[0]
        samples.append((str(compound_id), path, "".join(lines)))
    return samples


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    engines = {
        "en_v4": _english(OCRVersion.PPOCRV4),
        "default_v6": RapidOCR(params={}),
    }
    configs = [
        ("default_v6@2x+0px", "default_v6", 2, 0),
        ("default_v6@2x+6px", "default_v6", 2, 6),
        ("en_v4@2x+0px", "en_v4", 2, 0),
        ("en_v4@2x+6px", "en_v4", 2, 6),
        ("en_v4@1x+6px", "en_v4", 1, 6),
    ]

    corpora = {
        "WO2025162428": _old_samples(),
        "WO2022251372": _new_samples(),
    }

    rows: list[dict] = []
    for corpus, samples in corpora.items():
        for cmpd_id, path, reference in samples:
            cell = cv2.imread(str(path))
            if cell is None:
                continue
            for name, engine_name, scale, margin in configs:
                text = _read(cell, engines[engine_name], scale, margin)
                rows.append(
                    {
                        "corpus": corpus,
                        "cmpd_id": cmpd_id,
                        "config": name,
                        "reference_length": len(reference),
                        "ocr_length": len(text),
                        "edit_distance": edit_distance(reference, text),
                        "cer": round(cer(reference, text), 5),
                        "parses": _parses(text),
                        "cell_path": str(path.relative_to(REPO)).replace(
                            "\\", "/"
                        ),
                    }
                )

    with (OUT / "cross_patent_regression.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for corpus, samples in corpora.items():
        summary[corpus] = {}
        for name, _, _, _ in configs:
            subset = [
                row
                for row in rows
                if row["corpus"] == corpus and row["config"] == name
            ]
            if not subset:
                continue
            summary[corpus][name] = {
                "samples": len(subset),
                "exact": sum(row["edit_distance"] == 0 for row in subset),
                "total_edit": sum(row["edit_distance"] for row in subset),
                "mean_cer": round(
                    sum(row["cer"] for row in subset) / len(subset), 5
                ),
                "parses": sum(row["parses"] for row in subset),
                "edits": [row["edit_distance"] for row in subset],
            }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for corpus, configs_out in summary.items():
        print(f"=== {corpus} ===")
        for name, value in configs_out.items():
            print(
                f"  {name:20s} exact {value['exact']}/{value['samples']}  "
                f"edit {value['total_edit']:>3}  cer {value['mean_cer']:.4f}  "
                f"parse {value['parses']}/{value['samples']}  "
                f"{value['edits']}"
            )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
