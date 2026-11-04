"""A band-geometry matrix measured on both patents at once.

Two defects were measured, one per patent:

* WO2022251372 (serif, generous leading): the last glyph of every
  printed line sits flush against the crop border and is mis-read.  A
  small white margin fixes it.
* WO2025162428 (sans-serif, tight leading): the ink band is padded
  by a fixed 14/15 px, which reaches into the neighbouring lines and
  drags fragments of them into the crop.  Those fragments are what
  makes the recogniser unstable.

This script crosses the two ideas -- a fixed vertical pad, a vertical
pad limited by the gap to the neighbouring ink, and a horizontal white
margin -- with the two recognisers, and reports every configuration
on both patents against the hand transcriptions.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_adaptive_padding.py
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
OLD_META = OLD_CELLS / "cells_text_meta.csv"
OLD_REFERENCES = REPO / "data" / "audit" / "phase2" / "ocr_references.json"
NEW_CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase5" / "adaptive_padding"

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


def _read(
    cell: np.ndarray,
    engine: RapidOCR,
    *,
    scale: int,
    margin: int,
    vertical: str,
) -> str:
    """Read a cell with a configurable band geometry."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bands = ink_bands(gray)
    parts: list[str] = []
    for index, (top, bottom) in enumerate(bands):
        if vertical == "fixed":
            pad_top, pad_bottom = 14, 15
        else:
            gap_above = (
                top - bands[index - 1][1] - 1 if index else top
            )
            gap_below = (
                bands[index + 1][0] - bottom - 1
                if index + 1 < len(bands)
                else cell.shape[0] - bottom - 1
            )
            pad_top = min(14, max(0, gap_above // 2))
            pad_bottom = min(15, max(0, gap_below // 2))

        y1 = max(0, top - pad_top)
        y2 = min(cell.shape[0], bottom + pad_bottom)
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
        if path is None or not path.exists():
            continue
        samples.append((str(cmpd_id), path, payload["smiles"]))
    return samples


def _new_samples() -> list[tuple[str, Path, str]]:
    samples: list[tuple[str, Path, str]] = []
    for compound_id, lines in REFERENCE_LINES.items():
        matches = sorted(NEW_CELLS.glob(f"*_cmpd{compound_id}.png"))
        if not matches:
            continue
        samples.append((str(compound_id), matches[0], "".join(lines)))
    return samples


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    engines = {
        "en_v4": _english(OCRVersion.PPOCRV4),
        "default_v6": RapidOCR(params={}),
    }
    configs = []
    for engine_name in ("default_v6", "en_v4"):
        for vertical in ("fixed", "adaptive"):
            for margin in (0, 4, 6, 8):
                configs.append(
                    (f"{engine_name}|{vertical}|{margin}px", engine_name,
                     vertical, margin)
                )

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
            for name, engine_name, vertical, margin in configs:
                text = _read(
                    cell,
                    engines[engine_name],
                    scale=2,
                    margin=margin,
                    vertical=vertical,
                )
                rows.append(
                    {
                        "corpus": corpus,
                        "cmpd_id": cmpd_id,
                        "config": name,
                        "edit_distance": edit_distance(reference, text),
                        "cer": round(cer(reference, text), 5),
                        "parses": _parses(text),
                        "ocr_text": text,
                    }
                )

    with (OUT / "adaptive_padding_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for corpus in corpora:
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
                f"  {name:30s} exact {value['exact']}/{value['samples']}  "
                f"edit {value['total_edit']:>3}  cer {value['mean_cer']:.4f}  "
                f"parse {value['parses']}/{value['samples']}  {value['edits']}"
            )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
