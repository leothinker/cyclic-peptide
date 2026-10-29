r"""Page-level printed-SMILES extraction with correct text/box pairing.

Why this module exists
----------------------
Two defects in the earlier page-level experiments had to be removed before
any of their numbers could be reused.

1. Text/box mis-pairing. ``row_crop_audit.line_boxes`` sorts the detector
   boxes top-to-bottom, but the earlier helper then zipped that sorted box
   list against the *unsorted* ``result.txts``. Wherever the detector's
   emission order differed from the sorted order, a box received another
   row's text. On I100297 (36 boxes) only two slots were affected, so the
   1036/1037 numbers survived; on I100298 (92 boxes) the SMILES column
   showed ``uull`` / ``S`` / ``N.`` and the real SMILES strings were
   attached to boxes in the structure column. Every page-level number
   therefore has to be recomputed with the pairing fixed.

2. A confounded comparison. ``topk_preproc_probe`` compared
   ``RapidOCR(params={})`` on the raw page against the baseline engine on
   the preprocessed page. Those two differ in the preprocessing AND in the
 models: the default engine is PP-OCRv6 det_small + ch PP-OCRv6 rec small
 + ch cls,
   while the baseline is PP-OCRv4 mobile with an English recogniser. The
   resulting "preprocessing is the variable" conclusion was therefore not
   established, because the model moved with it.

This module separates the two axes explicitly. The extraction path (whole
page, per compound band) is fixed; the caller chooses the engine and the
preprocessing independently, so the 2x2 can be run.

Attribution does not rely on a "SMILES" header box. Continuation pages
carry no header at all (I100298 starts directly with SMILES rows), so the
header lookup returns nothing and every compound is lost. Here the
compound bands come from the audit manifest, which already records each
compound's cell rectangle in page coordinates.

Read-only with respect to the pipeline, ``src/cpd`` and the official CSV.

Run:
    .venv\Scripts\python.exe phase3_page_extract.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")


# ------------------------------------------------------------------ engines
def baseline_engine():
    """PP-OCRv4 mobile, English detector + English recogniser.

    This is the engine the production pipeline uses for SMILES cells
    (``RapidOcrTableExtractor.rec_engine``), so it is the reference point
    for any comparison.
    """
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import (
        EngineType, ModelType, OCRVersion, TaskType,
    )
    return RapidOCR(params={
        "Det.lang_type": "en",
        "Det.ocr_version": OCRVersion.PPOCRV4,
        "Det.model_type": ModelType.MOBILE,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.task_type": TaskType.DET,
        "Rec.lang_type": "en",
        "Rec.ocr_version": OCRVersion.PPOCRV4,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.task_type": TaskType.REC,
        "Global.text_score": 0.0,
        "Global.use_cls": False,
    })


def default_engine():
    """The upstream default: PP-OCRv6 det_small + ch PP-OCRv6 rec small
    + ch cls.

    This is what the orphaned ``_demo_topk.py`` used, and what the earlier
    probe called "raw_default".
    """
    from rapidocr import RapidOCR
    return RapidOCR(params={})


# ------------------------------------------------------------- preprocessing
def prep_none(img: np.ndarray) -> np.ndarray:
    """No preprocessing at all: the engine sees the page as scanned."""
    return img


def prep_up2(img: np.ndarray) -> np.ndarray:
    """2x INTER_CUBIC upscale, no binarisation."""
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def prep_up2_th200(img: np.ndarray) -> np.ndarray:
    """The baseline: 2x INTER_CUBIC then a fixed threshold of 200."""
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


PREP = {
    "none": prep_none,
    "up2": prep_up2,
    "up2_th200": prep_up2_th200,
}


# --------------------------------------------------------------- recognition
@dataclass
class Row:
    """One detected text line, with its text bound to its own box."""
    text: str
    poly: np.ndarray          # (4, 2) in the coordinates of the image passed
    score: float
    cx: float
    cy: float
    w: float
    h: float


def recognise_page(engine, img: np.ndarray) -> list[Row]:
    """Detect and recognise a whole page, pairing text with its own box.

    The pairing is done before any sorting, so the text always belongs to
    the box it was recognised from. Rows are returned in detector order;
    callers that need reading order sort the rows themselves.
    """
    result = engine(img)
    if not result:
        return []
    boxes = np.asarray(result.boxes, dtype=float)
    txts = list(getattr(result, "txts", []) or [])
    scores = list(getattr(result, "scores", []) or [])
    rows: list[Row] = []
    for i, poly in enumerate(boxes):
        if i >= len(txts):
            break
        text = txts[i] or ""
        if not text.strip():
            continue
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        xs = pts[:, 0]
        ys = pts[:, 1]
        rows.append(Row(
            text=text,
            poly=pts,
            score=float(scores[i]) if i < len(scores) else 0.0,
            cx=float(xs.mean()),
            cy=float(ys.mean()),
            w=float(xs.max() - xs.min()),
            h=float(ys.max() - ys.min()),
        ))
    return rows


# ---------------------------------------------------------------- attribution
@dataclass
class Cell:
    """One compound's printed-SMILES cell rectangle, in page coordinates."""
    cmpd_id: str
    page: str
    x1: float
    y1: float
    x2: float
    y2: float
    n_lines: int


def load_cells() -> dict[str, list[Cell]]:
    """Per page, the compound cell rectangles from the audit manifest."""
    out: dict[str, list[Cell]] = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no":
                continue
            page = r["page"]
            out.setdefault(page, []).append(Cell(
                cmpd_id=r["cmpd_id"],
                page=page,
                x1=float(r["bbox_x1"]),
                y1=float(r["bbox_y1"]),
                x2=float(r["bbox_x2"]),
                y2=float(r["bbox_y2"]),
                n_lines=int(r["n_lines"]),
            ))
    for v in out.values():
        v.sort(key=lambda c: c.y1)
    return out


def attribute(rows: list[Row], cells: list[Cell], scale: float) -> dict[str, list[Row]]:
    """Bucket page-level rows into compounds using the manifest rectangles.

    ``scale`` converts from the coordinates of the image that was recognised
    back to page coordinates (1.0 for the raw page, 0.5 for a 2x page).
    A row belongs to a compound when its centre falls inside that
    compound's rectangle. The SMILES column is the right-hand part of the
    rectangle, so the structure column is excluded by the same test.
    """
    out: dict[str, list[Row]] = {}
    for c in cells:
        picked = []
        for r in rows:
            cx = r.cx * scale
            cy = r.cy * scale
            if not (c.x1 <= cx <= c.x2):
                continue
            if not (c.y1 <= cy <= c.y2):
                continue
            picked.append(r)
        picked.sort(key=lambda r: r.cy * scale)
        out[c.cmpd_id] = picked
    return out


def joined(rows: list[Row]) -> str:
    """Concatenate a compound's rows in reading order."""
    return "".join(r.text.strip() for r in rows if r.text.strip())


# ------------------------------------------------------------------- control
def _control() -> None:
    """Report how far the detector's emission order differs from reading order."""
    from rapidocr import RapidOCR
    img = cv2.imread(str(GOLD / "PCTCN2025075389-ftappb-I100297.jpg"))
    det = RapidOCR(params={})(img)
    boxes = np.asarray(det.boxes, dtype=float)
    order = sorted(range(len(boxes)),
                   key=lambda i: (float(boxes[i][:, 1].min()),
                                  float(boxes[i][:, 0].min())))
    bad = [k for k, i in enumerate(order) if i != k]
    print(f"detector order differs from sorted order at {len(bad)} of "
          f"{len(boxes)} slots")
    if bad:
        print("  -> a helper that zips sorted boxes with unsorted txts "
              "would mis-attribute these rows")
    print("this module zips boxes with txts before sorting, so each row "
          "keeps its own text")


if __name__ == "__main__":
    _control()
