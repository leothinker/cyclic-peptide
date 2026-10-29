"""Audit the geometric quality of every printed-SMILES text cell.

Read-only with respect to the pipeline: this module re-derives geometry,
it does not change the cropper, the recogniser or the mass gate.

Coordinate chain established for this corpus (verified by pixel
comparison, see coordinate_reconciliation.txt):

    page image      bbox_x1..bbox_x2          data/raw/.../golden_tables
    cell crop       identical to page[bbox]  data/audit/phase2/cells_text
    processed image cell crop * 2            851x625 -> 1702x1250
    detector box    in processed coords

so  processed_x = (page_x - bbox_x1) * 2  and likewise for y.

Two earlier experiments reported conflicting ink_x1 values for cmpd 1036
because they used different ink definitions. Both are reproduced here
side by side rather than reconciled by assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PHASE2 = REPO / "data" / "audit" / "phase2"
CELLS = PHASE2 / "cells_text"
OUT = PHASE2 / "row_crop_audit"

INK = 128           # grey level below which a pixel counts as ink
RULE_FRAC = 0.60    # a column inked over this fraction of the cell is a rule
RULE_PAD = 2        # dilation applied to a rule before excluding it
CONTACT_PX = 3      # ink this close to a cell edge counts as edge contact


def rule_columns(gray: np.ndarray) -> np.ndarray:
    """Columns that are inked over most of the full cell height.

    These are the table's printed vertical rules, not glyphs. They must be
    removed before any statement about where a row's first inked column is,
    otherwise the cell's own border is read as a clipped character.
    """
    h = gray.shape[0]
    col_ink = (gray < INK).sum(axis=0)
    return np.where(col_ink > h * RULE_FRAC)[0]


def dilate(cols, pad: int, width: int) -> set[int]:
    """Expand a set of column indices by +-pad, clipped to the image."""
    out: set[int] = set()
    for c in cols:
        for d in range(-pad, pad + 1):
            v = int(c) + d
            if 0 <= v < width:
                out.add(v)
    return out


def ink_extent_full(gray, y1, y2):
    """Left/right inked columns of a row band, full image width.

    This is the definition used by the OLD experiment
    (_phase2_box_evidence.py): it scans the whole row, so it can see ink
    anywhere on the printed line, including ink to the left of the detector
    box. Columns whose ink density exceeds half the band height are dropped
    as rules.
    """
    y1 = max(0, y1 - 3)
    y2 = min(gray.shape[0], y2 + 3)
    band = gray[y1:y2]
    if band.size == 0:
        return None
    col = (band < INK).sum(axis=0)
    cols = np.where((col > 0) & (col <= band.shape[0] * 0.5))[0]
    if not len(cols):
        return None
    return int(cols.min()), int(cols.max())


def ink_extent_boxed(gray, x1, y1, x2, y2, excluded, window=30):
    """Ink extent restricted to box + -window, with rules removed.

    This is the definition used by the NEW experiment
    (phase2_recog_candidates.py). Note the hard cap: if the true ink starts
    further left than the window, this function CANNOT see it and the
    reported ink_x1 silently becomes the window edge.
    """
    h, w = gray.shape
    lo, hi = max(0, x1 - window), min(w, x2 + window)
    y1 = max(0, y1)
    y2 = min(h, y2 + 1)
    band = gray[y1:y2, lo:hi]
    if band.size == 0:
        return None
    cols = [
        lo + int(c)
        for c in np.where((band < INK).any(axis=0))[0]
        if (lo + int(c)) not in excluded
    ]
    if not cols:
        return None
    return min(cols), max(cols)


def ink_extent_row(gray, y1, y2, excluded):
    """Ink extent of a row band over the FULL width, rules removed.

    Same as :func:`ink_extent_full` except the rule columns are removed by
    explicit exclusion rather than by an implicit density test, and there is
    no search window. This is the only one of the three definitions that is
    not censored, so it is the one that can settle how much of a row the
    detector box actually overhangs.
    """
    y1 = max(0, y1 - 3)
    y2 = min(gray.shape[0], y2 + 3)
    band = gray[y1:y2]
    if band.size == 0:
        return None
    cols = [
        int(c)
        for c in np.where((band < INK).any(axis=0))[0]
        if int(c) not in excluded
    ]
    if not cols:
        return None
    return min(cols), max(cols)


def box_rect(box: np.ndarray) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def line_boxes(result) -> list[np.ndarray]:
    """Detector boxes sorted top-to-bottom then left-to-right."""
    boxes = np.asarray(result.boxes, dtype=float)
    order = sorted(
        range(len(boxes)),
        key=lambda i: (float(boxes[i][:, 1].min()), float(boxes[i][:, 0].min())),
    )
    return [boxes[i] for i in order]


def preprocess(img: np.ndarray, threshold: int = 200) -> np.ndarray:
    """The fixed OCR baseline preprocessing, reproduced exactly.

    2x INTER_CUBIC then a fixed binarisation threshold, so the geometry
    measured here is the geometry the recogniser actually saw.
    """
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(up, threshold, 255, cv2.THRESH_BINARY)
    return bw


def analyse_line(gray, poly, excluded) -> dict:
    """Per-line geometry for one detector box.

    Reports both ink definitions plus a censoring flag. It deliberately does
    NOT emit a boolean "truncated" verdict: whether ink just outside a box
    means a lost character is a visual question, and the acceptance criteria
    for this round forbid deciding it from pixel counts alone.
    """
    x1, y1, x2, y2 = box_rect(poly)
    full = ink_extent_full(gray, y1, y2)
    row = ink_extent_row(gray, y1, y2, excluded)
    boxed = ink_extent_boxed(gray, x1, y1, x2, y2, excluded)
    h, w = gray.shape
    rec: dict = {
        "box_x1": x1,
        "box_y1": y1,
        "box_x2": x2,
        "box_y2": y2,
        "proc_w": w,
        "proc_h": h,
        "ink_full_x1": "",
        "ink_full_x2": "",
        "out_left_full_px": "",
        "out_right_full_px": "",
        "ink_row_x1": "",
        "ink_row_x2": "",
        "out_left_row_px": "",
        "ink_box_x1": "",
        "ink_box_x2": "",
        "boxed_window_left": max(0, x1 - 30),
        "ink_box_censored": "",
    }
    if row:
        rec["ink_row_x1"] = row[0]
        rec["ink_row_x2"] = row[1]
        rec["out_left_row_px"] = x1 - row[0]
    else:
        rec["ink_row_x1"] = ""
        rec["ink_row_x2"] = ""
        rec["out_left_row_px"] = ""
    if full:
        rec["ink_full_x1"] = full[0]
        rec["ink_full_x2"] = full[1]
        rec["out_left_full_px"] = x1 - full[0]
        rec["out_right_full_px"] = full[1] - x2
    if boxed:
        rec["ink_box_x1"] = boxed[0]
        rec["ink_box_x2"] = boxed[1]
        rec["ink_box_censored"] = (
            "yes" if boxed[0] <= max(0, x1 - 30) else "no"
        )
    return rec


def edge_contact(gray, excluded) -> dict:
    """Which cell edges have ink touching them, with rules excluded.

    Cell-level, not line-level: a glyph running into the cell bbox edge means
    the cell crop itself may be too tight, which is a different defect from a
    detector box that is too tight inside a correctly cropped cell.
    """
    h, w = gray.shape
    mask = gray < INK
    keep = ~np.isin(np.arange(w), sorted(excluded))
    out: dict = {}
    n = min(CONTACT_PX, w)
    out["contact_left"] = "yes" if mask[:, :n][:, keep[:n]].any() else "no"
    out["contact_right"] = "yes" if mask[:, -n:][:, keep[-n:]].any() else "no"
    m = min(CONTACT_PX, h)
    out["contact_top"] = "yes" if mask[:m, :][:, keep].any() else "no"
    out["contact_bottom"] = "yes" if mask[-m:, :][:, keep].any() else "no"
    return out
