r"""Task 1: per-event visual review of the suspected row-start truncations.

Reads data/audit/phase2/row_crop_audit/manifest.csv, expands every
glyph_excluded event into an individual reviewable record, and writes a
four-panel evidence strip per event:

    1  page context      the row as printed, with the cell bbox marked
    2  geometry          the same strip with the detector box drawn and the
                         excluded ink highlighted
    3  what was fed in   exactly the pixels the recogniser received
    4  re-crop           the same row re-cut from the parent image with the
                         left edge extended

Status semantics, deliberately conservative:

    confirmed_truncation  a human confirmed the excluded ink is this row's
                          own leading '='
    not_truncated         a human confirmed the excluded ink belongs to
                          something else (rule, neighbouring row, noise)
    ambiguous             cannot be decided from the image
    unreviewed            not yet looked at

This script NEVER assigns confirmed_truncation on its own. It emits
``unreviewed`` for everything and produces the evidence needed for a human
(or a later pass) to promote a record. That is the whole point: the
acceptance criteria forbid confirming truncation from the 8-row signature
alone.

Read-only with respect to the pipeline.

Run:
    .venv\Scripts\python.exe pre_report_review.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    build_engine,
)

from row_crop_audit import (  # noqa: E402
    CELLS,
    INK,
    OUT,
    RULE_FRAC,
    RULE_PAD,
    box_rect,
    dilate,
    line_boxes,
    preprocess,
)
from row_crop_audit_batch import overhang_profile  # noqa: E402

OUTDIR = REPO / "data" / "audit" / "phase2" / "pre_report_validation"
GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"

RECROP_PAD = 40          # px, in processed coords, for panel 4
LEFT_PAD = 30            # px of page context to show left of the box
RIGHT_PAD = 210          # px of row to show right of the box

RED = (0, 0, 255)
GREEN = (0, 170, 0)
BLUE = (255, 120, 0)


def page_crop_for(rec):
    """Read the ORIGINAL page image and return it, or None."""
    path = GOLD / rec["source_image"]
    img = cv2.imread(str(path))
    return img


def events_for(rec, polys, gray, excluded):
    """Expand one manifest row into its individual glyph_excluded events."""
    out = []
    for li, poly in enumerate(polys):
        for side in ("left", "right"):
            px, ncol, nrow, label = overhang_profile(
                gray, poly, excluded, side
            )
            if label != "glyph_excluded":
                continue
            x1, y1, x2, y2 = box_rect(poly)
            out.append({
                "line_no": li,
                "side": side,
                "overhang_px": px,
                "ink_cols": ncol,
                "ink_rows": nrow,
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
            })
    return out


def strip_panels(rec, ev, proc_gray, page):
    """Four-panel evidence strip for one event."""
    li = ev["line_no"]
    x1, y1, x2, y2 = ev["box_x1"], ev["box_y1"], ev["box_x2"], ev["box_y2"]
    y0 = max(0, y1 - 8)
    y1b = min(proc_gray.shape[0], y2 + 8)
    c0 = max(0, x1 - LEFT_PAD)
    c1 = min(proc_gray.shape[1], x1 + RIGHT_PAD)

    p1 = proc_gray[y0:y1b, c0:c1]

    p2 = cv2.cvtColor(p1, cv2.COLOR_GRAY2BGR)
    cv2.line(p2, (x1 - c0, 0), (x1 - c0, p2.shape[0] - 1), RED, 2)
    cv2.line(p2, (x2 - c0, 0), (x2 - c0, p2.shape[0] - 1), RED, 1)
    _mark_excluded(p2, ev, x1, c0)

    p3 = proc_gray[y1:y2 + 1, x1:x2]

    p4 = proc_gray[y0:y1b, max(0, c0 - RECROP_PAD):c1]

    del page
    return [p1, p2, p3, p4]


def _mark_excluded(img, ev, x1, c0):
    """Shade the ink that sits outside the box on the event's side."""
    lo = x1 - ev["overhang_px"] - c0
    hi = x1 - c0
    if ev["side"] == "right":
        lo, hi = x1 - c0, x1 - c0 + ev["overhang_px"]
    lo = max(0, lo)
    hi = min(img.shape[1], hi)
    if hi > lo:
        overlay = img.copy()
        overlay[:, lo:hi] = (0, 190, 255)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.rectangle(img, (lo, 0), (hi - 1, img.shape[0] - 1), BLUE, 1)


def _fit(img, th=132):
    if img is None or img.size == 0:
        return np.full((th, th, 3), 255, np.uint8)
    k = max(1, int(round(th / max(img.shape[0], 1))))
    up = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    if up.ndim == 2:
        up = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    if up.shape[0] != th:
        if up.shape[0] > th:
            t = (up.shape[0] - th) // 2
            up = up[t:t + th]
        else:
            up = cv2.copyMakeBorder(
                up, 0, th - up.shape[0], 0, 0,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
    return up


def _caption(img, text):
    bar = np.full((19, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


PANEL_NAMES = ("1_page_row", "2_geometry", "3_as_fed", "4_recrop_left")


def assemble(panels, rec, ev):
    tiles = [_caption(_fit(p), n) for p, n in zip(panels, PANEL_NAMES)]
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    body = np.vstack(tiles)
    head = np.full((22, body.shape[1], 3), 255, np.uint8)
    txt = (f"{rec['page']} cmpd {rec['cmpd_id'] or '(header)'} line "
           f"{ev['line_no']} {ev['side']} overhang {ev['overhang_px']}px "
           f"box x1={ev['box_x1']}")
    cv2.putText(head, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, body])


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
           / "manifest.csv").open(newline="", encoding="utf-8-sig") as fh:
        manifest = list(csv.DictReader(fh))
    with (CELLS / "cells_text_meta.csv").open(
        newline="", encoding="utf-8-sig"
    ) as fh:
        meta = {r["cmpd_id"]: r for r in csv.DictReader(fh)}

    engine = build_engine()
    rows = []
    evdir = OUTDIR / "evidence"
    evdir.mkdir(exist_ok=True)
    for n, rec in enumerate(manifest, 1):
        if rec["is_header"] == "yes":
            continue
        if int(rec["n_lines_glyph_excluded"] or 0) <= 0:
            continue
        m = meta.get(rec["cmpd_id"])
        if m is None:
            continue
        img = cv2.imread(str(REPO / "data" / "audit" / "phase2"
                            / m["crop_path"]))
        if img is None:
            continue
        proc = preprocess(img, BASELINE_THRESHOLD)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        rules = np.where((gray < INK).sum(axis=0) > h * RULE_FRAC)[0]
        excluded = dilate(rules, RULE_PAD, w)
        polys = line_boxes(engine(proc))
        page = page_crop_for(rec)
        for k, ev in enumerate(events_for(rec, polys, gray, excluded)):
            ev_id = f"{rec['cmpd_id']}_L{ev['line_no']}_{ev['side'][0]}"
            ev["event_id"] = ev_id
            ev["page"] = rec["page"]
            ev["cmpd_id"] = rec["cmpd_id"]
            ev["source_image"] = rec["source_image"]
            ev["crop_path"] = rec["crop_path"]
            ev["bbox"] = (f"{rec['bbox_x1']},{rec['bbox_y1']},"
                          f"{rec['bbox_x2']},{rec['bbox_y2']}")
            ev["proc_scale"] = 2
            ev["evidence"] = f"evidence/{ev_id}.png"
            strip = assemble(strip_panels(rec, ev, gray, page), rec, ev)
            cv2.imwrite(str(evdir / f"{ev_id}.png"), strip)
            rows.append(ev)
        if n % 25 == 0:
            print(f"  ...{n}/{len(manifest)}", flush=True)

    cols = [
        "event_id", "cmpd_id", "page", "source_image", "crop_path",
        "bbox", "proc_scale", "line_no", "side", "box_x1", "box_y1",
        "box_x2", "box_y2", "overhang_px", "ink_cols", "ink_rows",
        "status", "judgement_basis", "evidence", "reviewer", "reviewed_at",
    ]
    with (OUTDIR / "review_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["status"] = "unreviewed"
            r["judgement_basis"] = ""
            r["reviewer"] = ""
            r["reviewed_at"] = ""
            w.writerow(r)
    print(f"{len(rows)} events -> {OUTDIR / 'review_manifest.csv'}")
    print(f"evidence strips -> {evdir}")


if __name__ == "__main__":
    main()
