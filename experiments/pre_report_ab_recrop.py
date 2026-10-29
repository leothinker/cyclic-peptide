r"""Task 2: A/B experiment for left-extended re-cropping.

Group A is the current pipeline: the detector box exactly as the detector
produced it. Group B re-cuts the same row from the same parent image with
the box's LEFT edge moved outward, so the excluded glyph's real pixels
re-enter the crop. White padding is never used: it cannot restore a glyph
that was never in the crop, and pretending otherwise would manufacture an
improvement that does not exist.

Everything else is held fixed:

    preprocessing  2x INTER_CUBIC then fixed binarisation at 200
    rectification  get_rotate_crop_image, the official perspective path
    recognition    engine.recognize_txt, the official entry point
    model/config   whatever build_engine() returns, untouched

Only the crop region varies. Each run records the crop width and a digest of
the crop pixels so that claim can be checked rather than trusted.

Safety: a requested extension is clamped so the new left edge cannot leave
the image, and a run is flagged when that clamped edge lands inside the
cell's own vertical rule, because that adds grid contamination and is
reported rather than hidden.

Read-only with respect to the pipeline.

Run:
    .venv\Scripts\python.exe pre_report_ab_recrop.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_box_expansion import rec_only  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    build_engine,
)

from row_crop_audit import (  # noqa: E402
    INK,
    RULE_FRAC,
    RULE_PAD,
    box_rect,
    dilate,
    line_boxes,
    preprocess,
)

OUTDIR = REPO / "data" / "audit" / "phase2" / "pre_report_validation"
PADS = (0, 16, 30, 36, 40)
CONTROL_CELLS = 20


def crop_line(proc, poly, pad_left):
    """Re-cut one row with its left edge moved outward by pad_left px.

    Mirrors the official ``extract_line`` but on the left-hand corners. The
    requested pad is clamped so the edge cannot leave the image, and the
    applied value is returned so the clamp is visible in the results.
    """
    pts = np.array(poly, dtype=np.float32).copy()
    applied = int(max(0, min(int(pad_left), int(float(pts[0][0])))))
    for i in (0, 3):
        pts[i][0] = max(0.0, float(pts[i][0]) - applied)
    if pts[3][0] > pts[2][0]:
        pts[3][0] = pts[2][0]
    return get_rotate_crop_image(proc, pts), applied


def digest(img):
    return hashlib.sha1(img.tobytes()).hexdigest()[:12]


def first_char_class(text):
    if not text:
        return "missing"
    c = text[0]
    if c == "=":
        return "equals"
    if c == "-":
        return "minus"
    return f"other:{c}"


def load_cell(engine, crop_rel):
    img = cv2.imread(str(REPO / "data" / "audit" / "phase2" / crop_rel))
    if img is None:
        return None, None, None
    proc = preprocess(img, BASELINE_THRESHOLD)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rules = np.where((gray < INK).sum(axis=0) > h * RULE_FRAC)[0]
    excluded = dilate(rules, RULE_PAD, w)
    return proc, line_boxes(engine(proc)), excluded


def run_event(engine, rec, line_no, correct_char, group):
    """Recognise one row under every pad and describe the outcome."""
    proc, polys, excluded = load_cell(engine, rec["crop_path"])
    if proc is None or line_no >= len(polys):
        return []
    poly = polys[line_no]
    x1 = box_rect(poly)[0]
    out = []
    for pad in PADS:
        line, applied = crop_line(proc, poly, pad)
        new_x1 = x1 - applied
        text = rec_only(engine, [line])
        out.append({
            "event_id": rec["event_id"],
            "cmpd_id": rec["cmpd_id"],
            "page": rec["page"],
            "line_no": line_no,
            "group": group,
            "pad_requested_px": pad,
            "pad_applied_px": applied,
            "pad_clamped": "yes" if applied != pad else "no",
            "new_box_x1": new_x1,
            "left_edge_in_cell_rule": "yes" if new_x1 in excluded else "no",
            "crop_w": int(line.shape[1]),
            "crop_h": int(line.shape[0]),
            "crop_digest": digest(line),
            "text": text,
            "text_len": len(text),
            "first_char_class": first_char_class(text),
            "correct_first_char": correct_char,
        })
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "review_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as fh:
        rev = list(csv.DictReader(fh))
    confirmed = [r for r in rev if r["status"] == "confirmed_truncation"]

    with (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
           / "manifest.csv").open(newline="", encoding="utf-8-sig") as fh:
        audit = list(csv.DictReader(fh))
    controls = [r for r in audit
                if r["is_header"] == "no"
                and int(r["n_lines_glyph_excluded"] or 0) == 0]
    step = max(1, len(controls) // CONTROL_CELLS)
    controls = controls[::step][:CONTROL_CELLS]

    engine = build_engine()
    rows = []
    for n, r in enumerate(confirmed, 1):
        rows += run_event(engine, r, int(r["line_no"]), "=", "truncated")
        if n % 25 == 0:
            print(f"  truncated {n}/{len(confirmed)}", flush=True)
    for r in controls:
        rows += run_event(engine, {
            "event_id": f"CTRL_{r['cmpd_id']}_L0",
            "cmpd_id": r["cmpd_id"],
            "page": r["page"],
            "crop_path": r["crop_path"],
        }, 0, "", "control")
    print(f"  controls {len(controls)}")

    _finish(rows)
    with (OUTDIR / "ab_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {OUTDIR / 'ab_results.csv'}")


def _finish(rows) -> None:
    """Add the comparison columns that depend on group A."""
    by_event: dict[str, dict] = {}
    for r in rows:
        by_event.setdefault(r["event_id"], {})[r["pad_requested_px"]] = r
    for runs in by_event.values():
        a = runs.get(0)
        if a is None:
            continue
        for r in runs.values():
            r["changed_vs_A"] = "yes" if r["text"] != a["text"] else "no"
            if r["group"] == "truncated":
                r["first_char_correct"] = (
                    "yes" if r["first_char_class"] == "equals" else "no"
                )
            else:
                r["first_char_correct"] = "na"
        for pad, r in sorted(runs.items()):
            if pad == 0:
                r["verdict_vs_A"] = "baseline"
            elif r["group"] != "truncated":
                r["verdict_vs_A"] = (
                    "changed" if r["changed_vs_A"] == "yes" else "unchanged"
                )
            elif r["first_char_class"] == "equals":
                r["verdict_vs_A"] = "improved"
            elif r["first_char_class"] == "minus":
                r["verdict_vs_A"] = "still_minus"
            else:
                r["verdict_vs_A"] = "no_improvement"
    ref = REFERENCES.get("1036", {}).get("smiles")
    for r in rows:
        if ref and r["cmpd_id"] == "1036":
            r["ref_available"] = "yes"
            d = edit_ops(ref, r["text"])
            r["edit_distance_vs_ref"] = d
            r["cer_vs_ref"] = round(d / len(ref), 5)
        else:
            r["ref_available"] = "no"
            r["edit_distance_vs_ref"] = ""
            r["cer_vs_ref"] = ""


if __name__ == "__main__":
    main()
