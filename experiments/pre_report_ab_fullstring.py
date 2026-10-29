r"""Task 2, third arm: full-string CER for 1036, and a real control test.

Two gaps in the earlier arms are closed here.

1. The per-event arm compared ONE row's text against the full 271-character
   reference, which makes an edit distance meaningless. For 1036 the only
   sample with a human-verified reference, this arm re-cuts the offending
   row, joins all seven rows exactly as the pipeline does, and reports CER
   and edit distance on the complete string.

2. The control arm in the exact-pad run used pad 0, because clean cells have
   zero measured overhang, so it tested nothing. Here controls are widened
   by a realistic 30 px, which is what production would do to them if the
   extension were applied globally. That is the regression test.

Same fixed preprocessing, rectification and recognition entry point.

Run:
    .venv\Scripts\python.exe pre_report_ab_fullstring.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_box_expansion import rec_only  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import BASELINE_THRESHOLD, build_engine  # noqa: E402

from pre_report_ab_recrop import crop_line, first_char_class, load_cell  # noqa: E402

OUTDIR = REPO / "data" / "audit" / "phase2" / "pre_report_validation"
CID = "1036"
PADS = (0, 16, 30, 34, 36, 40)
CONTROL_PAD = 30


def joined(engine, crop_path, line_no, pad):
    """Recognise every row, widening only ``line_no``, then join."""
    proc, polys, _ = load_cell(engine, crop_path)
    lines = []
    for i, poly in enumerate(polys):
        p = pad if i == line_no else 0
        line, _applied = crop_line(proc, poly, p)
        lines.append(line)
    return rec_only(engine, lines), len(polys)


def main() -> None:
    engine = build_engine()
    ref = REFERENCES[CID]["smiles"]
    crop = "cells_text/I100297_1036_text.png"

    rows = []
    for pad in PADS:
        text, n = joined(engine, crop, 2, pad)
        d = edit_ops(ref, text)
        rows.append({
            "kind": "1036_full_string",
            "cmpd_id": CID,
            "page": "I100297",
            "line_widened": 2,
            "pad_px": pad,
            "n_lines": n,
            "text_len": len(text),
            "first_char_class": first_char_class(text),
            "edit_distance_vs_ref": d,
            "cer_vs_ref": round(d / len(ref), 5),
            "ref_available": "yes",
            "text": text,
        })
        print(f"  pad {pad:>2d}: edits={d}  CER={d/len(ref):.5f}  "
              f"first={rows[-1]['first_char_class']}")

    with (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
           / "manifest.csv").open(newline="", encoding="utf-8-sig") as fh:
        audit = list(csv.DictReader(fh))
    controls = [r for r in audit
                if r["is_header"] == "no"
                and int(r["n_lines_glyph_excluded"] or 0) == 0]
    step = max(1, len(controls) // 20)
    controls = controls[::step][:20]
    for r in controls:
        t0, _ = joined(engine, r["crop_path"], 0, 0)
        t1, _ = joined(engine, r["crop_path"], 0, CONTROL_PAD)
        rows.append({
            "kind": "control_regression",
            "cmpd_id": r["cmpd_id"],
            "page": r["page"],
            "line_widened": 0,
            "pad_px": CONTROL_PAD,
            "n_lines": "",
            "text_len": len(t1),
            "first_char_class": first_char_class(t1),
            "edit_distance_vs_ref": "",
            "cer_vs_ref": "",
            "ref_available": "no",
            "changed_vs_baseline": "yes" if t0 != t1 else "no",
            "baseline_text": t0,
            "text": t1,
        })
    changed = sum(1 for x in rows
                  if x.get("changed_vs_baseline") == "yes")
    n_ctrl = sum(1 for x in rows if x["kind"] == "control_regression")
    print(f"  controls changed: {changed}/{n_ctrl}")

    cols = []
    for r_ in rows:
        for k in r_:
            if k not in cols:
                cols.append(k)
    with (OUTDIR / "ab_fullstring.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUTDIR / 'ab_fullstring.csv'}")


if __name__ == "__main__":
    main()
