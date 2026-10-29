r"""Task 2, second arm: pad each crop by the MEASURED overhang.

The fixed-pad arm in ``pre_report_ab_recrop.py`` turned out to be the wrong
instrument. The measured overhang is not a constant: across the 215
confirmed events it ranges from 24 to 36 px. A fixed 30 px therefore
covers the whole excluded glyph in only 37 of 215 cases, which is why the
fixed-pad results behave oddly and non-monotonically.

This arm sets pad = the overhang measured for that specific event, so the
new left edge lands exactly on the first clean ink column. It is the only
arm that can answer the real question: given that the glyph's pixels are
back in the crop, does the recogniser read '=' ?

Same fixed preprocessing, same rectification, same recognition entry point.
No white padding.

Run:
    .venv\Scripts\python.exe pre_report_ab_exact.py
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

from pre_report_ab_recrop import (  # noqa: E402
    crop_line,
    digest,
    first_char_class,
    load_cell,
)

OUTDIR = REPO / "data" / "audit" / "phase2" / "pre_report_validation"


def main() -> None:
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
    step = max(1, len(controls) // 20)
    controls = controls[::step][:20]

    engine = build_engine()
    rows = []
    for n, r in enumerate(confirmed, 1):
        rows += _run(engine, r, int(r["line_no"]), int(r["overhang_px"]),
                     "=", "truncated")
        if n % 40 == 0:
            print(f"  {n}/{len(confirmed)}", flush=True)
    for r in controls:
        rows += _run(engine, {
            "event_id": f"CTRL_{r['cmpd_id']}_L0",
            "cmpd_id": r["cmpd_id"], "page": r["page"],
            "crop_path": r["crop_path"],
            "overhang_px": 0,
        }, 0, 0, "", "control")

    with (OUTDIR / "ab_exact_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {OUTDIR / 'ab_exact_results.csv'}")


def _run(engine, rec, line_no, overhang, correct_char, group):
    proc, polys, excluded = load_cell(engine, rec["crop_path"])
    if proc is None or line_no >= len(polys):
        return []
    from row_crop_audit import box_rect
    poly = polys[line_no]
    x1 = box_rect(poly)[0]
    base, _ = crop_line(proc, poly, 0)
    text_a = rec_only(engine, [base])
    line, applied = crop_line(proc, poly, overhang)
    new_x1 = x1 - applied
    text = rec_only(engine, [line])
    ref = REFERENCES.get(rec["cmpd_id"], {}).get("smiles")
    row = {
        "event_id": rec["event_id"],
        "cmpd_id": rec["cmpd_id"],
        "page": rec["page"],
        "line_no": line_no,
        "group": group,
        "overhang_px": overhang,
        "pad_applied_px": applied,
        "new_box_x1": new_x1,
        "left_edge_in_cell_rule": "yes" if new_x1 in excluded else "no",
        "crop_w": int(line.shape[1]),
        "crop_digest": digest(line),
        "text_A": text_a,
        "text_B_exact": text,
        "len_A": len(text_a),
        "len_B": len(text),
        "first_char_A": first_char_class(text_a),
        "first_char_B": first_char_class(text),
        "correct_first_char": correct_char,
        "first_char_correct": (
            "yes" if first_char_class(text) == "equals" else "no"
        ) if group == "truncated" else "na",
        "changed_vs_A": "yes" if text != text_a else "no",
        "ref_available": "yes" if ref else "no",
    }
    if ref:
        d = edit_ops(ref, text)
        row["edit_distance_B_vs_ref"] = d
        row["cer_B_vs_ref"] = round(d / len(ref), 5)
        da = edit_ops(ref, text_a)
        row["edit_distance_A_vs_ref"] = da
        row["cer_A_vs_ref"] = round(da / len(ref), 5)
    else:
        row["edit_distance_B_vs_ref"] = ""
        row["cer_B_vs_ref"] = ""
        row["edit_distance_A_vs_ref"] = ""
        row["cer_A_vs_ref"] = ""
    return [row]


if __name__ == "__main__":
    main()
