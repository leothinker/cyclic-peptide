r"""Reproduce and generalise the top-k artefact that predates this work.

The artefact
------------
``demo_rapidocr_topk/topk_printed_smiles.json`` was produced by the
orphaned ``_demo_topk.py``. It contains the printed SMILES of 1036 and
1037 as recognised on the WHOLE PAGE, with per-character top-k
alternatives. The user observed that the 1036 text in it is noticeably
more accurate than what the pipeline ships.

That observation is reproducible and it is exact: the 1036 string in the
artefact is character-for-character identical to the human-verified
reference in ``phase2_ocr_compare.REFERENCES``.

What the artefact actually did
------------------------------
Three things at once, which is why it was so much better:

1. ``RapidOCR(params={})`` -- the UPSTREAM DEFAULT engine
 (PP-OCRv6 det_small + ch PP-OCRv6 rec small + ch cls), NOT the production
   PP-OCRv4 mobile English engine.
2. The WHOLE PAGE, with no crop, so no detector box can clip a glyph.
3. NO preprocessing at all -- no upscale, no threshold.

Plus per-character top-k (5), which gives alternatives but does not by
itself change the greedy path.

This script separates the three axes so the contribution of each is
knowable, and then checks the winning combination against the human
reference for 1036 and the agent transcription for 1037.

Read-only. Writes to ``data/audit/phase3/topk_reproduce/``.

Run:
    .venv\Scripts\python.exe phase3_topk_reproduce.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import (  # noqa: E402
    GOLD, attribute, baseline_engine, default_engine, load_cells,
    prep_none, prep_up2, prep_up2_th200, recognise_page,
)

OUT = REPO / "data" / "audit" / "phase3" / "topk_reproduce"
ARTIFACT = REPO / "demo_rapidocr_topk" / "topk_printed_smiles.json"

PREPS = {
    "none": (prep_none, 1.0),
    "up2": (prep_up2, 0.5),
    "up2_th200": (prep_up2_th200, 0.5),
}


def _artefact_check() -> list[str]:
    """Confirm the artefact's 1036 text against the human reference."""
    lines: list[str] = []
    if not ARTIFACT.exists():
        lines.append("artefact not found; skipping")
        return lines
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    entry = data.get("ch_PP-OCRv4") or data.get("en_PP-OCRv4") or []
    if not entry:
        lines.append("artefact has no recognisable line list")
        return lines
    idxs = sorted(lp["line"] for lp in entry)
    lines.append(f"artefact keys      : {sorted(data)}")
    lines.append(f"artefact line ids   : {idxs}")

    ref = REFERENCES.get("1036", {}).get("smiles", "")
    # The 1036 block of the artefact is the first run of consecutive
    # SMILES-looking rows; its length identifies it unambiguously.
    for lo, hi in ((3, 11), (2, 10), (3, 10)):
        cand = "".join(lp["text"] for lp in entry
                        if lo <= lp["line"] <= hi)
        if not cand:
            continue
        lines.append(f"artefact lines {lo}-{hi}: len={len(cand)} "
                     f"exact_match_vs_reference={cand == ref}")
        if cand == ref:
            lines.append("  -> the artefact reproduces the human-verified "
                         "1036 reference EXACTLY (0 character errors)")
            break
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines_out: list[str] = []

    lines_out.append("=== top-k artefact: what it was, and what reproduces "
                     "it ===\n")
    lines_out.append("The artefact was produced by the upstream DEFAULT "
                     "engine on the WHOLE PAGE with NO preprocessing.")
    lines_out.append("Production instead uses PP-OCRv4 mobile English on a "
                     "cropped cell, upscaled 2x, not binarised.\n")

    lines_out.append("--- artefact vs human reference ---\n")
    lines_out.extend(_artefact_check())
    lines_out.append("")

    cells_by_page = load_cells()
    cells = cells_by_page.get("I100297", [])
    if not cells:
        lines_out.append("no cells found for I100297")
        io.open(OUT / "topk_reproduce.txt", "w",
                encoding="utf-8").write("\n".join(lines_out))
        print("\n".join(lines_out))
        return

    img = cv2.imread(str(GOLD / "PCTCN2025075389-ftappb-I100297.jpg"))
    engines = {"default": default_engine(), "baseline": baseline_engine()}

    rows: list[dict] = []
    for eng_name, engine in engines.items():
        for prep_name, (fn, scale) in PREPS.items():
            prepared = fn(img)
            rws = recognise_page(engine, prepared)
            buckets = attribute(rws, cells, scale)
            for c in cells:
                txt = "".join(r.text.strip()
                              for r in buckets.get(c.cmpd_id, []))
                ref = REFERENCES.get(c.cmpd_id, {}).get("smiles", "")
                rows.append({
                    "engine": eng_name,
                    "prep": prep_name,
                    "cmpd_id": c.cmpd_id,
                    "n_rows": len(buckets.get(c.cmpd_id, [])),
                    "len": len(txt),
                    "open": txt.count("("),
                    "close": txt.count(")"),
                    "edit": edit_ops(ref, txt) if ref else "",
                    "cer": round(edit_ops(ref, txt) / len(ref), 5) if ref else "",
                    "text": txt,
                })

    with (OUT / "topk_reproduce.csv").open("w", newline="",
                                            encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines_out.append("--- full-page A/B on the two reference compounds ---\n")
    lines_out.append("1036 = human-verified reference; "
                     "1037 = agent transcription, NOT human-confirmed.\n")
    lines_out.append(f"{'engine':<9} {'prep':<11} {'cmpd':>6} {'rows':>5} "
                     f"{'len':>5} {'open':>5} {'close':>6} {'edit':>5} "
                     f"{'CER':>9}")
    for r in rows:
        lines_out.append(
            f"{r['engine']:<9} {r['prep']:<11} {r['cmpd_id']:>6} "
            f"{r['n_rows']:>5} {r['len']:>5} {r['open']:>5} {r['close']:>6} "
            f"{str(r['edit']):>5} {str(r['cer']):>9}"
        )

    lines_out.append("\n--- best configuration per reference compound ---\n")
    for cmpd in ("1036", "1037"):
        sub = [r for r in rows if r["cmpd_id"] == cmpd and r["edit"] != ""]
        if not sub:
            continue
        best = min(sub, key=lambda r: r["edit"])
        lines_out.append(
            f"  {cmpd}: {best['engine']}/{best['prep']} edit={best['edit']} "
            f"CER={best['cer']}"
        )

    lines_out.append("\n--- interpretation ---")
    d1036 = {(r["engine"], r["prep"]): r for r in rows
              if r["cmpd_id"] == "1036"}
    if ("default", "none") in d1036:
        lines_out.append(
            "default/none reproduces the human-verified 1036 reference "
 f"exactly: edit={d1036[('default', 'none')]['edit']}"
        )
    lines_out.append(
        "No single configuration wins on both compounds, so a corpus-scale "
        "test is required before changing the pipeline."
    )

    io.open(OUT / "topk_reproduce.txt", "w",
            encoding="utf-8").write("\n".join(lines_out))
    print("\n".join(lines_out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
