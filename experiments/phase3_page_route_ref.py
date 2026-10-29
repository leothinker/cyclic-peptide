r"""Character accuracy of the whole-page route on the reference set.

Why
---
The corpus screen is reference-free, so it cannot say whether the
characters are right. This script runs the leading whole-page
configuration on the pages that carry a reference transcription and
reports the edit distance and CER for each reference compound.

The route is the one measured in ``phase3_page_scale.py``: the whole
page is recognised with the upstream default engine after a 2x
upscale, and the recognised rows are attributed to compounds inside
the production SMILES column. The attribution band is clipped to that
column because the audit manifest's box runs 20 px past it, and whole
page recognition picks up that strip as text.

Reference status
----------------
    1036           human-verified transcription. The only reference
                   that can carry a correctness claim.
    1037, 1038,
    1040, 2173     agent transcriptions with known misreads. Reported
                   for completeness; a non-zero distance against them
                   is not evidence that the OCR is wrong.

Read-only. Writes to ``data/audit/phase3/page_route_ref/``.

Run:
    .venv\Scripts\python.exe phase3_page_route_ref.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import Cell, attribute  # noqa: E402
from phase3_page_extract import recognise_page  # noqa: E402
from phase3_page_scale import _engines  # noqa: E402
from phase3_page_scale import _load_manifest  # noqa: E402
from phase3_page_scale import _prep  # noqa: E402
from phase3_page_scale import _smiles_column  # noqa: E402
from phase3_quality_screen import screen_joined  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "page_route_ref"
GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")

ARM = "default/up2"
HUMAN_VERIFIED = {"1036"}


def _manifest():
    """Audit cells with the printed row count attached."""
    out = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no" or r["extraction_status"] != "ok":
                continue
            out.setdefault(r["page"], []).append({"cmpd_id": r["cmpd_id"], "x1": float(r["bbox_x1"]), "y1": float(r["bbox_y1"]), "x2": float(r["bbox_x2"]), "y2": float(r["bbox_y2"]), "n_lines": int(r["n_lines"])})
    for v in out.values():
        v.sort(key=lambda c: c["y1"])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _manifest()
    wanted = set(REFERENCES)

    lines = []
    lines.append("=== whole-page route, character accuracy on the "
                 "reference set ===")
    lines.append("")
    lines.append(f"arm: {ARM}  (whole page, 2x upscale, default engine, "
                 "rows attributed inside the production SMILES column)")
    lines.append("")
    lines.append(f"{'cmpd':<6} {'ref status':<19} {'edit':>5} {'cer':>8} "
                 f"{'len':>5} {'ref':>5} {'rows':>7} {'bal':>5} {'parse':>6}")

    out_rows = []
    for page in sorted(cells_by_page):
        cells = cells_by_page[page]
        targets = [c for c in cells if c["cmpd_id"] in wanted]
        if not targets:
            continue
        img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
        if img is None:
            continue
        col_x1, col_x2 = _smiles_column(img)
        prepared, scale = _prep(img, "up2")
        rws = recognise_page(_engines("default"), prepared)
        page_cells = []
        for c in targets:
            page_cells.append(Cell(cmpd_id=c["cmpd_id"], page=page, x1=max(c["x1"], col_x1), y1=c["y1"], x2=min(c["x2"], col_x2), y2=c["y2"], n_lines=c["n_lines"]))
        buckets = attribute(rws, page_cells, scale)
        for c in targets:
            picked = buckets.get(c["cmpd_id"], [])
            txt = "".join(r.text.strip() for r in picked)
            ref = REFERENCES[c["cmpd_id"]]["smiles"]
            s = screen_joined(c["cmpd_id"], txt)
            edit = edit_ops(ref, txt)
            verified = c["cmpd_id"] in HUMAN_VERIFIED
            lines.append(f"{c['cmpd_id']:<6} {'human_verified' if verified else 'agent_transcription':<19} {edit:>5} {edit / len(ref):>8.5f} {len(txt):>5} {len(ref):>5} {len(picked):>3}/{c['n_lines']:<3} {'yes' if s.joined_balanced else 'no':>5} {'yes' if s.joined_parses else 'no':>6}")
            out_rows.append({"cmpd_id": c["cmpd_id"], "page": page, "arm": ARM, "reference_status": "human_verified" if verified else "agent_transcription", "edit": edit, "cer": round(edit / len(ref), 5), "ocr_len": len(txt), "ref_len": len(ref), "n_rows": len(picked), "expected_rows": c["n_lines"], "balanced": s.joined_balanced, "parses": s.joined_parses, "text": txt})

    lines.append("")
    lines.append("Only 1036 carries a human-verified reference; the other")
    lines.append("rows are agent transcriptions and are indicative only.")
    lines.append("Text level. No structure is verified.")

    if out_rows:
        with (OUT / "page_route_ref.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
    io.open(OUT / "page_route_ref.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
