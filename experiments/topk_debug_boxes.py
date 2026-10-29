r"""Dump every page-level detector box with geometry and text.

Used to work out why the SMILES-column filter in topk_page_level.py keeps
so few rows. Diagnostic only.

Run:
    .venv\Scripts\python.exe topk_debug_boxes.py I100297
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_threshold_scan import BASELINE_THRESHOLD, build_engine  # noqa: E402
from row_crop_audit import line_boxes, preprocess  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
OUT = REPO / "data" / "audit" / "phase2" / "page_level_topk"


def main() -> None:
    page = sys.argv[1] if len(sys.argv) > 1 else "I100297"
    rel = f"PCTCN2025075389-ftappb-{page}.jpg"
    engine = build_engine()
    img = cv2.imread(str(GOLD / rel))
    proc = preprocess(img, BASELINE_THRESHOLD)
    boxes = line_boxes(engine(proc))

    out = io.open(OUT / f"boxes_{page}.txt", "w", encoding="utf-8")
    out.write(f"page {page}  proc shape {proc.shape}\n")
    out.write(f"{'i':>3} {'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5} "
              f"{'cx':>5} {'cy':>5} {'w':>5} {'h':>4}  text\n")
    for i, poly in enumerate(boxes):
        p = np.asarray(poly, dtype=np.float32).reshape(4, 2)
        x1, y1 = p.min(axis=0)
        x2, y2 = p.max(axis=0)
        from rapidocr.utils.process_img import get_rotate_crop_image
        crop = get_rotate_crop_image(proc, p)
        res = engine.recognize_txt([crop])
        r = res[0] if isinstance(res, list) else res
        t = (getattr(r, "txts", None) or ("",))[0]
        out.write(f"{i:>3} {x1:>5.0f} {y1:>5.0f} {x2:>5.0f} {y2:>5.0f} "
                  f"{(x1 + x2) / 2:>5.0f} {(y1 + y2) / 2:>5.0f} "
                  f"{x2 - x1:>5.0f} {y2 - y1:>4.0f}  {t!r}\n")
    out.write("\nmanifest cells:\n")
    man = (REPO / "data/audit/phase2/row_crop_audit/manifest.csv")
    for r in csv.DictReader(man.open(encoding="utf-8-sig")):
        if r["page"] == page and r["is_header"] == "no":
            out.write(f"  {r['cmpd_id']}: x[{r['bbox_x1']}..{r['bbox_x2']}] "
                      f"y[{r['bbox_y1']}..{r['bbox_y2']}] "
                      f"n_lines={r['n_lines']}\n")
    out.close()
    print(f"wrote {OUT / ('boxes_' + page + '.txt')}")


if __name__ == "__main__":
    main()
