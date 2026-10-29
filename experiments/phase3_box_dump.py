r"""Dump every detector box, with its own text, for the cell path and the
page path.

The row-by-row comparison shows the cell path dropping opening
parentheses from the MIDDLE of a line, not only from its start. A
mid-line character cannot be explained by the box being too narrow on the
left, so the boxes themselves have to be inspected.

This script writes, for both paths:

    boxes.txt   every box: coordinates, recognised text, score
    vis_page.png / vis_cell.png  the page or crop with every box drawn

Read-only.

Run:
    .venv\Scripts\python.exe phase3_box_dump.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402
from phase3_page_extract import (  # noqa: E402
    GOLD, PREP, baseline_engine, load_cells,
)

OUT = REPO / "data" / "audit" / "phase3" / "box_dump"
PAGE = "I100297"


def _dump(engine, img, tag: str, scale: float) -> list[str]:
    result = engine(img)
    boxes = np.asarray(result.boxes, dtype=float)
    txts = list(result.txts)
    scores = list(result.scores)
    out = [f"--- {tag}: {len(boxes)} boxes, image {img.shape} ---"]
    vis = img.copy()
    for i, poly in enumerate(boxes):
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        t = txts[i] if i < len(txts) else ""
        s = scores[i] if i < len(scores) else 0.0
        xs = pts[:, 0] / scale
        ys = pts[:, 1] / scale
        out.append(f"  box {i:3d} x[{xs.min():7.1f},{xs.max():7.1f}] "
                   f"y[{ys.min():7.1f},{ys.max():7.1f}] "
                   f"score={s:.3f} {t!r}")
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
        cv2.putText(vis, str(i), (int(xs.min()), int(ys.min()) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
    cv2.imwrite(str(OUT / f"vis_{tag}.png"), vis)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()[PAGE]
    page = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{PAGE}.jpg"))
    engine = baseline_engine()
    ex = RapidOcrTableExtractor(rec_top_k=0)

    lines: list[str] = []
    lines += _dump(engine, PREP["up2_th200"](page), "page", 2.0)
    lines.append("")

    meta = next(c for c in cells if c.cmpd_id == "1036")
    cell = page[int(meta.y1):int(meta.y2), 1088:1919]
    trimmed = ex._trim_to_content(cell)
    lines += _dump(engine, ex._upscale(trimmed), "cell", 2.0)
    lines.append("")
    lines.append("note: cell boxes are in 2x crop coordinates, so divide by 2 "
                 "to compare with the page boxes")
    lines.append(f"crop origin on the page: x=1088 y={int(meta.y1)}")

    (OUT / "boxes.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
