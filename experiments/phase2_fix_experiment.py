"""Independent test: fix line 2's LEFT clipping and re-measure.

Line 2's detector box starts at x=41 while its ink starts at x=11, so 30px
of printed glyphs - including the leading '=' - are clipped away. The
candidate experiment shows the posterior cannot recover a glyph that is
not in its input (insert '=' costs 12.4 nats). So extend the box left and
check whether the character comes back.
"""

from __future__ import annotations

from pathlib import Path

import csv
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_box_expansion import box_rect, line_boxes  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    PHASE2,
    build_engine,
    load_meta,
    load_rows,
    preprocess,
)
from phase2_recog_candidates import (  # noqa: E402
    OUT,
    ctc_logprob_exact,
)

OUT.mkdir(parents=True, exist_ok=True)
meta = load_meta()
rows = load_rows()
cid = "1036"
ref = REFERENCES[cid]["smiles"]
mz = float(rows[cid]["ms_mz"])
pol = (rows[cid].get("ms_polarity") or "").strip()

img = cv2.imread(str(PHASE2 / meta[cid]["crop_path"]))
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()
full = engine(proc)
polys = line_boxes(full)

results = []
for pad in (0, 8, 16, 24, 30, 36, 48, 64):
    lines = []
    for i, poly in enumerate(polys):
        p = np.array(poly, dtype=np.float32).copy()
        if i == 2 and pad:
            nl = max(0.0, float(p[0][0]) - pad)
            p[0][0] = nl
            p[3][0] = nl
        lines.append(get_rotate_crop_image(proc, p))
    texts = list(engine.recognize_txt(lines).txts)
    joined = "".join(t for t in texts if t)
    l2 = texts[2]
    edits = edit_ops(ref, joined)
    got_head = l2[:12]
    recovered = l2.startswith("=O)N")
    results.append(
        {
            "pad_left_px": pad,
            "line2_head": got_head,
            "leading_equals_recovered": recovered,
            "joined_len": len(joined),
            "edit_distance": edits,
            "cer": round(edits / len(ref), 6),
            "has_minus_error": "C(F)-C" in joined,
        }
    )
    print(
        f"pad={pad:3d} line2={l2[:12]!r:16s} lead'='={str(recovered):5s} "
        f"len={len(joined):4d} edits={edits:3d} "
        f"minus_err={'C(F)-C' in joined}"
    )

with (OUT / f"{cid}_left_extension.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0]))
    w.writeheader()
    w.writerows(results)
print(f"\nwrote {OUT/f'{cid}_left_extension.csv'}")

best = min(results, key=lambda r: r["edit_distance"])
print(
    f"best: pad={best['pad_left_px']} edits={best['edit_distance']} "
    f"(baseline edits=2)"
)
if best["edit_distance"] < 2:
    print("IMPROVEMENT over the existing baseline")
else:
    print("NO improvement - baseline retained")
