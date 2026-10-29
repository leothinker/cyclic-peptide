"""Why does rec-only differ from the pipeline's det+rec output?"""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_box_expansion import (  # noqa: E402
    box_rect, line_boxes, rec_only,
)
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD, build_engine, preprocess,
)

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()

full = engine(proc)
pipeline = list(full.txts)
polys = line_boxes(full)

print(f"pipeline texts: {len(pipeline)}")
crops = [get_rotate_crop_image(proc, np.array(p, dtype=np.float32))
         for p in polys]
print("rectified crop shapes:", [c.shape for c in crops])

mine = [rec_only(engine, c) for c in crops]
for n, (a, b) in enumerate(zip(pipeline, mine)):
    same = a == b
    print(f"line {n}: pipeline={len(a):4d} reconly={len(b):4d} same={same}")
    if not same:
        print(f"    pipeline: {a!r}")
        print(f"    reconly : {b!r}")

rects = [box_rect(p) for p in polys]
print()
print("detector box rects (processed coords):")
for n, r in enumerate(rects):
    print(f"  line {n}: x={r[0]}-{r[2]}  y={r[1]}-{r[3]}  crop_w={crops[n].shape[1]}")

print()
print("--- batched recognize_txt([all 7 crops]) vs pipeline ---")
batch = engine.recognize_txt(crops)
for n, (a, b) in enumerate(zip(pipeline, batch.txts)):
    print(f"line {n}: pipeline={len(a):4d} batch={len(b):4d} same={a == b}")
print("joined equal:", "".join(pipeline) == "".join(batch.txts))

print()
print("--- single-image recognize_txt([crop]) per line ---")
for n, c in enumerate(crops):
    one = engine.recognize_txt([c])
    t = one.txts[0] if isinstance(one.txts, (list, tuple)) else one.txts
    print(f"line {n}: pipeline={len(pipeline[n]):4d} single={len(t):4d} "
          f"same={pipeline[n] == t}")
