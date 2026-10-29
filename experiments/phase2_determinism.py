"""Is the RapidOCR recognise path deterministic for the same input?"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_threshold_scan import build_engine, preprocess, recognise  # noqa: E402

crop = Path("data/audit/phase2/cells_text/I100297_1036_text.png")
img = cv2.imread(str(crop))
proc = preprocess(img, 200)

engine = build_engine()
outs = [recognise(engine, proc) for _ in range(4)]
for n, o in enumerate(outs):
    print(f"run {n}: len={len(o)}")
print("all identical:", len(set(outs)) == 1)
for n in range(1, len(outs)):
    if outs[n] != outs[0]:
        print(f"  run0 vs run{n} differs at:",
              [(i, a, b) for i, (a, b) in enumerate(zip(outs[0], outs[n]))
               if a != b][:10])

engine2 = build_engine()
outs2 = [recognise(engine2, proc) for _ in range(2)]
print("fresh engine matches first engine:", outs2[0] == outs[0])
