"""Exhaustive search: is '=' anywhere in the CTC posterior, per line?"""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_box_expansion import line_boxes  # noqa: E402
from phase2_threshold_scan import build_engine, preprocess  # noqa: E402

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(img, 200)
engine = build_engine()
polys = line_boxes(engine(proc))
rec = engine.text_rec
chars = rec.postprocess_op.character
eq_i, dash_i = chars.index("="), chars.index("-")
imgH, imgW = rec.rec_image_shape[1], rec.rec_image_shape[2]
max_wh_ratio = imgW / imgH

full = engine(proc)
texts = list(full.txts)
print(f"{'line':>4s} {'chars':>5s} {'steps':>5s} {'max_p(=)':>9s} "
      f"{'steps p(=)>=.05':>16s}  text tail")
for li, p in enumerate(polys):
    line = get_rotate_crop_image(proc, np.array(p, dtype=np.float32))
    norm = rec.resize_norm_img(line, max_wh_ratio)[np.newaxis, :].astype(np.float32)
    preds = rec.session(norm)[0]
    pe = preds[:, eq_i]
    strong = int((pe >= 0.05).sum())
    print(
        f"{li:4d} {len(texts[li]):5d} {preds.shape[0]:5d} {pe.max():9.4f} "
        f"{strong:16d}  ...{texts[li][-14:]!r}"
    )
    top = np.argsort(-pe)[:3]
    for t in sorted(top):
        arg = chars[int(preds[t].argmax())]
        print(
            f"      t={t:3d} p(=)={pe[t]:.4f} argmax={arg!r} "
            f"p(-)={preds[t][dash_i]:.4f}"
        )

print()
print("=== last 3 timesteps of line 1 and first 3 of line 2 ===")
for li in (1, 2):
    line = get_rotate_crop_image(proc, np.array(polys[li], dtype=np.float32))
    norm = rec.resize_norm_img(line, max_wh_ratio)[np.newaxis, :].astype(np.float32)
    preds = rec.session(norm)[0]
    ts = range(preds.shape[0] - 3, preds.shape[0]) if li == 1 else range(3)
    for t in ts:
        p = preds[t]
        order = np.argsort(-p)[:5]
        top = [(chars[i], round(float(p[i]), 4)) for i in order]
        print(f"  line{li} t={t:3d} argmax={chars[int(p.argmax())]!r} {top}")
