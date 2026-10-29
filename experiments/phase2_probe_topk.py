"""Probe what the installed recogniser actually exposes."""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_threshold_scan import build_engine, preprocess  # noqa: E402
from phase2_box_expansion import box_rect, line_boxes  # noqa: E402

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(img, 200)
engine = build_engine()

full = engine(proc)
polys = line_boxes(full)
print("detector lines:", len(polys))

# --- 1. are preds probabilities? run the raw session ourselves ---
rec = engine.text_rec
print("TextRecognizer:", type(rec).__name__)
print("cfg Rec keys:", list(rec.cfg.keys()))
print("rec_img_shape:", rec.rec_image_shape)
print("rec_batch_num:", rec.rec_batch_num)
print("character dict size:", len(rec.postprocess_op.character))

imgH, imgW = rec.rec_image_shape[1], rec.rec_image_shape[2]
max_wh_ratio = imgW / imgH
norm = rec.resize_norm_img(proc, max_wh_ratio)
preds = rec.session(norm[np.newaxis, :].astype(np.float32))
print("preds shape:", preds.shape)
print("preds min/max:", float(preds.min()), float(preds.max()))
sums = preds[0].sum(axis=1)
print("per-timestep class-sum: min %.6f max %.6f" % (sums.min(), sums.max()))

# --- 2. does the patched top_k path return alternatives? ---
engine.return_top_k = 5
engine.cfg.Rec["top_k"] = 5
lines = [get_rotate_crop_image(proc, np.array(p, dtype=np.float32))
         for p in polys]
out = engine.recognize_txt(lines)
print()
print("top_k_lines present:", out.top_k_lines is not None)
if out.top_k_lines:
    tk = out.top_k_lines[0]
    print("line0 chars:", len(tk), "first 8 chars with alternatives:")
    for i, alts in enumerate(tk[:8]):
        top = [(c, round(p, 4)) for c, p in alts[:4]]
        print(f"  {i}: {top}")

# --- 3. does the char at the two error positions have '=' as alternative? ---
joined = "".join(out.txts)
all_tk = [t for line in out.top_k_lines if line for t in line]
print()
print("joined len:", len(joined), "total chars with alternatives:", len(all_tk))
print()
print("--- all timesteps whose top-1 is '-' or '=' ---")
for i, alts in enumerate(all_tk):
    if not alts:
        continue
    top = alts[0][0]
    if top in "-=":
        near = joined[max(0, i - 10):i + 11]
        print(f"  idx {i:3d} top1={top!r} alts="
              f"{[(c, round(p, 4)) for c, p in alts[:4]]}")
        print(f"          context {near!r}")
