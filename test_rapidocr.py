"""
test_rapidocr.py — single-image test for RapidOCR with EN model + per-character confidences + top-K.

USAGE:
      python test_rapidocr.py                       # default image:  data/processed/cells/PCTCN2025075389-ftappb-I100280_1001_struct.png
      python test_rapidocr.py path/to/img.jpg       # any image path

PREREQUISITES (first run will auto-download ~30 MB into .venv/Lib/site-packages/rapidocr/models/):
      det: en_PP-OCRv3_det_mobile.onnx
      rec: en_PP-OCRv4_rec_mobile.onnx
      dict: en_dict.txt (96 ASCII chars + specials)

OUTPUT (printed to stdout):
   - N recognised text regions: text + line-level score
   - per-character confidence from word_results
   - top-5 character alternatives per non-blank frame from the raw softmax
   - ambiguity statistics (how often is there a useful second-best candidate?)

HOW TO SWITCH MODELS:
   * For Chinese (default):            engine = RapidOCR()
   * For English:                      engine = RapidOCR(params={'Det.lang_type': 'en',
                                                                'Rec.lang_type': 'en'})
   * For EN + PP-OCRv5 (newer, less battle-tested):
                                       from rapidocr.utils.typings import OCRVersion
                                       params['Rec.ocr_version'] = OCRVersion.PPOCRV5
   * For digits-only (number recipes):                 engine = RapidOCR(params={'Rec.lang_type': 'en'})  # same EN model
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import (
    EngineType,
    LangCls,
    ModelType,
    OCRVersion,
    TaskType,
)

# ===========================================================================
# 1. ENGINE INITIALISATION  (English model)
# ===========================================================================
#
# RapidOCR takes a dict of dotted-path params, e.g. 'Det.lang_type'.
# lang_type accepts a plain string ('en'); but engine_type / model_type /
# ocr_version / task_type REQUIRE the actual Enum (the parser validates).
#
# Available en models:
#   en_PP-OCRv4_rec_mobile  (default, well-tested, 96-char ASCII dict)
#   en_PP-OCRv5_rec_mobile  (newer, same dict)
#
engine = RapidOCR(params={
    # ---- Detection (English text detector) ----
    'Det.lang_type': 'en',
    'Det.ocr_version': OCRVersion.PPOCRV4,
    'Det.model_type': ModelType.MOBILE,
    'Det.engine_type': EngineType.ONNXRUNTIME,
    'Det.task_type': TaskType.DET,
    # ---- Recognition (English text recogniser) ----
    'Rec.lang_type': 'en',
    'Rec.ocr_version': OCRVersion.PPOCRV4,           # swap to PPOCRV5 for newer EN
    'Rec.model_type': ModelType.MOBILE,
    'Rec.engine_type': EngineType.ONNXRUNTIME,
    'Rec.task_type': TaskType.REC,
    # ---- Classifier (Chinese-only, kept for orientation) ----
    'Cls.lang_type': LangCls.CH.value,
    # ---- Engine-wide ----
    'Global.text_score': 0.0,                        # don't drop low-confidence text
    'Global.return_word_box': True,                  # enables per-character confidence
    'Global.use_cls': True,                          # optional: 0/180-degree rotation
})

# ===========================================================================
# 2. HOOK INTO THE SOFTMAX  (top-K lives here, before the default decoder)
# ===========================================================================
# Default flow inside rapidocr.ch_ppocr_rec.main.TextRecognizer.__call__:
#
#     preds = self.session(norm_img_batch)            # shape (B, T, num_classes)
#     line_results, word_results = self.postprocess_op(preds, ...)
#
# Default postprocess_op = CTCLabelDecode.__call__:
#     preds_idx = preds.argmax(axis=2)                # <-- keeps ONLY argmax char
#     preds_prob = preds.max(axis=2)                  # <-- keeps ONLY its prob
#     ... collapse repeats/blanks ...
#
# We replace __call__ with a wrapper that ALSO captures `preds` so we can pull
# top-K from the raw softmax BEFORE the default decoder discards it.
import rapidocr.ch_ppocr_rec.utils as rec_utils
_RAW: dict = {}
_orig_ctc = rec_utils.CTCLabelDecode.__call__


def _capture_call(self, preds, return_word_box=False, **kwargs):
    _RAW['preds'] = preds.copy()                       # (B, T, num_classes)
    return _orig_ctc(self, preds, return_word_box, **kwargs)


rec_utils.CTCLabelDecode.__call__ = _capture_call

# ===========================================================================
# 3. RUN OCR
# ===========================================================================
DEFAULT_IMG = 'data/processed/cells/PCTCN2025075389-ftappb-I100280_1001_struct.png'
img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
print(f'OCR input: {img_path}')
img = cv2.imread(img_path)
result = engine(img)

# ===========================================================================
# 4. WHAT YOU CAN READ FROM result (the easy stuff)
# ===========================================================================
# result.txt / result.txt            tuple[str, ...]  recognised text per region
# result.scores                      tuple[float, ...] line-level MEAN of per-char confidences
# result.boxes                       np.ndarray       4-point corner coordinates of each region
# result.word_results[i]             WordInfo         per-word info (incl. per-char confidence)
#   ├─ .words                         list[list[str]] grouped chars per word
#   ├─ .word_cols                     list[list[int]]  model-frame column index per char
#   ├─ .word_types                    list[WordType]  CN/EN_NUM per word
#   └─ .confs                         list[float]      per-character confidence (max prob)
# result.vis(path)                   np.ndarray       save annotated image
#
print(f'\n=== recognised {len(result.txts)} text regions (line-level score) ===')
for i, (txt, sc) in enumerate(zip(result.txts, result.scores)):
    flag = ''
    if sc < 0.7:
        flag = '  <-- low confidence'
    if '@' in txt or '=' in txt:
        flag += '   (SMILES-like)'
    print(f'  [{i:>2}]  conf={sc:.4f}   {txt[:80]!r}{"..." if len(txt)>80 else ""}{flag}')

# ===========================================================================
# 5. PER-CHARACTER CONFIDENCE  (from word_results)
# ===========================================================================
print('\n=== per-character confidence (word_results[i].confs) ===')
for i, wr in enumerate(result.word_results[:5]):
    if wr is None or not wr.confs:
        continue
    chars = ''.join(c for w in wr.words for c in w)
    print(f'  [{i}]  text={chars[:60]!r}{"..." if len(chars)>60 else ""}')
    print(f'        confs  = {[round(c, 3) for c in wr.confs[:25]]}'
          f'{"..." if len(wr.confs) > 25 else ""}')

# ===========================================================================
# 6. TOP-K PER FRAME  (from the raw softmax that the default decoder discards)
# ===========================================================================
preds = _RAW.get('preds')
chars_dict = engine.text_rec.postprocess_op.character   # list[str], index = class id
print(f'\n=== raw softmax shape: {preds.shape}  '
      f'(batch, T, num_classes={preds.shape[2]}) ===')

TOPK = 5
# argsort once over the last axis (descending)
order = np.argsort(-preds[0], axis=1)[:, :TOPK]              # (T, TOPK)
probs = np.take_along_axis(preds[0], order, axis=1)          # (T, TOPK)

print('\n=== top-5 alternatives per non-blank decoded frame ===')
print(f'  {"frame":>5}  {"decoded":>5}  {"top-5 candidates":<55}')
prev_idx = None
emit = 0
decoded_text = result.txts[0] if result.txts else ''
for t in range(preds.shape[1]):
    a_idx = int(order[t, 0])
    a_ch = chars_dict[a_idx]
    # match CTC collapse: skip blanks and consecutive duplicates
    if a_ch == 'blank' or a_idx == prev_idx:
        prev_idx = a_idx
        continue
    top = [(chars_dict[int(order[t, k])], float(probs[t, k]))
           for k in range(TOPK)]
    line = '  '.join(f'{c!r}:{p:.2f}' for c, p in top)
    decoded_ch = decoded_text[emit] if emit < len(decoded_text) else '·'
    print(f'  {t:>5}  {decoded_ch!r:>5}  {line}')
    emit += 1
    prev_idx = a_idx

# ===========================================================================
# 7. AMBIGUITY SUMMARY  (where top-K would actually help)
# ===========================================================================
print('\n=== ambiguity statistics over non-blank frames ===')
n = n_low = n_top2 = n_top2_strong = 0
for t in range(preds.shape[1]):
    if chars_dict[int(order[t, 0])] == 'blank':
        continue
    n += 1
    p1 = float(probs[t, 0])
    p2 = float(probs[t, 1])
    if p1 < 0.8:
        n_low += 1
    if p2 > 0.05:
        n_top2 += 1
    if p2 > 0.3 * p1:
        n_top2_strong += 1
print(f'  frames with low argmax (p1 < 0.80)         : {n_low}/{n}')
print(f'  frames with non-trivial 2nd  (p2 > 0.05)   : {n_top2}/{n}')
print(f'  frames with SIMILAR 2nd       (p2 > 0.3*p1): {n_top2_strong}/{n}')

# ===========================================================================
# 8. SAVE VISUALISATION
# ===========================================================================
out_vis = Path(img_path).with_name(Path(img_path).stem + '_vis.jpg')
result.vis(str(out_vis))
print(f'\nvis image:  {out_vis}')
