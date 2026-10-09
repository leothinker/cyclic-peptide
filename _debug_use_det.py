"""Minimal test: does use_det=False work on a single image?
If yes, why did test_rapidocr_v2.py return empty?"""
import time
import numpy as np
import cv2
from rapidocr import RapidOCR
from rapidocr.utils.typings import OCRVersion, ModelType, EngineType, TaskType, LangCls

img = cv2.imread('testocr.jpg')
h0, w0 = img.shape[:2]
img_up = img.copy()  # no upscale
print(f'img shape: {img.shape}  upscaled: {img_up.shape}')

eng = RapidOCR(params={
    'Det.lang_type': 'en',
    'Det.ocr_version': OCRVersion.PPOCRV4,
    'Det.model_type': ModelType.MOBILE,
    'Det.engine_type': EngineType.ONNXRUNTIME,
    'Det.task_type': TaskType.DET,
    'Rec.lang_type': 'en',
    'Rec.ocr_version': OCRVersion.PPOCRV4,
    'Rec.model_type': ModelType.MOBILE,
    'Rec.engine_type': EngineType.ONNXRUNTIME,
    'Rec.task_type': TaskType.REC,
    'Cls.lang_type': LangCls.CH.value,
    'Global.text_score': 0.0,
    'Global.return_word_box': True,
})

print()
print('--- run 1: use_det=False, use_cls=False (no detection) ---')
t0 = time.perf_counter()
r = eng(img_up, use_det=False, use_cls=False)
print(f'time: {time.perf_counter() - t0:.3f}s')
print(f'type: {type(r).__name__}')
print(f'txts: {r.txts!r}')
print(f'scores: {r.scores!r}')
print(f'len(scores): {len(r.scores) if r.scores else 0}')

print()
print('--- run 2: use_det=True, use_cls=False (default detection) ---')
t0 = time.perf_counter()
r2 = eng(img_up)  # all defaults
print(f'time: {time.perf_counter() - t0:.3f}s')
print(f'type: {type(r2).__name__}')
print(f'len(txts): {len(r2.txts) if r2.txts else 0}')
for i, t in enumerate(r2.txts[:5] if r2.txts else []):
    print(f'  [{i}]  {t!r}')

print()
print('--- run 3: use_det=False, default cls (cls ON) ---')
t0 = time.perf_counter()
r3 = eng(img_up, use_det=False, use_cls=True)
print(f'time: {time.perf_counter() - t0:.3f}s')
print(f'type: {type(r3).__name__}')
print(f'len(txts): {len(r3.txts) if r3.txts else 0}')
for i, t in enumerate(r3.txts[:5] if r3.txts else []):
    print(f'  [{i}]  {t!r}')
