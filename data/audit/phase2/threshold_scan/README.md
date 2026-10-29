# Phase 2 - binarisation threshold sweep + dilation test

PRELIMINARY. Every number here is scored against **one** reference string
(cmpd 1036, supplied by the researcher). This measures how stable the
OCR configuration is on that single string. It is **not** a dataset
accuracy rate and must not be reported as one.

## Commands

```powershell
.venv\Scripts\python.exe phase2_threshold_scan.py   # threshold sweep
.venv\Scripts\python.exe phase2_dilate_test.py      # dilation, run after
```

Both write only inside this directory.

## Fixed across the whole sweep

| parameter | value |
|---|---|
| crop | `../cells_text/I100297_1036_text.png` (Task-2 extract) |
| upscale | 2x, `INTER_CUBIC` |
| strip borders | 0 px |
| recogniser | RapidOCR PP-OCRv4 mobile, `en` det + `en` rec |
| text_score | 0.0 |
| use_cls | False |

Only the OpenCV threshold varied: 150 / 180 / 200 / 220 / 240.

## Result

| threshold | len | edits | CER | RDKit parse |
|---|---|---|---|---|
| 150 | 267 | 12 | 0.04428 | no |
| 180 | 272 | 24 | 0.088561 | no |
| **200** | **270** | **2** | **0.00738** | **yes** |
| 220 | 269 | 3 | 0.01107 | no |
| 240 | 260 | 458 | 1.690037 | no |

Threshold 200 is the sweep winner and is also the earlier
`binarize_up2_en` baseline, so the sweep improved nothing: edit distance
2 -> 2. The optimum is sharp and non-monotonic; 240 destroys the glyphs
(strokes thinner than the threshold) and 180/150 over-thicken them into
merged blobs.

## Dilation (separate experiment)

| kernel | len | edits | CER | RDKit parse |
|---|---|---|---|---|
| 1x1 (none) | 270 | 2 | 0.00738 | yes |
| 2x2 | 269 | 4 | 0.01476 | no |
| 3x1 | 268 | 11 | 0.04059 | no |
| 3x2 | 269 | 19 | 0.070111 | no |

Dilation does not help. Thickening strokes merges adjacent characters
and costs more accuracy than it recovers.

## Residual errors and where they come from

Both survivors sit at a **detector line-box right edge**, not in the
glyph shape:

1. `@77` `C(=O)` -> `C(O)`: the `=` is the last glyph on printed line 2.
   The box returned for that line ends at x=1667/2 ~ 833 (2x image) while
   the ink continues to x=841 in the source crop.
2. `@115` `C(F)=C5` -> `C(F)-C5`: same line-3 right-edge truncation. The
   two `=` earlier on that same line are recognised correctly.

`error_zoom/line02_x35-1673.png` shows this directly: of the three `=`
on that line, only the final one is damaged, and it is visibly narrower.

So the remaining bottleneck is **line-box truncation at the right margin**,
not the equals sign. Padding the crop's right side before detection, or
shrinking the detected box before recognition, is the next thing worth
testing.

## Files

- `threshold_scan.csv` - one row per (cmpd, threshold)
- `threshold_scan_report.txt` - tables + residual diffs
- `dilate/dilate_scan.csv`, `dilate/dilate_report.txt` - dilation run
- `raw_ocr/<cmpd>_th<N>.txt` - raw recognised string per threshold
- `raw_ocr/<cmpd>_th<N>_diff.json` - character-level opcodes
- `error_zoom/line*.png` - 3x nearest-neighbour crops of each detected line
- `run_threshold_scan.log`, `run_dilate.log` - stdout/stderr

## Caveat

Only cmpd 1036 has a reference that passed transcription QC. 1037 is
withheld until a human re-reads it; 1038/1040/2173 failed transcription
QC (mass-gate deltas of -14.2 / +41.9 / +13.9 Da indicate whole
residues were misread) and are excluded. Do not add 1037 back until it is
confirmed.
