# Phase 5 — the printed-SMILES route for a second patent

Status: **the recognition repair is implemented and measured**.  The
formal CSV was not touched, no existing route changed behaviour, and
nothing was committed to Git.

## Short answer

**Yes, the recognition can be repaired, and it is now repaired.**  On
WO2022251372 the printed SMILES went from **1 of 28 RDKit-parseable**
to **17 of 28 as recognised and 19 of 28 after the repair chain**, and
the character error rate against the hand transcriptions fell from
2.6 % to 0.4 %.

The fix is not a better model.  It is **6 px of white margin around
each printed line**, because the crop ended on the last ink pixel and
the recogniser could not read a glyph that touches the border.

The same margin **must not** be applied to the old patent
WO2025162428, whose rows are packed tightly enough that the margin
reaches into the neighbouring lines.  The fix is therefore a **third
route**, not a change to the two that exist.

## 1. Where the recognition actually failed

The ink-band segmentation was already correct: all 60 printed lines of
the six transcribed compounds were found with the right boundaries.
The damage came afterwards, from the recogniser, and it concentrated
on the **last glyph of every line**.

`text_span()` returns the first and the last column that contains any
ink, so the cropped band ends exactly on the final ink pixel.  That
glyph is flush against the image border with no margin at all:

```
0 px margin ->  'H](Cc2ccc(Cl)cc2)NC(=O)'    the '[' is lost
6 px margin ->  'H](Cc2ccc(Cl)cc2)NC(=O)['   exact
```

The pixels never change -- only the margin does.

## 2. The measured margin curve

60 hand-transcribed lines, English PP-OCRv4 recogniser, no upscale:

| margin | 0 px | 2 px | 4 px | **6 px** | 8 px | 10 px | 12 px | 16 px | 20 px | 24 px |
| - | - | - | - | - | - | - | - | - | - | - |
| line edits | 31 | 26 | 13 | **5** | 8 | 8 | 9 | 11 | 9 | 21 |

The optimum is sharp, so 6 px is a measured candidate, not a settled
constant.  Between 6 and 20 px the result is not critical; below 4 px
it clearly is.

## 3. The two patents need opposite settings

This is the most important constraint in this phase.  The same margin
measured on the *old* patent makes it much worse:

| patent | route | total line edits (5 references) |
| - | - | - |
| WO2025162428 | `validated` (no margin) | **15** |
| WO2025162428 | `padded` (6 px margin) | 95 |
| WO2022251372 | `validated` (no margin) | 31 |
| WO2022251372 | `padded` (6 px margin) | **5** |

The reason is measured, not guessed: on WO2025162428 the ink band of
compound 1036 spans y = 96-129 and already contains fragments of the
lines above and below, because the rows sit close together
(`PAD_TOP` = 14, `PAD_BOTTOM` = 15 reach into the neighbours).  On
WO2022251372 the rows are far enough apart that the band stays clean.
The evidence is in `tools/old_1036_band2.png` and
`tools/old_1036_band2_pad6.png`.

**So the margin is a separate route, and the default stays as it is.**

## 4. What was changed in the code

Three changes, all in `src/cpd/`, all guarded.

### 4.1 A third OCR route (`src/cpd/ocr/printed_smiles.py`)

`ROUTE_PADDED = "padded"` joins `production` and `validated`.  It
shares the ink-band geometry and the crop of `validated`, and differs
in two measured ways:

| | `validated` | `padded` |
| - | - | - |
| recogniser | upstream PP-OCRv6 | English PP-OCRv4 mobile |
| white margin | none | 6 px left and right |
| upscale | 2x | none |

The route recognises at 1x because that was measured: over the 28
Table 1a compounds, 1x gives 21/28 bracket balanced and 17/28
parseable, while 2x gives 20/28 and 15/28.  The character accuracy
against the hand transcriptions is identical either way, so the
difference is in bracket balance rather than in character accuracy.

### 4.2 A false positive removed (`src/cpd/ocr_rewrite.py`)

`rewrite_oxo` rewrote every `X=O)` into `X(=O)`.  The lactam of a
cyclic peptide is printed as `C1=O)`, where the ring-closure digit
sits between the carbon and the `=O`, so the rule inserted a spurious
bracket into a string that was never damaged.

The guard skips the rewrite when the character before the `=` is a
digit.

### 4.3 A misread oxygen repaired (`src/cpd/chem.py`)

`auto_repair_smiles` now reads a lone `=0` as `=O`.  Zero is not an
element, so `=0` is never valid SMILES; and a ring-closure digit
always appears in a matched pair, so a string with exactly one `0` has
a misread oxygen rather than a ring closure.  The guard is the count:
two zeros means two ring-closure digits and nothing is changed.

## 5. The two repair-chain guards help the old patent too

Measured on the real OCR text of both patents
(`repair_chain_regression/`), with the shipped chain against the
previous one:

| patent / route | parses before | parses after | line edits before | after |
| - | - | - | - | - |
| WO2025162428 / `validated` | 169 / 336 | **183 / 336** | 15 | **13** |
| WO2022251372 / `validated` | 0 / 28 | **1 / 28** | 29 | **24** |
| WO2022251372 / `production` | 3 / 28 | 3 / 28 | 19 | **15** |

The attribution, so it is clear which guard does what
(`padded_scale_ab/` and the guard split in the same run):

| variant | WO2025162428 (`validated`) | WO2022251372 (`validated`) |
| - | - | - |
| shipped, no guards | 169 parses | 0 parses |
| `rewrite_oxo` guard only | 172 parses | 0 parses |
| `=0` -> `=O` only | 180 parses | 1 parse |
| **both guards** | **183 parses** | **1 parse** |

Neither guard makes anything worse on either patent.

## 6. The result on WO2022251372

All 28 Table 1a compounds, with the shipped code:

| route | bracket balanced | RDKit parses | RDKit parses after repair |
| - | - | - | - |
| `production` (detector) | 2 / 28 | 1 / 28 | 3 / 28 |
| `validated` (PP-OCRv6) | 3 / 28 | 0 / 28 | 1 / 28 |
| **`padded`** (PP-OCRv4 + 6 px) | **21 / 28** | **17 / 28** | **19 / 28** |

Against the six hand transcriptions (cmpd 1, 2, 3, 4, 10, 28), the
`padded` route has a total edit distance of **5** characters, or
0.4 % CER.  The same six compounds have 31 characters of error on the
`validated` route.

Per line, **56 of the 60 printed lines are character-exact**.  The
four that are not each lose exactly one character:

| compound, line | printed | read |
| - | - | - |
| cmpd 2, line 9 | `@@H](C)OC(=O)c1ccccc1` | `@@H](C)OC(=O)c1ccccc` |
| cmpd 10, line 7 | `@@H](C)N(C)C(=O)[C@H](C` | `@H](C)N(C)C(=O)[C@H]C` |
| cmpd 28, line 3 | `(C)N(C)C(=O)[C@@H](CCO` | `(C)N(C)C(=O)[C@H](CCO` |
| cmpd 28, line 5 | `(=O)[C@H](Cc2ccccc2)NC(C` | `(=O)[C@H](Cc2ccccc2)N(C` |

The remaining 9 compounds that do not parse fail on a lost bracket
or a lost trailing ring-closure digit, not on the margin.

## 7. What was *not* changed, and why that matters

The two existing routes were re-run over all **364 printed cells of
both patents** before and after the change
(`routes_before.json` / `routes_after.json`):

* same cell set, **0 differences**, and the two files are
  **byte-identical**.

So the published behaviour of WO2025162428 is unchanged: the `validated`
route still gives 183 / 336 parseable, 13 characters of total error
on the five references, and the `production` route is untouched.

The formal CSV `data/processed/compounds_combined.csv` was not
modified.  Its SHA256 is still
`99F746AF20E4904EE6583C70461506783D2573AE01D7A69D127AD5F3AB927D0D`.
Re-running `auto_repair_smiles` over all 406 records and all ten
SMILES columns changes **0 strings**.

The test suite is at **183 passed** (the baseline was 170; the 13 new
tests cover the two guards and the new route).  The one error is the
known sandbox issue: pytest cannot write to
`C:\Users\Admin\AppData\Local\Temp\pytest-of-Admin`.

## 8. Still open

1. **The route is not wired into the patent pipeline.**  The
   end-to-end adapter for WO2022251372 is a separate task: the pages
   are TIFF (the scanner only globs `*.jpg`), the compound ids are
   1-28 rather than four digits, Table 1 has 8 columns where the
   extractor knows 3, 6 and 7, the Table 1a columns are the other way
   round (`SMILES | Structure` instead of `Structure | SMILES`), and
   the activity columns hold letter grades rather than numbers.
   Feeding the pages to the current extractor returns **0 compounds and
   0 assays**.  That is the next piece of work.
2. **No mass gate is possible** on this patent: it prints no m/z.
3. **The 6 px margin is measured on 60 lines of 4 pages.**  It should
   be confirmed on more hand transcriptions before it is trusted for
   a third patent.
4. **"Parses" is not "is the right structure."**  The 19 parseable
   compounds have not been compared with the printed structure.

## 9. Files

Under `data/audit/phase5/`:

| artefact | contents |
| - | - |
| `padded_route_verification/` | all three routes over both patents, per cell |
| `repair_chain_regression/` | the shipped chain against the previous one |
| `padded_scale_ab/` | the 1x / 2x comparison behind `PADDED_SCALE` |
| `upscale_parity/` | which experiment the shipped route reproduces |
| `remaining_failures/` | per-line damage of the 4 imperfect lines |
| `cross_patent_regression/` | the two-patent constraint on the margin |
| `adaptive_padding/` | the band-geometry matrix |
| `routes_before.json`, `routes_after.json` | the byte-identical regression pair |
| `figures/wo2022251372_recognition.png` | the summary figure |

Scripts under `experiments/`:

```
.venv\Scripts\python.exe experiments\phase5_padded_route_verification.py
.venv\Scripts\python.exe experiments\phase5_repair_chain_regression.py
.venv\Scripts\python.exe experiments\phase5_padded_scale_ab.py
.venv\Scripts\python.exe experiments\phase5_upscale_parity.py
.venv\Scripts\python.exe experiments\phase5_remaining_failures.py
.venv\Scripts\python.exe experiments\phase5_make_figure.py
.venv\Scripts\python.exe data\audit\phase5\tools\snapshot_routes.py data\audit\phase5\routes_after.json
```

The hand transcriptions live in `experiments/wo2022251372_references.py`
and are marked as *text verified by hand*.  That is not a statement
about chemical structure validation.
