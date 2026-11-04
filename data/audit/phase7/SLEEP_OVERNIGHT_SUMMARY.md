# Overnight work summary — 2026-11-04

What was asked for, what was found, and what the state of the two
repositories is now.  Every number below is recomputed from a file
that ships with the repository.

## 1. The headline

The version handed to the supervisor **would have reproduced a data
loss**, and two independent defects in the activity-table reader were
responsible.  Both are now fixed, both are locked by tests, and the
second patent now runs through the same code as a configuration
change.

## 2. What was wrong with the delivered repository

`drug_ocr` was a copy of `cyclic-peptide` taken before the adapter
layer existed.  Comparing the two file by file:

| area | state of `drug_ocr` before tonight |
|---|---|
| profile registry (`cpd.parsers.profiles`) | present but with the **old** default route |
| `--profile` switch on the CLI | **absent** — only one patent could be run |
| `cpd.schema`, `cpd.postprocess`, `cpd.ocr_search` | **absent** |
| third OCR route (`padded`) | present |
| candidate scoring | **defective** (defect 1) |
| compound-number reading | **defective** (defect 2) |
| `docs/ARCHITECTURE.md`, `tools/sync_delivery.py` | **absent** |
| audit evidence | 227 tracked files, mostly one-off scratch |
| `LICENSE` | overwritten with the **AGPL** text by accident; the project is MIT |
| venv editable install | pointed at the **working** repository, so the tests could measure the wrong tree |

## 3. The two defects

Both are *reader* defects.  The patent's activity table was always
legible.

### Defect 1 — the candidate scorer preferred garbage

`_score_candidate` scored an over-segmented reading (one box per
character) at `-1.0`, while a *correct* reading scored `-2.0`.  The
"numeric-tolerant" pattern `[\d.,\- <>~nNaAmMpPuU/]+` does not allow
the letters in `40-100-2MIN` (the `I`) or in `[M+H]+` (`[`, `]`,
`+`), so the correct reading was penalised and the shattered string
won the vote.  The column was written with `1 R R - X 1 1 4 +`.

Over-segmented readings now score `-10.0`, below anything plausible.

### Defect 2 — the compound number was dropped on many rows

The compound-number cell is ~330 px wide and almost entirely empty;
the digits occupy a narrow block in the middle.  The detector was
handed the whole cell, saw a large low-contrast canvas and returned
**no text box at all** for many rows.  On page I100377 that was
**26 of 58** numbers read.  Trimming the cell to the bounding box of
its dark content first gives **58 of 58**.

Both defects were pre-existing: an A/B run of the old and the new
extractor over the same page confirmed that the old code produced
the same 26 of 58.  The loss was confined to the seven dense
activity pages (I100376–I100382); the twenty-two SMILES pages were
never affected.  **184 activity rows** were lost across those pages.

### A third defect, found by the rebuild itself

The repair layer's `_num()` helper assumed every cell was a string,
but the extractor hands it a float.  The rebuild crashed at the
repair stage with
`AttributeError: 'float' object has no attribute 'strip'`.
`_num()` now accepts both, and the behaviour is pinned by a test.

## 4. The adapter layer

Everything that differs between patents now lives in one place,
`src/cpd/parsers/profiles.py`.  Nothing else in the code asks which
patent it is running on.

| | WO2025162428 | WO2022251372 |
|---|---|---|
| input | 103 cropped JPEG tables | 20 full-page TIFF scans |
| layouts | activity / dense / smiles | activity8 / smiles3 |
| compound id | four digits | one or two digits, top-left of the band |
| activity values | numbers and free text | letter grades A–D, closed-set reader |
| printed-SMILES route | `validated` | `padded` |
| cross-page rows | no | yes |

Re-measured tonight, unchanged from the phase-6 numbers:

```
WO2022251372   pages 20 | compounds 28 | RDKit-parseable 23
                structures 28 | assay rows 28
                hand references 6 | exact 4/6 | CER 0.48 %
                printed lines touching a column rule 0 of 141
```

The two patents need *opposite* settings and that is measured, not
assumed: `padded` fixes the last glyph of every line on patent 2 and
makes patent 1 much worse, because there the rows are packed tightly
enough that the same margin pulls in fragments of the neighbouring
lines.

## 5. DECIMER

Shelved, as agreed.  Its output stays in its own columns and is not
promoted into the deliverable on its own.  The rebuild reuses the
existing DECIMER columns instead of re-running the model, so the
printed-SMILES work ships without touching it.

## 6. What did **not** change

The mass gate still rejects every candidate, and that is now
understood rather than mysterious: the patent's printed `m/z` sits
about **+0.17 Da** above the monoisotopic `[M+H]+` of the printed
SMILES (compound 1036: printed 1610.9 → implied neutral 1609.8927
versus exact 1609.6912, i.e. +125 ppm, against a tolerance of
±0.05 Da ≈ 31 ppm).  The tolerance was **not** widened.

No claim about chemical *structure* is made anywhere.  A SMILES that
parses is a well-formed string; a SMILES that matches a mass is a
string with the right formula.  Comparing the graph with the printed
structure image is still not automated, and the reference strings are
text-level hand transcriptions, not chemical ground truth.

## 7. The two repositories

`tools/sync_delivery.py` copies a curated set from the working
repository to the delivery repository and verifies every file by
SHA256, so "the delivery repo matches the working repo" is a
checkable statement rather than a hope:

```powershell
# report what would change (the default is a dry run)
.venv\Scripts\python.exe tools\sync_delivery.py

# copy, and drop everything that is not part of the curated set
.venv\Scripts\python.exe tools\sync_delivery.py --apply --prune
```

The curated set now also carries the scripts that reproduce the
headline numbers (`experiments/rebuild_csv.py`, the phase-6 and
phase-7 audit scripts, `analysis/verify_deliverable.py`), so a
reviewer can re-run the audit instead of taking it on trust.

## 8. Files

| Path | What |
|---|---|
| `src/cpd/parsers/table_extractor.py` | the two reader fixes |
| `src/cpd/repair_policy.py` | `_num()` accepts a parsed number |
| `tests/test_activity_reading.py` | regression tests for all three defects |
| `conftest.py` | pins the tests to *this* checkout |
| `src/cpd/parsers/profiles.py` | the patent registry |
| `src/cpd/ocr/printed_smiles.py` | the three recogniser routes |
| `docs/ARCHITECTURE.md` | module map, what changes per patent, how to add one |
| `tools/sync_delivery.py` | the curated set and the sync |
| `data/audit/phase7/` | the evidence for this round |
| `PROGRESS.md` | the research log, newest section at the top |
