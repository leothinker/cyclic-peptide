# Delivery repository audit — `drug_ocr`

Date: 2026-11-04
Scope: read-only audit of `D:\GitHub\drugdesign\drug_ocr` against the
working repository `D:\GitHub\drugdesign\cyclic-peptide`.

> **This is a historical record.**  It describes the state of the
> delivery repository *at the time of the audit*, before the fixes
> in §5 were applied.  The present state is in §8 and §9; the
> findings below are kept because they are what was wrong, not
> because they are still true.

> **Status: the findings below have been fixed.**  The delivery
> repository was synced from the working repository on the night of
> 2026-11-04 (`tools/sync_delivery.py --apply --prune`), the staged
> AGPL `LICENSE` was unstaged, and the repository was then verified
> **from inside itself**, with its own virtual environment, by
> `experiments/phase9_delivery_check.py`.  The result is recorded in
> §8.  The audit is kept as the record of what was wrong.

## 1. Question

The repository that was handed to the supervisor is the one that has
to work.  Two things were checked:

1. does it extract the printed SMILES of WO2025162428 correctly?
2. is it a coherent deliverable, or a copy of the working tree?

## 2. Verdict at the time of the audit

**The delivered repository did not reproduce the correct SMILES, and
the reason was a single stale default.**  The code that reads the
printed SMILES correctly was present in the working repository and
was *not* present in the delivery repository, because the delivery
repository was one sync behind.

The activity-table defects (the split `1.72`, the reordered `[M+H]+`)
and the two missing modules (`cpd/ocr/lines.py`,
`cpd/ocr/activity.py`) are the same story: they were fixed in the
working repository after the delivery repository was last synced.

| # | Finding | Evidence | Severity |
| - | - | - | - |
| D1 | The profile still selects the **old** `production` SMILES route | `drug_ocr/src/cpd/parsers/profiles.py:220` → `smiles_route="production"`; the working repo has `"validated"` at `src/cpd/parsers/profiles.py:228` | **critical** |
| D2 | `main.py` hard-codes `smiles_route: str = "production"` as the default of the public API | `drug_ocr/main.py:142` and `:621`; the working repo uses `str \| None = None` and falls back to the profile | **critical** |
| D3 | `cpd/ocr/lines.py` and `cpd/ocr/activity.py` are missing | the two `src/cpd/ocr/` listings differ by exactly these two files | **critical** |
| D4 | `LICENSE` in the working tree is the **AGPL v3** text, and the change is **staged** | `drug_ocr/LICENSE` is 35 184 B of AGPL; `git show HEAD:LICENSE` is the MIT text; `git status` shows `M  LICENSE` (staged) | **high** |
| D5 | `pyproject.toml` would build a wheel without the sub-packages | `drug_ocr/pyproject.toml` → `packages = ["cpd"]`; `cpd.ocr` and `cpd.parsers` would be missing from the wheel | high |
| D6 | 545 files are tracked, most of them outside the curated set | `git ls-files` → 545; the curated set is 279 files | medium |
| D7 | The delivered `compounds_combined.csv` carries the corrupted SMILES | see §3 | **critical** |

## 3. What the delivered CSV contained

Both files had 406 rows and 48 columns, so the *shape* was right and
the defect was invisible to a row count.  The content was not:

| compound | `drug_ocr` (delivered) | `cyclic-peptide` (working) |
| - | - | - |
| 1036 | `CC[C@H]C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C=ON3[C@@H]CC3C=O)NC)...` | `CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N...` |
| 1037 | `CC[C@H](C)C@H]1C=ON(C)[C@@H](CC2=CC=CC-C2C-O)N3[C@@H]CC3)C(=O)N(C)[C@@H4C=O)N...` | `CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](CC2=CC=CC=C2)C(=O)N3[C@@H](CC2)C(=O)N(C)[C@@H4C(=O)N...` |
| 1038 | `CC[C@H](C[C@H]1C(=O)N(C)[C@@H](C)C=O)N2[C@@H]CC2C-O)N(CCCF)F)F)[C@H]3C=O)...` | `CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCCF(F)(F)F)[C@H]3C(=O)N(C)CC(=O)N...` |

The delivered strings showed the signature of the English PP-OCRv4
recogniser: **the left bracket of every `C(=O)` is dropped** and the
remaining bracket is misread.  `C(=O)` becomes `C=O`, `C(=O)N3`
becomes `C=ON3`, `CC(=O)N` becomes `CC=O)N`.  This is the defect that
Phase 3 identified and that the `validated` route fixes.

The working repository's 1036 string was **character-for-character the
hand transcription** supplied by the researcher:

```
CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C
```

The delivered 1036 string did not match it.  So the answer to "does
the delivered repository identify the SMILES of WO2025162428
correctly" was **no**, and it was fixable by the sync, because the
correct code already existed.

## 4. What was already correct in the delivery repository

* the raw patent inputs are complete and identical
  (103 JPEG tables of WO2025162428, 20 pages of WO2022251372);
* the virtual environment resolves `cpd` to **its own** `src/`
  (`D:\GitHub\drugdesign\drug_ocr\src\cpd\__init__.py`), so the
  delivered code is what runs;
* `src/cpd/chem.py`, `src/cpd/merge.py` and `src/cpd/ocr_rewrite.py`
  are byte-identical to the working repository;
* the curated set has **no broken links** (every relative link of the
  curated markdown resolves inside the delivery repository).

## 5. The fix

`tools/sync_delivery.py` copies the curated set from the working
repository to the delivery repository and verifies every file by
SHA256.  It carries D1, D2, D3, D5 and D7.  D4 is not a file-content
problem but a **git-index** problem: the AGPL text is staged, so it
must be unstaged after the copy, or an ordinary `git commit` would
ship the AGPL licence with the MIT project.

```
.venv\Scripts\python.exe tools\sync_delivery.py --apply --prune
cd ..\drug_ocr
git restore --staged LICENSE
```

## 6. Two repositories, one code base

The two repositories have different jobs and are kept in step by
`tools/sync_delivery.py`:

* `cyclic-peptide` — the working repository: the code, every
  experiment, every audit artefact, the research log.  A new patent is
  adapted here.
* `drug_ocr` — the delivery repository: the same code, the patent
  inputs, the combined CSV and the evidence that backs the headline
  claims, and nothing else.

The manifest in `tools/sync_delivery.py` is the contract between
them: 350 curated files, checked by SHA256, with a link check so that
"the delivery repository matches the working repository" is a
verifiable statement rather than a hope.

The sync reports **which** files are out of step, not just how many.
That detail earned its place: on the second sync of the night the
delivery repository's `tests/test_profiles.py` was found to still
assert the old routes (`validated` / `padded`) while its own
`profiles.py` had already moved to the vote routes, so three of its
tests failed.  The code was right and the test was stale -- but a
reviewer running the suite in the delivery repository would have seen
three failures and had no way to tell a stale assertion from a real
defect.  The file list makes that distinction immediate.

## 7. The modular split (what makes a second patent cheap)

Adding a patent must not mean editing the OCR code.  The split is:

| patent-specific (data) | shared (code) |
| - | - |
| input file patterns | page/table locating |
| column layouts and positions | the row reader and the cell readers |
| compound-number syntax and geometry | the printed-SMILES routes |
| how an activity value is written | the repair chain and the RDKit validation |
| the printed-SMILES route | the merge and the CSV writer |

Everything patent-specific lives in `src/cpd/parsers/profiles.py`.
WO2022251372 is already served this way: it runs through the same
extractor, the same recogniser and the same repair chain, and differs
only in its profile (full-page scans instead of table crops, letter
grades instead of numbers, a header band to skip, rows that continue
over a page break).

## 8. What is still open

* The structure pass (DECIMER) is shelved.  `smiles_best` therefore
  falls back to DECIMER for the compounds whose printed text does not
  parse, and those strings are **not** verified against the drawing.
* The chemical mass gate does not pass for the printed structures;
  `data/audit/phase7/MASS_OFFSET.md` documents the investigation.  The
  tolerance was **not** widened.
* No compound has been verified against its drawn structure (the level
  the project calls L5).  Nothing in the CSV claims otherwise.

---

## 9. The repair, and how it was verified

The delivery repository was brought up to date with the working one
on the night of 2026-11-04:

```powershell
.venv\Scripts\python.exe tools\sync_delivery.py --apply --prune
git -C ..\drug_ocr restore --staged LICENSE
```

The sync reported:

```
curated files      : 318
identical          : 236      different: 13      missing in target: 69
copied 318 files into D:\GitHub\drugdesign\drug_ocr
pruned 1303 files that are not part of the curated set
after the copy     : identical 318, different 0, missing 0
broken links       : 0
```

So every curated file is now byte-identical to the working
repository's, checked by SHA256, and the 1303 files that had been
copied in by earlier syncs and one-off experiments are gone.  The
`LICENSE` was unstaged, so an ordinary commit ships the MIT licence
and not the AGPL text.

The verification runs **inside** the delivery repository, with **its
own** interpreter, so that it cannot accidentally measure the working
repository:

```powershell
cd ..\drug_ocr
.venv\Scripts\python.exe experiments\phase9_delivery_check.py
```

and it reported:

| check | result |
|---|---|
| the `cpd` package resolves to | `drug_ocr\src\cpd\__init__.py` -- **its own** code |
| the route of WO2025162428 | `validated_vote` |
| the route of WO2022251372 | `padded_vote` |
| compound 1036 | **character-exact** (edit distance 0, 271 of 271 characters) |
| compound 1038 | **character-exact** (edit distance 0, 273 of 273 characters) |
| compound 1037 | **character-exact** after the repair chain (edit distance 0, 275 of 275 characters; 3 before it) |

The delivery repository's own test suite passes there:

```
258 passed
```

**All three hand-verified references are now character exact.**  The
raw reading of 1037 still has three errors: the recogniser reads the
letter `O` as the digit `0`, so `C(=O)` becomes `C(=0)`, and it
reads a closing bracket as its mirror.  The repair chain turns
`C(=0)` into `C(=O)` and puts the bracket back, so the string that
reaches the CSV is exact.

The vote alone cannot fix the bracket.  The second recogniser reads
the same ink as `C(O)` -- a *missing* bond -- and the vote's bond
guard refuses a reading that loses a bond, because `C1=CC=CC=C1`
and `C1-CC-C-C-C1` both parse and are different molecules.  The
mirrored-bracket rule is what closes the gap, and it is guarded: it
fires only when exactly one flip of the offending bracket makes the
string parse.  See `data/audit/phase9/SUMMARY.md` §6a.

The delivery repository holds 350 curated files.  It is the code, the
patent inputs, the combined CSV and the evidence that backs the
headline numbers -- and nothing else.  The working repository keeps
the rest.
