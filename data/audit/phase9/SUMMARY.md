# Phase 9 — a wrong measurement was corrected, and the delivery
# repository was brought into step

Date: 2026-11-04
Scope: the printed-SMILES route measurement of WO2025162428, the
repair chain's handling of a dropped chiral bracket, and the state of
the delivery repository `drug_ocr`.

## 1. The headline

Three things happened.  A number that had been quoted for two days
was found to be **wrong**, and the measurement that produced it was
measuring the wrong thing.  A systematic character error that the
repair chain did not cover was found and fixed, which made **all
three** hand-verified references character exact.  And the delivery
repository was found to be carrying a stale test alongside current
code.

The corrected measurement is:

| route | bracket balanced | RDKit parses | of 335 cells |
|---|---:|---:|---|
| `validated` | 210 | 169 | the pages that carry a printed-SMILES column |
| `validated_vote` | 211 | **171** | |

And the three hand-transcribed references are now **character
exact**:

| compound | before | after |
|---|---:|---:|
| 1036 | edit 0 | **edit 0** |
| 1037 | edit 1 | **edit 0** |
| 1038 | edit 0 | **edit 0** |

## 2. What was wrong with the old measurement

The figure quoted before was "29 of 334 cells parse with `validated`,
41 with `validated_vote`".  Both numbers were wrong, and the reason
was structural rather than a coding slip.

`experiments/phase8_pipeline_cells_corpus.py` cut **only the first
band** of a printed row.  A printed SMILES that runs over the bottom
of a page is read by the pipeline as two cells, and the pipeline
appends the second to the first.  The measurement saw only the first
half, so it read a *truncated molecule* and scored it as broken.

That is why the measurement reported 29 well-formed readings where
the delivered CSV carries 170.  Both cannot be right about the same
ink, so the two were cutting different cells -- which is what
`experiments/phase9_csv_stats.py` now settles by counting the CSV
itself.

The script was rewritten.  The extractor does not expose the cell it
hands to the recogniser, so the script subclasses the real extractor
and overrides `_ocr_smiles_with_stats` -- the one method that does
the handing over.  The subclass records the crop and returns an empty
reading, which keeps the extraction pass free of OCR; the two routes
are then run on the captured crops afterwards, on identical pixels.
The continuation bands are joined the way the pipeline joins them.

The corrected script finds **335** compounds -- exactly the 335 the
delivered CSV carries -- and its numbers agree with the CSV.

### The correction, in one table

| | before | after |
|---|---:|---:|
| compounds measured | 334 | 335 |
| `validated` parses | 29 | **169** |
| `validated_vote` parses | 41 | **170** |
| strings the vote changed | 12 | **4** |
| of those, newly parseable | not stated | **2** |
| readings the vote broke | 0 | **0** |

The two numbers that did not change are the ones that were already
right: the vote never breaks a reading that parsed before, and it
never loses a double bond.

## 2a. The bond guard, measured on the corrected cells

The vote is allowed to fix syntax, not chemistry, and a parse check
cannot tell the two apart: `C1=CC=CC=C1` and `C1-CC-C-C-C1` both
parse and they are different molecules.  So a combination that
declares fewer double or triple bonds than the primary reading is
refused.

The figures for this guard had been measured on the flawed corpus, so
they were re-measured on the 335 cells the pipeline cuts
(`experiments/phase9_bond_guard_evidence.py`):

| vote | strings changed | of those, lost a bond |
|---|---:|---:|
| with the guard (shipped) | 12 | **0** |
| without the guard | 17 | **5** |

The five compounds the guard protects are 1053, 1095, 1154, 2141 and
2160.  Compound 2160 is the worst case: without the guard it loses
four bonds at once, because the second recogniser read an aromatic
ring as saturated while the first read it as aromatic, and only the
saturated reading parsed as a whole.

So the guard is not decorative -- without it the vote would damage
five molecules to repair two.

## 3. The vote's actual effect on the corrected cells

The vote changed **12** compounds -- one printed line each:

```
1054, 1067, 1094, 1196, 1220, 1221, 1229, 1230, 1231,
2053, 2093, 2145
```

Of the twelve changes, eight are the letter `O` read as the digit
`0` -- and the repair chain fixes that pattern on its own
(`C(=0)` -> `C(=O)`).  So after the repair chain, only **4** of the
twelve still differ:

| change | cells |
|---|---:|
| one ``0`` becomes ``O`` | 3 |
| a damaged line replaced by the second recogniser's reading | 1 |

Two of the four became parseable; the other two already parsed and
kept parsing.

### Why the vote is quoted two ways

The two numbers are not in conflict -- they measure the vote at two
different points:

| measured on | the vote changes |
|---|---:|
| the raw recognition, before the repair chain | 12 compounds |
| the repaired text, which is what the CSV holds | 4 compounds |

Eight of the twelve changes are repaired by the chain anyway, so only
four of them reach the deliverable.  Both numbers are quoted in the
documents, and each is labelled with what it is measured on
(`experiments/phase9_vote_unit_check.py` reconciles them).

## 4. A dropped chiral bracket, and what it cost

Compound 1037 was described as having three character errors.  The
raw OCR really does have three, but only **one** of them reached the
deliverable -- and after this phase's repair, none of them does.

The raw OCR of 1037 reads `[C@@H4C(=O)` where the patent prints
`[C@@H]4C(=O)`: the recogniser dropped the closing bracket.  The
repair chain already handled the same failure in front of a
parenthesis (`[C@H(` -> `[C@H](`), but not in front of a
ring-closure digit.  The raw reading also read one `O` as `0`, but
the repair chain turns `C(=0)` into `C(=O)`, so that one never
reaches the deliverable.

The same failure is not rare.  Scanning the delivered CSV for a
chiral atom whose bracket was dropped finds **29 rows**, not one:

```powershell
.venv\Scripts\python.exe experiments\phase9_missing_bracket_scan.py
```

The repair is narrow.  A digit cannot sit inside a bracket atom in
SMILES -- the bracket holds an isotope, an element, an H count, a
charge and an atom class -- so `[C@@H4` is never valid SMILES and
cannot be what was printed.  The digit is restored as a ring closure,
which means it has to have a partner in the string; a digit without
one is left alone, because then the repair would be a guess.

Measured on the delivered CSV
(`experiments/phase9_bracket_rule_effect.py`):

| | rows |
|---|---:|
| changed by the new rule | 29 |
| newly parseable | 1 (compound 1037) |
| no longer parseable | **0** |

The other 28 rows keep their other error -- they read `[C@@H1(` where
the patent prints `[C@@H]1C(=O)`, so a `C` was read as a `(` as
well.  Restoring the bracket does not repair that, and it does not
hide it: those rows were unparseable before the rule and are still
unparseable after it.  A repair that fixes one error and leaves the
others visible is the behaviour we want.

For 1037 the bracket was the *only* error, so the repair makes it
character exact -- and with it, all three hand-verified references
are character exact:

```powershell
.venv\Scripts\python.exe experiments\phase9_reference_check.py
```

```
1036: edit distance 0 (271 of 271 characters)
1037: edit distance 0 (275 of 275 characters)
1038: edit distance 0 (273 of 273 characters)

3 of 3 are character exact, 0 edits in total
```

## 4a. The CSV was rebuilt, and the diff is clean

The CSV was rebuilt with the same route (`validated_vote`) and the
same DECIMER columns, and the two builds were compared column by
column (`experiments/phase9_csv_diff.py`):

| column | rows changed |
|---|---:|
| `smiles` | 29 |
| `ocr_bracket_deficit` | 22 |
| `ocr_difficulty` | 19 |
| `canonical_smiles` | 1 |
| `smiles_best` | 1 |
| `smiles_best_source` | 1 |
| `molecular_weight`, `heavy_atom_count`, `is_valid` | 1 each |
| `structure_confidence`, `ms_error_ppm` | 1 each |
| `kd_nm`, `rt_min`, `ms_mz`, `lcms_method`, `ms_polarity` | **0** |

The printed SMILES that RDKit accepts go from 170 to **171**.  One
compound becomes parseable, **none** becomes unparseable, and not one
activity value moves.

## 4b. A counting bug, found while checking

The comparison script reported that 241 of the printed SMILES parsed.
They do not: 171 do.  RDKit returns an *empty molecule* for an empty
string rather than ``None``, so the 71 rows with no printed SMILES at
all were counted as well-formed.  The check now rejects an empty
cell first.

The number was never quoted in a document -- it was caught because it
did not agree with `experiments/phase9_csv_stats.py`, which counts
the same file.  Two scripts that measure the same thing should agree;
when they do not, one of them is wrong.

## 5. The delivery repository

The delivery repository `drug_ocr` was checked against the working
one again, and the check found a new problem: its
`tests/test_profiles.py` still asserted the **old** routes
(`validated` / `padded`) while its own `profiles.py` had already
moved to the vote routes.  Three of its tests failed.

The code was right and the test was stale -- but a reviewer running
the suite in the delivery repository would have seen three failures
and no way to tell a stale assertion from a real defect.  That is
what the sync's file list is for: `tools/sync_delivery.py` now names
the files that are out of step instead of only counting them.

## 6. What is still open

* **DECIMER is shelved.**  Reading a 2D structure drawing is not
  accurate enough for these macrocycles to be trusted.  Its output
  stays in the CSV as a second opinion and as the fallback for the
  compounds whose printed text does not parse, and those strings are
  **not** verified against the drawing.
* **The mass gate does not pass.**  The patent's printed `m/z` sits
  about +0.17 Da above the monoisotopic `[M+H]+` of the printed
  SMILES.  The tolerance was **not** widened; see
  `data/audit/phase7/MASS_OFFSET.md`.
* **No compound is verified against its drawn structure.**  The four
  levels -- text recognised, RDKit parseable, mass gate passed,
  structure verified -- are kept strictly apart, and no column claims
  the fourth.
* **28 compounds still carry a lost `C` that was read as `(`.**  The
  bracket rule repairs the bracket; the mis-read character needs a
  different fix.
* **Compound 2173's reference transcription is still unverified by a
  human**, like the 1039-1041 references.
* **No compound has been checked against a *second* transcription.**
  1036, 1037 and 1038 are character exact against the hand
  transcriptions, but 3 of 335 is a small sample.  The other 332
  printed SMILES are unverified text: they parse, but nobody has
  compared them with the printed page.

## 6a. WO2022251372: the ground truth was wrong, not the recogniser

The second patent was measured against six hand transcriptions, and
it looked worse than it was: 4 edits over 1252 reference characters
(0.320 % CER), 4 of the 6 compounds exact.

Those transcriptions were re-checked against the magnified pixels,
one character at a time.  **Three of them were wrong.**  The patent
prints its chiral marker as a *circled italic "a"*, and that is
exactly the glyph that makes ``[C@@H]`` easy to mis-transcribe:

| compound | line | transcribed | printed | evidence |
|---|---:|---|---|---|
| 10 | 7 | `[C@@@H]` | `[C@@H]` (two circled ``a``) | `patent2_reference_audit/cmpd10_line07.png` |
| 28 | 3 | `[C@@H]` | `[C@H]` (one circled ``a``) | `patent2_zoom/cmpd28_band2_line03.png` |
| 28 | 5 | `NC(C` | `N(C` | `patent2_zoom/cmpd28_band2_line05.png` |

So the OCR was being marked down for transcriptions that were wrong.
Measured against the **corrected** references the same readings carry
**1 edit** (0.080 % CER) and **5 of 6** are character exact.  The
corrections are recorded in `experiments/wo2022251372_references.py`
(`CORRECTIONS`), each with the path to its pixel evidence, and
`experiments/phase9_patent2_corrected_references.py` reproduces both
sides of the correction so the original numbers stay recoverable.

### The last error was a bracket read as its mirror image

The remaining edit is compound 28 reading `C(=O)]C@@H](C)` where the
page prints `C(=O)[C@@H](C)`: the recogniser saw the *closing*
bracket where the page has the opening one.

The first attempt at a repair rule for this was **wrong**.  It
required the string to hold exactly one ``]`` more than ``[``, but a
bracket read as its mirror image moves that difference by **two** --
one bracket becomes its opposite, so the counts differ by two, not
one.  A difference of one can therefore never come from this
mistake, and the rule as written could never fire at all: it returned
0 rows on both patents' corpora, which read as a clean negative result
but was in fact a rule that could not match anything.

The corrected rule fires on the right row and on nothing else:

| corpus | readings | repaired |
|---|---:|---:|
| WO2022251372 | 28 | **1** (compound 28) |
| WO2025162428 | 335 | **0** |

It is narrow on purpose.  It only fires when the string holds two
more ``]`` than ``[``, the parentheses still balance, the string does
not parse as it stands, exactly one position can be flipped into a
string that parses, and the flip does not pull a digit inside a
bracket -- where it would change from a ring closure into an isotope,
and with it the molecule.  No reference transcription and no mass
value is consulted.

It lives in `cpd.chem.auto_repair_smiles`, so it applies to every
patent without configuration, and it changes **0 of the 406** rows of
the delivered CSV for WO2025162428.

### Where WO2022251372 stands now

End to end over the real pipeline -- same extractor, same repair
chain, same merge code:

| route | transcribed exact | edits | CER | all 28 parse | all 28 balanced |
|---|---:|---:|---:|---:|---:|
| `padded` | 6 / 6 | 0 | 0.000 % | 24 / 28 | 24 / 28 |
| `padded_vote` | 6 / 6 | 0 | 0.000 % | **27 / 28** | **27 / 28** |

Measured against the *original* transcriptions the same readings
carry 4 edits (0.320 % CER) and 4 of the 6 are exact.  The one
compound that still does not parse fails on a stereo confusion
(``@@`` read as ``@``), which a parse check cannot see.

## 7. Reproducing

```powershell
.venv\Scripts\python.exe tools\run_tests.py
.venv\Scripts\python.exe experiments\phase9_csv_stats.py
.venv\Scripts\python.exe experiments\phase8_pipeline_cells_corpus.py
.venv\Scripts\python.exe experiments\phase9_missing_bracket_scan.py
.venv\Scripts\python.exe experiments\phase9_bracket_rule_effect.py
.venv\Scripts\python.exe experiments\phase9_bond_guard_evidence.py
.venv\Scripts\python.exe experiments\phase9_csv_diff.py
.venv\Scripts\python.exe experiments\phase9_patent2_corrected_references.py
.venv\Scripts\python.exe experiments\phase9_mirrored_bracket_rule.py
.venv\Scripts\python.exe experiments\phase8_patent2_route_compare.py
```

## 8. Files

| file | what it is |
|---|---|
| `experiments/phase9_csv_stats.py` | the authoritative statistics of the delivered CSV |
| `experiments/phase9_csv_diff.py` | the two builds compared column by column |
| `experiments/phase9_missing_bracket_scan.py` | how many rows lose a chiral bracket |
| `experiments/phase9_bracket_rule_effect.py` | what the bracket rule does to the CSV |
| `experiments/phase9_bond_guard_evidence.py` | the bond guard, measured on the corrected cells |
| `experiments/phase9_zero_o_scan.py` | how often `O` is read as `0` |
| `experiments/phase9_delivery_check.py` | the delivery repository's own self-check |
| `experiments/phase9_patent2_corrected_references.py` | the corrected patent-2 references, and the size of the correction |
| `experiments/phase9_patent2_reference_audit.py` | the printed band, the reference and the reading, per line |
| `experiments/phase9_patent2_zoom.py` | the SMILES cell the pipeline hands to the recogniser, at 3x |
| `experiments/phase9_mirrored_bracket_rule.py` | the mirrored-bracket rule, measured on both corpora |
| `data/audit/phase8/pipeline_cells_route_compare.csv` | the corrected route comparison, per cell |
| `data/audit/phase9/bond_guard_evidence.csv` | the bond guard, per cell |
| `data/audit/phase9/patent2_corrected_reference_comparison.csv` | original against corrected references |
| `data/audit/phase9/patent2_reference_audit/` | the printed band of every transcribed line, with the reference |
| `data/audit/phase9/patent2_zoom/` | the magnified pixels behind the transcription corrections |
| `data/audit/phase9/mirrored_bracket_rule_evidence.csv` | the mirrored-bracket rule, per reading |
