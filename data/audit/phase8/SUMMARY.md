# Phase 8 — the activity reader was still wrong, and a second
# recogniser settles the characters the first one loses

Date: 2026-11-04
Scope: the activity-table reader of WO2025162428, the printed-SMILES
routes of both patents, and the module split that lets a second
patent be served by configuration.

## 1. The headline

Two things were found, and both change a shipped number.

**The shipped CSV's activity columns were still wrong.**  The
`ms_polarity` column read `M+H]+` instead of `[M+H]+` on **all 406
rows** -- the opening bracket was being cut off before the recogniser
ever saw it -- and `ms_mz` was empty on 33 rows and wrong on 2 more.
The cause was that the band-based line reader reaches past the value
into the column rules on some cells.  The reader now takes a fourth
reading of the cell, the whole value region, and the policy chooses
between the readings.

**A second recogniser recovers characters that the first one drops.**
Every printed SMILES line is now read by both the English PP-OCRv4
model and the upstream PP-OCRv6 model, and the two readings are
combined by a rule that never looks at a reference.  On
WO2022251372 this takes the RDKit-parseable compounds from 23/28 to
26/28.  On WO2025162428 it takes the parseable printed SMILES from
169/335 to 171/335, and it breaks nothing.

## 2. The activity reader

### 2.1 What was wrong

The reader took three readings of every activity cell: the printed
line segmented by our own ink bands, and the Chinese and English
detector readings.  The band reading was the one that usually won,
and it has two failure modes.

**It reached into the column rules.**  The polarity cell of the
dense table reads `[M+H]+`.  The band reading returned
`T1 O1 [M+II]+` -- the residue of the neighbouring cells, plus a
misread `H`.  The opening bracket was missing on **all 406 rows** of
the shipped CSV, and the shipped CSV said `M+H]+`.

**It lost the decimal point.**  The retention time `1.53` of
compound 1048 was read as `123`: the band reader measures every band
separately, and the band that holds the `1` and the band that holds
the `53` are not the same band, so the point fell between them.

### 2.2 The fourth reading

`cpd.ocr.activity` now also reads the **whole value region** of the
cell: the rules are taken out, the remaining ink is measured, and the
result is recognised in one go.  It is the mirror image of the band
reading.

| | band reading | whole-crop reading |
|---|---|---|
| good at | a cell with two printed lines (`40-100-` above `2min`) | a cell with one printed value |
| bad at | the residue of the column rules | merging two printed lines (`42-100-`) |

Neither is right everywhere, so both are read and the policy picks.

### 2.3 The policy, measured

`experiments/phase7_activity_dump.py` dumps **all four** readings of
every activity cell -- 2025 cells over 103 pages -- and
`experiments/phase7_activity_policy.py` evaluates the candidate
policies against the archived extraction in a second, without
reading a single image again.

| policy | kd_nm | rt_min | ms_mz | lcms_method | ms_polarity | total |
|---|---:|---:|---:|---:|---:|---:|
| line first (the old behaviour) | 403 | 382 | 370 | 405 | 337 | 1897 |
| detector first (Chinese) | 381 | 338 | 356 | 364 | 400 | 1839 |
| detector first (English) | 76 | 12 | 71 | 71 | 71 | 301 |
| whole crop first | 405 | 405 | 403 | 378 | 405 | 1996 |
| crop for the numbers, line for the text | 405 | 405 | 403 | 405 | 337 | 1955 |
| **agreeing group (shipped)** | **405** | **405** | **403** | **405** | **405** | **2023** |

The shipped policy is the agreeing group: the readings are grouped
by the value they stand for, and the largest group wins.  A reading
that needed several detector boxes has already lost information and
cannot outvote a reading that was recognised as one token.

**2023 of 2030 cells agree with the archived extraction.**  The two
that do not are a single ambiguous digit each: compound 1166 reads
`1664.9` where the archive has `1664.8`, and compound 2116 reads
`1558.9` where the archive has `1538.9`.  Both are visible in the
image and both readings are plausible; they are recorded rather than
silently corrected.

## 3. A second recogniser settles the characters the first one loses

### 3.1 The idea

One recogniser reads a printed line and sometimes drops or confuses a
single character: a bracket, a parenthesis, a doubled ``@@``.  The
damage is small but systematic, and the kind of damage a *different*
model does not make in the same place.

The two models available here make different mistakes.  On the 60
hand-transcribed lines of WO2022251372 the English PP-OCRv4 mobile
model misses 5 characters and the upstream PP-OCRv6 model misses 29,
and they miss different ones: PP-OCRv6 likes to read ``O`` as ``0``,
while PP-OCRv4 drops brackets.

So every printed line is now read **twice**, once by the recogniser of
the route and once by the other one, and the two readings are
combined.

### 3.2 The rule, and what it is not allowed to look at

The combination rule is:

> **Take the combination that RDKit can parse, and among those the one
> that replaces the fewest printed lines of the primary reading.**

The rule is deliberately **not** allowed to look at a reference
transcription.  It may look at the two readings, at how much they
differ, and at whether the result is a well-formed SMILES.  That
makes it usable on a patent nobody has transcribed.

Two properties make it safe to run on every cell:

* the combinations are enumerated by how much they change the primary
  reading, so the primary reading is the *first* candidate and is
  used whenever it already parses;
* the search is bounded -- only the lines whose two readings differ
  are branched over, and at most three lines may be changed at once.

What the rule may **not** claim: choosing between two readings of the
same ink by asking which one is a well-formed SMILES is a *text*
decision.  It says "these two pixels were read in two ways and only
one of them is a valid SMILES".  It does **not** say that the result
is the compound the patent drew.  Only a comparison against the drawn
structure can say that, and that comparison is still not automated.

### 3.3 Measured on WO2022251372

The `padded_vote` route is the `padded` route with the vote added.
End to end, over the real pipeline -- same extractor, same repair
chain, same merge code:

| route | transcribed exact | transcribed edits | CER | all 28 parse | all 28 balanced |
|---|---:|---:|---:|---:|---:|
| `padded` | 6 / 6 | 0 | 0.000 % | 24 / 28 | 24 / 28 |
| `padded_vote` | 6 / 6 | 0 | 0.000 % | **27 / 28** | **27 / 28** |

The character accuracy against the hand transcriptions is **exact**
-- the vote does not make the reading worse -- while four more
compounds become well-formed.

> **These numbers were corrected in phase 9.**  Three of the six hand
> transcriptions were themselves wrong when first read from the page:
> the patent prints its chiral marker as a *circled italic "a"*, which
> is exactly the glyph that makes ``[C@@H]`` easy to mis-read as
> ``[C@H]`` or ``[C@@@H]``.  The table above is measured against the
> **corrected** transcriptions.  Measured against the original ones the
> same readings carry 4 edits, 0.320 % CER, and 4 of the 6 are exact.
> The corrections and their pixel evidence are in
> ``experiments/wo2022251372_references.py`` and
> ``data/audit/phase9/patent2_reference_audit/``.

The one compound that is still not parseable fails on a stereo
confusion (``@@`` read as ``@``), which a parse check cannot see and
which the vote cannot fix.

The per-line view (60 transcribed lines) shows where the vote gets
its material:

| reading | lines exact | edits |
|---|---:|---:|
| English PP-OCRv4 (`padded`'s own recogniser) | 56 / 60 | 5 |
| upstream PP-OCRv6 (`validated`'s own recogniser) | 32 / 60 | 29 |
| the longer of the two | 57 / 60 | 3 |

28 of the 60 lines are read differently by the two models, which is
what gives the vote something to work with.

### 3.4 Measured on WO2025162428 -- the vote repairs the `O`/`0` confusion

On the three hand-verified references the two routes are identical,
and that is the expected outcome: the `validated` route already
returns a parseable string for these compounds, so the vote keeps the
primary reading and never consults the second recogniser.

| route | 1036 | 1037 | 1038 |
|---|---:|---:|---:|
| `validated` | edit 0 | edit 0 | edit 0 |
| `validated_vote` | edit 0 | edit 0 | edit 0 |

(The 1037 bracket was repaired in phase 9, so all three references are
character exact in the delivered CSV.  See
`data/audit/phase9/SUMMARY.md`.)

Over the **whole corpus** the vote does have work to do.  Measured on
the cells the pipeline itself cuts (`experiments/phase8_pipeline_cells_corpus.py`,
335 compounds of the pages that carry a printed-SMILES column):

| route | bracket balanced | RDKit parses | cells with a character outside the SMILES alphabet |
|---|---:|---:|---:|
| `validated` | 210 / 335 | 169 / 335 | 8 |
| `validated_vote` | 211 / 335 | **171 / 335** | **8** |

The vote changed **12** strings, of which only **4** survive the
repair chain (the other eight are the `C(=0)` -> `C(=O)` repair the
chain makes anyway).  Of the four, **2** became parseable.  None of
the readings that parsed before stopped parsing.

The earlier version of this section reported 29/41 parses on 334
cells.  Those numbers were wrong: the measurement cut only the
*first band* of a printed row, so a SMILES that runs over the bottom
of a page was read as a truncated molecule.  The numbers above come
from the corrected measurement, which captures the cells the pipeline
actually hands to the recogniser and joins the continuation bands the
way the pipeline joins them.  They agree with the delivered CSV
(171 parses, `experiments/phase9_csv_stats.py`).

### 3.5 What the vote actually changes

The change is almost always the same one: the recogniser read the
letter ``O`` as the digit ``0``, so ``C(=O)`` became ``C(=0)``.  That
string is not a SMILES -- a ring-closure digit cannot stand where an
atom belongs -- so the cell failed to parse.  The second recogniser
reads the same ink as ``O``, and the vote takes it.

The four that survive the repair chain break down as:

| change | cells |
|---|---:|
| one ``0`` becomes ``O`` | 3 |
| a damaged line replaced by the second recogniser's reading | 1 |

Two of the four became parseable; the other two already parsed and
kept parsing.  None of them stopped parsing.

### 3.6 The guards, and why they are needed

Two guards keep the rule from changing chemistry while it fixes
syntax.

**The characters must be SMILES characters.**  RDKit is more
permissive than a printed column: it reads ``"CC O"`` as ``CC`` and
accepts ``"CC*O"`` as it stands.  A printed SMILES column has
neither a space nor a wildcard atom, so a combination that contains
one is rejected however well it parses.

The alphabet is built from the **element symbols**, so it holds the
letters of the two-letter ones -- ``l`` for ``Cl`` and ``r`` for
``Br`` -- and it holds the letters of the bracketed atoms of any
element (``[Se]``, ``[Cu]``, ``[Na]``).  This is not academic: the
corpus contains real chlorines (``C(Cl)=C8F``), and an alphabet
without ``l`` would reject them.  The list is checked against
RDKit's own periodic table by `tests/test_vote.py`.

**The bond order may not shrink.**  A parse check cannot tell
``C1=CC=CC=C1`` from ``C1-CC-C-C-C1``: both are valid SMILES and
they are different molecules.  Measured on the corpus, a vote
*without* this guard turns a double bond into a single bond: on the
335 cells the pipeline cuts it changes 17 strings and loses a bond in
5 of them, because the second recogniser read an aromatic ring as
saturated while the first read it as aromatic, and the whole string
only parsed with the second reading.  With the guard the vote changes
12 strings and **no cell loses a bond**
(`experiments/phase9_bond_guard_evidence.py`).

The guard is deliberately conservative in one direction: the vote may
*add* a bond (a lost ``=`` restored) and it may not *delete* one.

### 3.7 The routes

| route | geometry | recogniser | reading |
|---|---|---|---|
| `production` | detector box, 2x | English PP-OCRv4 | one model |
| `validated` | ink bands, table rules trimmed, 2x | PP-OCRv6 | one model |
| `padded` | ink bands, 6 px white margin, no upscale | English PP-OCRv4 | one model |
| `validated_vote` | as `validated` | PP-OCRv6 + English PP-OCRv4 | the vote |
| `padded_vote` | as `padded` | English PP-OCRv4 + PP-OCRv6 | the vote |

The first three are the historical routes and keep their measured
behaviour exactly.  The last two are the same routes with the vote
added, and they are new in this phase.

The two profiles now use the vote routes: `validated_vote` for
WO2025162428 and `padded_vote` for WO2022251372.  The route of a
patent is still one line of its profile, so a third patent chooses
its route in the same way.

---

## 4. The module split -- what a new patent has to change

Everything that legitimately differs between patents is **data** in
one file, `src/cpd/parsers/profiles.py`.  Nothing else in the code
asks which patent it is running on.

| patent-specific (data) | shared (code) |
|---|---|
| accepted input file patterns | page and table locating |
| cropped table vs full page | the row reader and the cell readers |
| table layouts and column positions | the printed-SMILES routes |
| compound-number syntax and geometry | the activity cell reader |
| how an activity value is written (number, letter grade) | the repair chain and the RDKit validation |
| the measured printed-SMILES route | the merge and the CSV writer |

Two patents are configured today:

| | WO2025162428 | WO2022251372 |
|---|---|---|
| input | 103 cropped table JPGs | 20 full-page TIFF scans |
| layouts | activity / dense / smiles | activity8 / smiles3 |
| activity values | numbers and free text | letter grades A-D, closed-set reader |
| compound id | four digits | one or two digits, top-left of the row band |
| printed-SMILES route | `validated_vote` | `padded_vote` |
| cross-page rows | no | yes |

Adding a third patent is a new profile plus, at most, one new reader
for a genuinely new kind of cell.  The module map is in
`docs/ARCHITECTURE.md`.

---

## 5. What is still open

* **DECIMER is shelved.**  Reading a 2D structure drawing is a much
  harder problem than reading printed text, and its output is not
  trustworthy for these macrocycles.  It stays in the code and its
  output stays in the CSV as a second opinion, but the printed text
  is the primary source.  `smiles_best` falls back to DECIMER for the
  compounds whose printed text does not parse, and those strings are
  **not** verified against the drawing.
* **The mass gate does not pass.**  The patent's printed `m/z` sits
  about +0.17 Da above the monoisotopic `[M+H]+` of the printed
  SMILES.  That points at the mass table the patent used for its own
  column, not at the printed text.  The tolerance was **not** widened
  and the gate was **not** changed; this is a chemistry decision.
  See `data/audit/phase7/MASS_OFFSET.md`.
* **No compound is verified against its drawn structure.**  The four
  levels -- text recognised, RDKit parseable, mass gate passed,
  structure verified -- are kept strictly apart, and no column claims
  the fourth.
* **Two activity cells differ from the archived extraction** by a
  single ambiguous digit (compound 1166 `m/z`, compound 2116 `m/z`).
  Both readings are plausible and both are recorded rather than
  silently corrected.
* **1037's one character error was fixed in phase 9.**  The
  recogniser drops the closing bracket of `[C@@H4`, which leaves a
  ring-closure digit attached to the chiral atom; the raw OCR also
  read one `O` as `0`, but `auto_repair_smiles` already fixed that
  (`C(=0)` -> `C(=O)`).  The vote could not help with the bracket:
  the second recogniser reads the same ink as `C(O)` -- a *missing*
  bond -- which the bond guard refuses.  The repair chain now
  restores the bracket instead, which makes 1037 character exact and
  all three hand-verified references exact together.  See
  `data/audit/phase9/SUMMARY.md`.

---

## 6. Reproducing

```powershell
# the activity reader, measured over every activity cell
.venv\Scripts\python.exe experiments/phase7_activity_dump.py
.venv\Scripts\python.exe experiments/phase7_activity_policy.py

# the vote, on WO2022251372
.venv\Scripts\python.exe experiments/phase8_wo2022251372_line_vote.py
.venv\Scripts\python.exe experiments/phase8_patent2_route_compare.py

# the vote, over the cells the pipeline itself cuts
.venv\Scripts\python.exe experiments/phase8_pipeline_cells_corpus.py

# the routes against the hand-checked references
.venv\Scripts\python.exe experiments/phase7_route_verify.py

# the test suite
.venv\Scripts\python.exe tools\run_tests.py
```

## 7. Files

| file | what it is |
|---|---|
| `src/cpd/ocr/activity.py` | the activity cell reader, now with the whole-crop reading |
| `src/cpd/ocr/lines.py` | the ink-band and box-ordering geometry both readers share |
| `src/cpd/ocr/vote.py` | the reference-free rule that combines two recognisers |
| `src/cpd/ocr/printed_smiles.py` | the five routes |
| `src/cpd/parsers/profiles.py` | the per-patent configuration |
| `data/audit/phase7/activity_dump.csv` | all four readings of all 2025 activity cells |
| `data/audit/phase7/DELIVERY_AUDIT.md` | the audit of the delivery repository |
| `data/audit/phase7/MASS_OFFSET.md` | why the mass gate does not pass |
| `data/audit/phase8/line_vote.csv` | both readings of all 60 transcribed lines |
| `data/audit/phase8/patent2_route_compare.csv` | the route comparison of WO2022251372 |
| `data/audit/phase8/pipeline_cells_route_compare.csv` | the route comparison over the pipeline's own 335 cells |
| `data/audit/phase8/patent2_vote_effect.csv` | every patent-2 cell before and after the vote |
| `data/audit/phase9/bond_guard_evidence.csv` | the bond guard, measured on the corrected cells |
| `experiments/phase9_csv_stats.py` | the authoritative statistics of the delivered CSV |
| `tests/test_vote.py` | the vote rule |
| `tests/test_activity_reading.py` | the activity reader |
| `tools/run_tests.py` | the test runner (see §8) |

---

## 8. The test suite on this machine

`pytest` creates the directories of its `tmp_path` fixture with
`mode=0o700`.  On this machine that turns into an ACL that excludes
the account the sandbox runs as, and the fixture then fails at
*setup* -- before any pipeline code runs -- with
`PermissionError: [WinError 5] Access is denied`.  It is an
environment quirk, not a defect in the code: the same test passes
when the directory is created with inherited permissions.

`tools/run_tests.py` does that, and nothing else.  The tests
themselves are untouched.

```
258 passed
```
