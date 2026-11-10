# Architecture

The project turns a patent's characterisation table into a CSV of
compounds: the printed SMILES, the activity values and the structure
image.  This document describes how the code is arranged, which part
changes between patents, and which part never does.

## 1. The two kinds of change

Almost everything in this pipeline is patent independent.  What
changes between patents is a short list of facts about the *table*:

* which image files are inputs (JPEG crops of a table, or full-page
  TIFF scans);
* how many columns the table has and what each column is called;
* where the compound number is and what it looks like;
* how an activity value is written (a number, or a letter grade);
* how the printed SMILES cell has to be read.

All five live in one **profile** object
(`src/cpd/parsers/profiles.py`).  Nothing else in the code asks which
patent it is running on.  Adding a patent means adding a profile and,
at most, one new *reader* for a genuinely new kind of cell.

## 2. The module map

```
main.py                     the CLI: run_pipeline(), the argument parser
 |
 +-- cpd.parsers.profiles    per-patent configuration (the only place
 |                           that knows a patent exists)
 +-- cpd.parsers.page_tables find the ruled table inside a full page
 +-- cpd.parsers.table_extractor
 |                           layout -> CompoundRecord / AssayRecord:
 |                           column assignment, cell reading, structure crops
 +-- cpd.ocr.printed_smiles  the five routes for reading a printed SMILES cell
 +-- cpd.ocr.vote            combine two recognisers' readings of the same lines
 +-- cpd.ocr.lines           the ink-band geometry the readers share
 +-- cpd.ocr.activity        the activity cell reader
 +-- cpd.ocr.constrained     reading a cell whose value comes from a closed set
 +-- cpd.chem                RDKit validation, the OCR auto-repair, descriptors
 +-- cpd.ocr_rewrite         the deterministic carbonyl rewrite
 +-- cpd.decimer             the structure-recognition model (shelved)
 +-- cpd.render              2D rendering of a structure
 +-- cpd.ms_validation       the mass gate: does a formula reproduce the m/z?
 +-- cpd.repair_audit        chemistry invariants of a repair (atoms, rings, amides)
 +-- cpd.repair_policy       which candidate goes into smiles_best
 +-- cpd.mass_constrained_search
 |                           the speculative OCR repair search
 +-- cpd.ocr_search          the wrapper that runs it over a CSV
 +-- cpd.postprocess         the derived columns of the CSV
 +-- cpd.schema              the column schema of the CSV
 +-- cpd.merge               merging records that arrive on several pages
```

## 3. The pipeline, step by step

`main.run_pipeline()` does four things.

### 3.1 Scan

`_iter_images(img_dir, profile)` returns the input files that match
the profile's patterns, in a stable order.  The pattern list is part of
the profile because one patent ships JPEG crops of a table and another
ships full-page TIFF scans.

### 3.2 Extract

`_extract_all()` walks the pages and calls the extractor.  For each
page the extractor

* works out the table layout (which profile layout matches the number
  of columns or the header words it can read);
* crops the structure column to `data/processed/cells/`;
* reads the activity cells -- numerically, or through the constrained
  reader when the profile says the values are a closed set;
* reads the printed SMILES column through the route the profile names.

Records are keyed by compound number.  A printed row may run over the
bottom of a page and resume on the next one; the resumed band carries
no number, and the extractor appends its SMILES to the compound that
is still open rather than starting a new record.

### 3.3 Reading an activity cell

An activity cell holds one short printed value: a retention time, an
m/z, a potency, a gradient, an ionisation mode.  `cpd.ocr.activity`
reads it, and the rule it follows is that **a printed value is one
token**, so the detector must not be allowed to decide how the value
is cut into pieces.

The reader takes three readings of the cell:

* the **line reading** -- the printed line is segmented by our own ink
  bands (`cpd.ocr.lines`) and recognised in recognition-only mode, so
  the value is one string by construction;
* the **Chinese** and **English** detector readings, used as
  corroboration.

Two defects made the old reader produce wrong numbers, and both are
fixed here.

**A number split into several boxes lost its decimal point.**  The
English detector cut the printed `1.72` into the boxes `1` and `72`.
The boxes were joined without a separator, so the retention time came
out as `172`.  On a dense page the printed `1.48` was cut into `7`,
`48` and `-`, which does not parse as a number at all and was dropped
from the CSV.  A reading that needed several boxes has therefore lost
information, and it cannot outvote a reading that was recognised as
one token.

**The boxes were put back in the wrong order.**  They were sorted by
their vertical midpoint, which only works while every box on a line
has the same height.  A line such as `[M+H]+` mixes tall brackets
with shorter letters, so the midpoint order returned `H] + [M+] 1`.
`cpd.ocr.lines.order_boxes` now groups the boxes into lines by their
vertical overlap and sorts them left to right inside a line.

When the readings agree -- even when they only differ in spacing, as
`10-80- 2min` and `10-80-2min` do -- the detector's text is kept, so
the column keeps the format it already had.  When they disagree, the
line reading wins.

**A sparse cell must be trimmed before it is recognised.**  The
compound-number column of the dense activity table is about 330 px
wide and almost empty: the digits occupy a narrow block in the
middle.  The detector, given the whole cell, sees a large
low-contrast canvas and returns *no box at all* for many rows, even
though the digits are perfectly legible.  `_read_cmpd_id` therefore
reads the bounding box of the cell's dark content first and keeps the
untrimmed crop as a fallback.  On page I100377 that changed 26 of 58
numbers being read into 58 of 58.

All three are *reader* defects: the patent's table was always legible.
They are locked by `tests/test_activity_reading.py`, and the reading
is checked against the archived extraction by
`experiments/phase7_activity_validate.py`.

### 3.4 Structure recognition (shelved)

`_run_decimer()` runs DECIMER on every cropped structure and caches the
result on disk.  DECIMER reads a *2D structure image*, which is a much
harder problem than reading printed text, and its output is currently
not good enough to be trusted for these macrocycles.  The stage is
kept because its output is still a useful second opinion and because
the cache makes it a one-off cost.

### 3.5 Assemble

`_build_combined_rows()` flattens the records into the pipeline
columns.  Two more layers are then applied, in this order:

* `cpd.repair_policy.process_rows()` -- the repair layer.  It decides
  what goes into `smiles_best` and records the mass verdict.  The
  printed SMILES wins whenever it parses, because it is the patent's
  own text; DECIMER's output is the fallback and a validated repair is
  the last resort.
* `cpd.postprocess.apply_derived_columns()` -- the derived layer: the
  ring statistics of the DECIMER structure, the data-quality flags and
  the bracket deficit of the printed text.

The OCR-search layer (`cpd.ocr_search`) is *not* run by default.  It
proposes strings that reproduce the measured mass, and a proposal is
not a reading.

## 4. The printed-SMILES routes

`cpd.ocr.printed_smiles` implements five routes.  A profile names
one of them.

| route | recogniser | crop | used by |
| - | - | - | - |
| `production` | English PP-OCRv4 mobile | detector box, 2x | the historical behaviour; still selectable, still reproduces the first-generation CSV |
| `validated` | upstream PP-OCRv6 | own ink-band segmentation, table rules trimmed, 2x, 14/15 px band padding | the single-recogniser form of the WO2025162428 route |
| `padded` | English PP-OCRv4 mobile | own ink-band segmentation, 6 px white margin, no upscale | the single-recogniser form of the WO2022251372 route |
| `validated_vote` | PP-OCRv6 + English PP-OCRv4 | as `validated` | **WO2025162428** -- 171/335 well-formed against 169/335 |
| `padded_vote` | English PP-OCRv4 + PP-OCRv6 | as `padded` | **WO2022251372** -- 27/28 well-formed against 24/28 |

The `production` route is kept because it is the baseline the newer
routes are measured against, not because it is good: the English
PP-OCRv4 recogniser drops the left bracket of every `C(=O)`, so none
of the 335 printed SMILES it produced parse.

The two patents need opposite settings, and that is measured rather
than assumed.  On WO2022251372 the printed lines are far enough apart
that a 6 px margin reaches only white paper, and it fixes the last
glyph of every line.  On WO2025162428 the rows are packed tightly
enough that the same margin pulls in fragments of the neighbouring
lines and makes the result much worse.  Hence two routes rather than
one parameter.

### 4.1 The vote routes

`cpd.ocr.vote` combines the readings of two recognisers.  One
recogniser reads a printed line and sometimes drops or confuses a
single character; a *different* model does not make the same mistake
in the same place.  So a vote route reads every printed line with the
route's own recogniser **and** with the other one, and combines them
with one rule:

> **Take the combination that RDKit can parse, and among those the one
> that replaces the fewest printed lines of the primary reading.**

The rule is reference-free: it may look at the two readings, at how
much they differ, and at whether the result is a well-formed SMILES,
but never at a hand transcription.  Two properties make it safe to run
on every cell:

* the combinations are enumerated by how much they change the primary
  reading, so the primary reading is the *first* candidate and is
  kept whenever it already parses;
* the search is bounded -- only the lines whose two readings differ
  are branched over, and at most three lines may be changed at once.

Two **guards** keep the rule from changing chemistry while it fixes
syntax:

* *the characters must be SMILES characters.*  RDKit is more
  permissive than a printed column -- it reads `"CC O"` as `CC` and
  accepts `"CC*O"` -- so a combination holding a space or a wildcard
  atom is rejected however well it parses.  The alphabet is built
  from the **element symbols**, so it holds the second letter of the
  two-letter ones (`l` for `Cl`, `r` for `Br`) and the letters of any
  bracketed atom (`[Se]`, `[Cu]`, `[Na]`).  The corpus contains real
  chlorines, so an alphabet without `l` would reject them.
* *the bond order may not shrink.*  `C1=CC=CC=C1` and
  `C1-CC-C-C-C1` both parse and they are different molecules, so a
  parse check cannot tell them apart.  A combination that declares
  fewer double or triple bonds than the primary reading is refused.
  The vote may *add* a bond -- a lost `=` restored -- and may not
  delete one.  Measured on the 335 cells the pipeline cuts, a vote
  without this guard lost a bond in 5 of the 17 strings it changed;
  with it, no cell loses one.

The rule is a **text-level** decision.  Choosing between two readings
of the same ink by asking which one is a well-formed SMILES says
"these two pixels were read in two ways and only one of them is a
valid SMILES".  It does **not** say that the result is the compound
the patent drew; only a comparison against the drawn structure can
say that, and that comparison is not automated.

### 4.2 The repairs that run afterwards

The recognisers and the vote settle the *text*.  What happens to that
text before it becomes a CSV column is `cpd.chem.auto_repair_smiles`,
which is shared by both patents and by the structure-recognition
output, so a repair is a rule about SMILES and never about a patent.

Most of its rules are lexical -- a carbonyl read as `C-O)`, the `]`
of a chiral atom dropped, a ring-closure digit inside a bracket atom.
One rule needs a parse check, because the damage it repairs is a
*substitution* rather than a deletion:

> **A bracket read as its mirror image.**  The recogniser sometimes
> sees the closing bracket where the page has the opening one, so
> `C(=O)[C@@H](C)` is read as `C(=O)]C@@H](C)`.

The signature is precise, and that is what makes the rule safe.  A
bracket turned into its opposite moves the difference between the
counts of `[` and `]` by **two**, so the string holds two more `]`
than `[`; the parentheses still balance, so the confusion is not
wider than one bracket; and the string does not parse as it stands.
The rule then flips a closing bracket into an opening one and accepts
the result only when it is *unambiguous* -- exactly one position
yields a string that parses -- and when the flip does not pull a digit
inside the bracket, where it would change from a ring closure into an
isotope, and with it the molecule.

Measured on the two patents' corpora the rule fires on 1 of 28
compounds of WO2022251372 and on 0 of 335 of WO2025162428, and it
changes 0 of the 406 rows of the delivered CSV.  It consults no
reference transcription and no mass value.

## 5. What the columns mean

The four levels below are **not** interchangeable, and the column
names keep them apart.

| level | what it says | column |
| - | - | - |
| L1 text | the recogniser returned this string | `smiles` |
| L2 syntax | the string is a well-formed SMILES | `canonical_smiles`, `is_valid` |
| L3 text repair | the string was rewritten deterministically | `smiles_ocr_repaired`, `ocr_repair_trace` |
| L4 chemistry | the formula reproduces the measured m/z | `ms_check_verdict`, `ms_error_ppm`, `structure_confidence` |
| L5 structure | the graph is the compound in the patent | *no column claims this* |

`structure_confidence` names the level that was actually reached:

| value | meaning |
| - | - |
| `printed_mass_verified` | the printed SMILES parses and its formula reproduces the measured m/z |
| `printed_text` | the printed SMILES parses; nothing independent confirms the structure |
| `mass_verified` | a DECIMER-derived structure whose formula reproduces the m/z |
| `mass_mismatch` | it does not |
| `no_mass_reference` | there is a structure but no usable m/z |
| `no_structure` | the row has no candidate structure |

No column claims L5.  A SMILES that parses is a well-formed string; a
SMILES that matches the mass is a string with the right formula.  Only
a comparison against the printed structure image can say whether the
graph is the right molecule, and that comparison is not automated yet.

## 6. Adding a patent

1. Put the source images in `data/raw/<patent>/golden_tables/`.
2. Add a profile to `src/cpd/parsers/profiles.py`:
   * `image_patterns` -- what the input files are called;
   * `source_mode` -- `table_image` (already cropped) or `full_page`;
   * `layouts` -- the columns of each table of that patent, and their
     fractional positions (or nothing, for a full page);
   * `cmpd_id_pattern`, `cmpd_id_top_px`, `cmpd_id_inset_px` -- where
     the compound number sits;
   * `assay_fields` -- the activity columns and how each is written;
   * `smiles_route` -- the route that reads the printed SMILES.

   A field in `assay_fields` whose name is not one of the five
   canonical ones (`kd_nm`, `rt_min`, `ms_mz`, `lcms_method`,
   `ms_polarity`) becomes an **extra column** of that patent's own
   combined CSV.  `TableProfile.extra_assay_fields` lists them and
   `cpd.schema.combined_columns()` appends them after the canonical
   block, in declaration order.  The values travel on
   `AssayRecord.extra`, which the extractor fills from the row and
   `_build_combined_rows` writes out.  A patent that declares no
   extra field -- the first one -- keeps exactly the shipped 48
   columns.

3. Run it:

   ```
   .venv\Scripts\python.exe main.py --profile <patent>
   ```

   The combined CSV lands in `data/processed/compounds_combined.csv`
   for the first patent.  Give a second patent its own output
   directory so the two deliverables cannot overwrite each other:
   `main.py --profile <patent> --out-dir data/processed/<patent>`.

4. Check the result against a hand transcription of a few rows before
   trusting the numbers.

If the patent's table has a column kind the pipeline does not know
how to read, add a reader next to
`cpd/ocr/constrained.py` and wire it into the extractor through the
profile -- not as an ``if patent == ...`` branch.

## 7. The two repositories

`cyclic-peptide` is the working repository: the code, every
experiment, every audit artefact and the research log.

`drug_ocr` is the delivery repository: the same code, the patent
inputs, the deliverable CSV and the evidence documents -- and nothing
else.

`tools/sync_delivery.py` copies the curated set from `cyclic-peptide`
into `drug_ocr` and verifies every file by SHA256:

```
.venv\Scripts\python.exe tools/sync_delivery.py
.venv\Scripts\python.exe tools/sync_delivery.py --apply --prune
```

