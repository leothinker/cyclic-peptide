# Cyclic Peptide Dataset Pipeline

<!-- role-banner:start -->
> **Working repository.**  This is `cyclic-peptide`: the code, every
> experiment, every audit artefact and the research log.  The
> reviewable deliverable is a separate repository, `drug_ocr`, which
> `tools/sync_delivery.py` fills from here and verifies file by file.
<!-- role-banner:end -->

Extract cyclic-peptide SMILES and KRAS G12V GDP KD bioactivity data from
patent characterization tables (e.g. WO2025162428). One combined CSV is the
main deliverable: **406 cmpds x 48 columns** -- SMILES, DECIMER 2D re-render,
LCMS RT / MS m/z / method / polarity, KRAS G12V GDP KD, OCR trust scores,
macrocycle-intact diagnostic, and a post-hoc repair stage that is audited
against both chemical invariants and the patent's own LC-MS mass before
anything is promoted. See `PROGRESS.md` for the current findings.

The printed-SMILES column can be read by five interchangeable OCR routes
(see [Reading the printed SMILES](#reading-the-printed-smiles)):
`production` (the historical route, which produced the first-generation
CSV), `validated` (the route that reaches character-level accuracy on
WO2025162428), `padded` (the route measured on WO2022251372), and the two
*vote* routes `validated_vote` / `padded_vote`, which read every printed
line with a second recogniser as well and let the two readings settle the
characters they disagree on. The patent's profile names the route;
`--smiles-route` overrides it. **The two profiles use the vote routes**:
`validated_vote` for WO2025162428 and `padded_vote` for WO2022251372.

The pipeline is **patent-configured, not patent-specific**: everything
that differs between patents -- the input file kind, the table layout,
the compound-number syntax, how an activity value is written, and how
the printed SMILES is read -- lives in a profile in
`src/cpd/parsers/profiles.py`. Two patents are configured today:
WO2025162428 (cropped JPEG tables) and WO2022251372 (full-page TIFF
scans with letter-grade activity values). See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module map and for
how to add a third.

## Quick Start

```powershell
# 1. Install dependencies (one-time)
uv sync

# 2. Optional: install DECIMER for OCSR SMILES (pulls PyTorch ~2 GB).
#    Skip it if you only need the printed-SMILES column; the OCSR stage
#    is currently shelved because it is not accurate enough to trust.
uv pip install --python .venv/Scripts/python.exe DECIMER

# 3. Run the pipeline (reads data/raw/WO2025162428/golden_tables by default)
$env:PYTHONIOENCODING = "utf-8"
uv run python main.py

# 4. Run it for the other patent
uv run python main.py --profile WO2022251372

# 5. Build the second patent's deliverable CSV
uv run python experiments/phase10_patent2_delivery.py

# 6. Read the printed SMILES with a specific route
#    (any of: production, validated, padded, validated_vote, padded_vote)
uv run python main.py --smiles-route validated_vote

# 7. Rebuild only the OCR half of the CSV, keeping the DECIMER columns
uv run python experiments/rebuild_csv.py --route validated_vote \
    --base data/processed/compounds_combined.csv \
    --out data/processed/compounds_combined.csv

# 8. Run the test suite
# On Windows the sandbox denies os.scandir on the directories pytest
# creates for its tmp_path fixture; tools/run_tests.py works around it.
uv run python tools/run_tests.py
```

When DECIMER is not installed the `smiles_decimer` column stays empty;
install the package and re-run to fill it.

## Main Outputs

- `data/processed/compounds_combined.csv` -- one row per cmpd, 48 columns:
  cmpd_id, smiles, canonical_smiles, smiles_decimer,
  canonical_smiles_decimer, smiles_best, smiles_best_source, smiles_agree,
  molecular_weight, heavy_atom_count, is_valid, structure_image,
  source_image, kd_nm, rt_min, ms_mz, lcms_method, ms_polarity,
  assay_source_image, decimer_min_conf, decimer_low_conf_count,
  decimer_render_path, ocr_avg_entropy, ocr_min_top1, ocr_low_conf_count,
  data_quality_flags, data_quality,
  decimer_largest_ring_size, decimer_macrocycle_intact, decimer_num_rings,
  repaired_smiles, is_repaired, repair_method, smiles_candidate_repair,
  repair_confidence, repair_review_status, structure_confidence,
  repair_atoms_invented, repair_atoms_deleted, ms_check_verdict,
  ms_error_ppm, smiles_ocr_repaired, ocr_repair_trace, ocr_repair_formula,
  ocr_repair_largest_ring, ocr_repair_mass_ppm, ocr_bracket_deficit,
  ocr_difficulty.
  Read with `pandas.read_csv("data/processed/compounds_combined.csv")`.
- `data/processed/WO2022251372/compounds_combined.csv` -- the second
  patent's own deliverable: 28 rows x 54 columns.  The 48 canonical
  columns come first, then the six letter-grade activity columns
  (`mdm2`, `mdm4`, `molm13`, `mv4_11`, `waga`, `mkl1`) that this
  patent has instead of the five numeric ones.  The column set is
  profile-driven, so the first patent's CSV keeps exactly its
  shipped 48 columns and its bytes.
  Print it readably with
  `uv run python experiments/phase10_patent2_inspect.py`.
- `data/processed/cells/{src_stem}_{cmpd_id}_struct.png` -- cropped
  structure image per compound (406 PNGs).
- `data/processed/rendered/{cmpd_id}.png` -- DECIMER-canonicalised
  re-render of every valid structure (406 PNGs; CoordGen + linearised
  macrocycle when intact, as-is when broken).
- `PROGRESS.md` -- current progress report: the repair audit, the LC-MS mass
  validation, and the corrected recovery figures.

### data_quality distribution (pre-repair diagnostic)

| label                | count | meaning                                              |
|----------------------|------:|------------------------------------------------------|
| ok                   |   239 | intact cyclic peptide (largest ring >= 12 atoms)     |
| macrocycle_broken    |   141 | DECIMER SMILES parses but its largest ring is < 12   |
| partial              |    26 | multi-fragment / parse edge case                     |

The earlier `macrocycle_repair` pass filled `repaired_smiles` for 121 of
those 141 rows, and an earlier report quoted that as an 86% recovery rate.
That number has since been audited and does not hold up: most of those
repairs invented atoms or bonds. See `PROGRESS.md` for the audited figures
and for the two gates (chemical invariants, and the patent's own LC-MS mass)
that now decide whether a repair may be promoted to `smiles_best`.

## Project Structure

```text
cyclic-peptide/
 +- main.py                       # Pipeline entry point (OCR + DECIMER -> one CSV)
 +- pyproject.toml                # Project metadata + dependency pins
 +- uv.lock                       # Locked dependency graph
 +- README.md                     # This file
 +- PROGRESS.md                   # Progress report + audit results
 +- LICENSE
 +- src/cpd/
 |   +- __init__.py
 |   +- chem.py
 |   +- decimer.py
 |   +- macrocycle_repair.py
 |   +- mass_constrained_search.py
 |   +- merge.py
 |   +- ms_validation.py
 |   +- ocr_edit_model.py
 |   +- ocr_mass_gate.py
 |   +- ocr_repair_costs.py
 |   +- ocr_rewrite.py
 |   +- ocr_search.py
 |   +- postprocess.py
 |   +- render.py
 |   +- render_polygonal.py
 |   +- repair_audit.py
 |   +- repair_policy.py
 |   +- schema.py
 |   +- ocr/
 |       +- __init__.py
 |       +- activity.py
 |       +- constrained.py
 |       +- lines.py
 |       +- printed_smiles.py
 |       +- vote.py
 |   +- parsers/
 |       +- __init__.py
 |       +- beam_search.py
 |       +- grammar.py
 |       +- headers.py
 |       +- page_tables.py
 |       +- prefix_validator.py
 |       +- profiles.py
 |       +- table_extractor.py
 +- tests/
 |   +- __init__.py
 |   +- test_activity_reading.py
 |   +- test_chem.py
 |   +- test_combined_pipeline.py
|   +- test_delivery_contract.py
 |   +- test_macrocycle_repair.py
 |   +- test_merge.py
 |   +- test_new_repair_strategies.py
 |   +- test_ocr_repair_search.py
 |   +- test_ocr_rewrite.py
 |   +- test_postprocess.py
 |   +- test_prefix_validator.py
 |   +- test_printed_smiles_routes.py
 |   +- test_profiles.py
 |   +- test_render.py
 |   +- test_repair_audit.py
 |   +- test_sync_delivery.py
 |   +- test_table_pipeline.py
 |   +- test_vote.py
 +- tools/
 |   +- check_readme_tree.py
 |   +- run_tests.py
 |   +- sync_delivery.py
 +- experiments/                  # One-off analysis scripts (not a package)
 +- analysis/                     # Exploratory analysis scripts
 +- docs/
 |   +- ARCHITECTURE.md           # module map, the routes, adding a patent
 +- data/
 |   +- raw/WO2025162428/
 |   |   +- golden_tables/        # 103 source pages used by the pipeline
 |   +- raw/WO2022251372/
 |   |   +- golden_tables/        # 20 full-page TIFF scans
 |   +- processed/
 |   |   +- compounds_combined.csv
 |   +- audit/                    # evidence: phase2 / phase3 / demos
 +- references/                   # Read-only upstream OCSR/OCR source clones
 +- .gitignore
```

## Pipeline Workflow

1. **Header aliasing (`cpd.parsers.headers`)**
   Patent authors write the same column under different names
   (`Cmpd #` vs `Compound #`, `G12V GDP KD nM` vs `KRAS G12V GDP KD (nM)`).
   A small dictionary collapses every variant to one canonical field.
2. **Unified table extraction (`cpd.parsers.table_extractor`)**
   OCR the top header band to learn the column layout, then for every
   row OCR each text cell and crop the `structure` column to a PNG.
   One code path handles all three layouts found in golden tables:
   3-col SMILES, 7-col activity (with structure), and 6-col dense
   activity (no structure). Continuation pages reuse the previous
   image's layout when their own header is missing.
3. **Chemical validation (`cpd.chem`)**
   Auto-repair common OCR artefacts (amide carbonyls `C(O)N -> C(=O)N`,
   unclosed chiral brackets, lopped-off `CC[C@H` prefixes) and validate
   the canonicalised SMILES with RDKit.
4. **DECIMER OCSR (`cpd.decimer`)**
   When the DECIMER package is installed, run it on every cropped
   structure PNG to get a SECOND SMILES string plus per-token
   confidences. Results are cached on disk
   (`data/processed/decimer_cache.json`) so a second run only re-does
   new cells.
5. **Macrocycle diagnostic + 2D re-render (`cpd.render`)**
   `is_macrocycle_intact(smi)` returns True when the parsed molecule's
   largest ring is at least 12 atoms -- the threshold for a real
   cyclic-peptide backbone. The same module renders DECIMER
   canonicals: when the macrocycle is intact it opens one amide C-N
   bond to linearise the chain for cleaner thumbnails; when broken it
   renders the molecule as-is. `ring_closure_audit(smi)` reports the
   per-digit ring-closure counts for diagnostics.
6. **Macrocycle repair (`cpd.macrocycle_repair`)** -- NEW
   For every SMILES that trips the macrocycle gate, try two
   graph-level heuristics (ring-closure digit swap, terminal bridge)
   and keep the first one that produces an intact macrocycle. This
   "produces an intact macrocycle" test is necessary but **not**
   sufficient -- see step 8.
7. **Repair audit (`cpd.repair_audit` + `cpd.ms_validation`)**
   Every repair is checked against two independent gates before it can
   be trusted: a chemistry gate (did the repair invent or delete atoms,
   lose stereocentres, or produce an implausible macrocycle?) and a mass
   gate (does the resulting formula reproduce the patent's own reported
   `m/z`?). Repairs are tiered `high` / `medium` / `low` / `experimental`
   / `rejected` and only validated ones reach `smiles_best`.
8. **Combined-table output (`cpd.repair_policy` + `main.py`)**
   `main.py` runs OCR + DECIMER in a single pass and writes one
   `compounds_combined.csv` with both SMILES streams side-by-side,
   RDKit canonical forms, an `smiles_agree` flag, the activity
   columns, the macrocycle diagnostic, the per-token
   DECIMER/OCR trust scores, the post-hoc repair result, and the audit
   columns (`repair_review_status`, `structure_confidence`,
   `ms_error_ppm`).

## Reading the printed SMILES

The patent prints its SMILES as plain text in a table column, so the
text can be read directly instead of being inferred from the 2D
structure image. Five routes are implemented in
`src/cpd/ocr/printed_smiles.py`. A patent's profile names one of them;
`--smiles-route` overrides the profile:

| | `production` | `validated` | `padded` | `validated_vote` | `padded_vote` |
|---|---|---|---|---|---|
| recogniser | English PP-OCRv4 mobile | upstream default (PP-OCRv6 rec) | English PP-OCRv4 mobile | PP-OCRv6 + English PP-OCRv4 | English PP-OCRv4 + PP-OCRv6 |
| line segmentation | OCR detector boxes | our own ink-band projection | our own ink-band projection | our own ink-band projection | our own ink-band projection |
| crop geometry | trim to content, 2x upscale | trim table rules, pad each ink band | trim table rules, 6 px white margin | as `validated` | as `padded` |
| cell column | fractional boundary | extended to the table's right-hand rule | extended to the table's right-hand rule | extended to the table's right-hand rule | extended to the table's right-hand rule |
| upscale | 2x | 2x | none | 2x | none |
| reading | one model | one model | one model | the vote | the vote |
| measured on | the first-generation CSV | WO2025162428 | WO2022251372 | WO2025162428 | WO2022251372 |
| status | historical baseline | measured | measured | **the default for WO2025162428** | **the default for WO2022251372** |

The two patents need opposite settings, and that was measured rather
than assumed. On WO2022251372 the printed lines are far enough apart
that a 6 px margin reaches only white paper, and it fixes the last
glyph of every line (60 hand-transcribed lines: 31 errors without the
margin, 5 with it). On WO2025162428 the rows are packed tightly
enough that the same margin pulls in fragments of the neighbouring
lines and makes the result much worse (5 references: 15 errors
without the margin, 95 with it). Hence two routes rather than one
tunable parameter.

### The vote routes: two recognisers, one string

One recogniser reads a printed line and sometimes drops or confuses a
single character. The damage is small but systematic, and the kind of
damage a *different* model does not make in the same place. On the 60
hand-transcribed lines of WO2022251372 the English PP-OCRv4 model
misses 5 characters and the upstream PP-OCRv6 model misses 29, and
they miss different ones: PP-OCRv6 reads `O` as `0`, while PP-OCRv4
drops brackets.

The vote routes therefore read every printed line **twice** and combine
the two readings with one rule:

> **Take the combination that RDKit can parse, and among those the one
> that replaces the fewest printed lines of the primary reading.**

The rule never looks at a reference transcription, so it works on a
patent nobody has transcribed. The primary reading is the first
candidate and is kept whenever it already parses, and at most three
lines may be changed, so the rule cannot make a working reading worse.

End to end on WO2022251372, over the real pipeline:

| route | transcribed exact | CER | all 28 parse | all 28 balanced |
|---|---:|---:|---:|---:|
| `padded` | 6 / 6 | 0.000 % | 24 / 28 | 24 / 28 |
| `padded_vote` | 6 / 6 | 0.000 % | **27 / 28** | **27 / 28** |

All six hand transcriptions are character exact.  Three of the six
transcriptions were themselves wrong when first read from the page --
the patent prints its chiral marker as a circled italic "a", which is
easy to mis-read as one or three `@` signs -- and the corrections are
recorded with their pixel evidence in
`experiments/wo2022251372_references.py`.  Measured against the
original transcriptions the same readings carry 4 edits
(0.320 % CER); measured against the corrected ones they carry 0.

On WO2025162428 the vote changes nothing on the three checked
compounds, because the `validated` route already returns a parseable
string there and the vote keeps the primary reading. Over the whole
corpus it does have work to do: on the
335 cells the pipeline itself cuts, 169 parse with `validated` and 171
with `validated_vote`. The vote changes 12 strings -- one printed line
each -- and breaks none of the readings that parsed before. Eight of
the twelve are the same repair, the letter `O` read as the digit `0`,
so `C(=O)` became `C(=0)`, which is not a SMILES at all; the repair
chain mends that pattern anyway, so only four of the twelve reach the
deliverable.

Two guards keep the vote from changing chemistry while it fixes
syntax. The characters must be SMILES characters: RDKit reads
`"CC O"` as `CC` and accepts `"CC*O"`, and a printed column has
neither a space nor a wildcard atom. And the bond order may not
shrink: `C1=CC=CC=C1` and `C1-CC-C-C-C1` both parse and they are
different molecules, so a combination that declares fewer double
bonds than the primary reading is refused. Measured on the 335 cells
the pipeline cuts, a vote *without* that guard changes 17 strings and
loses a bond in 5 of them -- compound 2160 loses four bonds at once.
With the guard the vote changes 12 strings and **no** cell loses one.

The vote never looks at a reference transcription, so it works on a
patent nobody has transcribed. It is a **text-level** decision: it
says which of two readings of the same ink is a well-formed SMILES.
It does not say that the result is the compound the patent drew.

### Why the validated route extends the column

The predefined column fractions stop a few pixels *before* the
vertical rule that closes the table. On patent page I100298 the
printed SMILES of compound 1040 reaches page x = 1930 while the
column boundary sits at x = 1925, so the final glyph is sliced
through the middle.

A half-cut glyph is not a small error. The CTC recogniser returns
garbage for the *entire* row when the trailing glyph is incomplete:
measured on I100298, compounds 1040 and 1041 both went from edit
distance 0 to 29. Extending the SMILES cell to the table's
right-hand rule restores the complete glyph and both compounds go
back to edit distance 0.

The fix lives in `RapidOcrTableExtractor._widen_text_column_to_border`
and only applies to the `validated` route; the `production` route
keeps its historical crop so the shipped CSV stays reproducible.

### What the two routes achieve

On the four compounds of patent page I100298, against a
researcher-supplied reference for 1038 (held out of the
configuration search) and phase-3 transcriptions for 1039-1041:

| cmpd | `production` edit | `validated` edit |
|---|---|---|
| 1038 | 35 | 1 |
| 1039 | 33 | 1 |
| 1040 | 39 | 0 |
| 1041 | 37 | 0 |

These are **text-level** results. A string that is read correctly is
not yet a verified chemical structure, and the four levels --
text recognised, RDKit parseable, mass gate passed, structure
verified against the drawing -- are kept strictly separate
throughout this project.

### Over the whole corpus

Across the 335 printed-SMILES cells the pipeline itself cuts (the
pages that carry a printed-SMILES column), the route gives:

| metric | `validated` | `validated_vote` (shipped) |
|---|---:|---:|
| bracket-balanced | 210 | **211** |
| RDKit-parseable | 169 | **171** |
| cells with characters outside the SMILES alphabet | 8 | **8** |

Bracket balance and row completeness are **text-level proxies, not
accuracy**: they say the string looks like a whole SMILES, not that
the structure is right.

The three hand-verified references are **character exact** in the
delivered CSV -- 1036, 1037 and 1038 all at edit distance 0.  See
`experiments/phase9_reference_check.py`.

### Why the recogniser matters more than the cropping

A controlled comparison holds the crop geometry completely fixed --
same ink bands, same table rules trimmed, same padding, same 2x
upscale, same recognition-only mode -- and swaps only the recogniser:

| | English PP-OCRv4 mobile (`production`) | default PP-OCRv6 small (`validated`) |
|---|---:|---:|
| bracket-balanced (of 336) | **2** | **228** |
| cells with illegal characters | 148 | 62 |
| total left brackets | 10434 | 12815 |
| total right brackets | 12863 | 12881 |

The English recogniser drops **2429 left brackets** across the corpus.
That is the direct cause of the 335/335 bracket imbalance in the
shipped CSV, and it is independent of how the cells are cropped.

### A note on upscaling

Upscaling is not a resolution operation. The recogniser is
configured with `rec_img_shape: [3, 48, 320]` and always normalises
the strip to **48 px tall**, so it rescales whatever height it is
given. The route's ink band is padded to 54-62 px, already above 48,
so the recogniser always shrinks it: at scale 1 that is one
interpolation, at scale 2 we double the strip first and the
recogniser shrinks it again -- two interpolations, the second one on
its own sampling grid, which is where the bars of `=` and the ring of
`@` get merged. Upscaling therefore never adds information, it only
adds a resampling step. Measured with only the scale factor
changing: scale 1 and scale 2 are identical on 1036 and 1037, scale 3
is worse on both, and 2173 is the one sample where upscaling helps.

### An external check: compound 2173

Compound 2173 sits on page I100376, which was never used for tuning,
and the pipeline had never managed to read it -- the shipped CSV
carries a bracket-less string and an empty `smiles_best`. Re-reading
the printed text from the golden table and comparing:

| source | length | edit vs reference | brackets |
|---|---:|---:|---|
| **validated route** | 282 | **1** | 38/38 |
| production route | 250 | 35 | 14/32 |
| shipped CSV `smiles` | 229 | 33 | 16/23 |
| DECIMER (from the drawing) | 237 | 18 | 34/34 |

The single error is a `C(=O)` read as `C(=0)`. The reference is a
transcription awaiting a human check, like the 1039-1041 references.
See `data/audit/phase3/external/EXTERNAL_VALIDATION.md`.

## A second patent: WO2022251372

WO2022251372 arrives as full-page TIFF scans rather than cropped
tables, its tables have eight columns where the first patent has
three to seven, its compound numbers are one or two digits instead of
four, and every activity value is a letter grade rather than a number.
All of that is configuration, so all of it lives in the profile
`WO2022251372` in `src/cpd/parsers/profiles.py`:

| | WO2025162428 | WO2022251372 |
|---|---|---|
| input | 103 cropped JPEG tables | 20 full-page TIFF scans |
| page handling | the image *is* the table | find the ruled table in the page (`cpd.parsers.page_tables`) |
| layouts | 3-column SMILES, 6-column dense, 7-column activity | 8-column activity, 3-column SMILES |
| compound id | four digits | one or two digits, top-left of the row band |
| activity values | numbers | letter grades A-D, read through the closed-set reader (`cpd.ocr.constrained`) |
| printed SMILES route | `validated_vote` | `padded_vote` |
| row continuation | no | yes -- a row may run over the page break |

Measured on the 28 compounds of the patent, against six hand
transcriptions: the printed text comes back with **0 character
errors** over 1252 reference characters (0.000 % CER), and all 6
transcribed compounds are character-exact and parse with RDKit. Across
all 28 compounds, 27 strings parse with the `padded_vote` route the
profile selects -- 24 with the single recogniser.

Those six transcriptions were themselves corrected during this work:
the patent prints its chiral marker as a *circled italic "a"*, which
is exactly the glyph that makes `[C@@H]` easy to mis-read as `[C@H]`
or `[C@@@H]`. Three of the six were wrong in the ground truth, and
the corrections are recorded with their pixel evidence in
`experiments/wo2022251372_references.py`. Measured against the
*original* transcriptions the same readings carry 4 edits
(0.320 % CER) and 4 of the 6 are exact.

The one compound that still does not parse fails on a stereo
confusion (`@@` read as `@`), which a parse check cannot see.

The two patents need *opposite* settings, and that is measured rather
than assumed: the `padded` route fixes the last glyph of every line on
WO2022251372 and makes the first patent much worse, because there the
rows are packed tightly enough that the same margin pulls in fragments
of the neighbouring lines. Hence two routes rather than one parameter.

The letter grades are read by a *closed-set* reader: the glyph is
located by ink analysis instead of by the text detector, and it is
accepted only if it is one of A/B/C/D. A misread glyph therefore
leaves the cell empty instead of writing a wrong value -- with the
one exception that an A read as a B (or B as a C) cannot be detected
from inside the allowed set.

See `data/audit/phase6/SUMMARY.md` for the full measurement and
`data/audit/phase6/accuracy_vs_reference.csv` for the per-compound
comparison.

The second patent now has a deliverable of its own:
`data/processed/WO2022251372/compounds_combined.csv`, produced by
`experiments/phase10_patent2_delivery.py` from the same pipeline code
that serves the first patent.  Its six letter-grade columns are
carried by the same `AssayRecord` mechanism the five numeric fields
use, so nothing had to be special-cased for the grade values.

## What is shelved, and why

**Reading a 2D chemical structure image.** DECIMER recognises the
cropped macrocycle drawings, and for these compounds it is not
accurate enough to be trusted: its output routinely loses ring
closures or breaks the backbone into fragments, and the repair
heuristics that were written to mend it invent atoms and bonds. The
stage is kept in the code and its output is kept in the CSV, because
it is a second opinion and because it is cheap to re-run from the
on-disk cache -- but the *printed* SMILES is now the primary source
of the structure, and nothing in the deliverable is derived from
DECIMER unless the printed text is missing.

**The macrocycle repair heuristics.** They are still measured and
still reported, but they no longer promote themselves into
`smiles_best`; the printed SMILES wins whenever it parses. The
audit that motivated this is in `PROGRESS.md`.

**The mass-constrained OCR search.** It proposes strings that
reproduce the measured mass. A proposal is not a reading, so it
writes its own columns and never touches `smiles_best`. It is run
on request (`--ocr-search`).

## The two repositories

`cyclic-peptide` is the working repository: the code, every
experiment, every audit artefact and the research log.

`drug_ocr` is the delivery repository: the same code, the patent
inputs, the deliverable CSV and the evidence documents -- and nothing
else.

`tools\sync_delivery.py` copies the curated set from `cyclic-peptide`
into `drug_ocr` and verifies every file by SHA256, so "the delivery
repo matches the working repo" is a checkable statement:

```powershell
# report what would change (the default is a dry run)
.venv\Scripts\python.exe tools\sync_delivery.py

# copy, and drop everything that is not part of the curated set
.venv\Scripts\python.exe tools\sync_delivery.py --apply --prune
```

The curated set includes the scripts that reproduce the headline
numbers (`experiments/rebuild_csv.py`, the phase-6 and phase-7 audit
scripts, and `analysis/verify_deliverable.py`), so a reviewer can
re-run the audit rather than take it on trust.

`analysis/verify_deliverable.py` now summarises **every** configured
patent, not just the first: it finds the patents through the profile
registry, reads each one's own CSV and prints its rows, columns,
route, activity-field coverage, printed-SMILES quality and the
distribution of the audit columns.

## Open Research: the repair strategies do not survive auditing

Once the chemistry and mass gates are applied, **0 of the 119 candidate
repairs are validated**; the honest intact-macrocycle figure is 178 / 406
(43.8 %), not the 86 % previously quoted. Most candidate repairs come from
`terminal_bridge`, which fabricates a bond between two distant atoms --
it closes *a* ring, which is exactly why a ring-only test waved it through.

See `PROGRESS.md` for the full audit, the LC-MS evidence, and the
recommended next steps. The two queued fixes are (a) multi-pass
DECIMER with different seeds and consensus voting, and (b) cross-
validation with the Hiro-OCSR model already cloned under
`references/Hiro-OCSR/`. See `PROGRESS.md` for the detailed status.

## References

`references/RapidOCR/`, `references/DECIMER-Image_Transformer/`, and
`references/Hiro-OCSR/` are local read-only clones used while
developing the OCSR / OCR wrappers in `src/cpd/`. They are not
modified by the pipeline.
