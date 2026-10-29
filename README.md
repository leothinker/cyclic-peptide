# Cyclic Peptide Dataset Pipeline

Extract cyclic-peptide SMILES and KRAS G12V GDP KD bioactivity data from
patent characterization tables (e.g. WO2025162428). One combined CSV is the
main deliverable: **406 cmpds x 48 columns** -- SMILES, DECIMER 2D re-render,
LCMS RT / MS m/z / method / polarity, KRAS G12V GDP KD, OCR trust scores,
macrocycle-intact diagnostic, and a post-hoc repair stage that is audited
against both chemical invariants and the patent's own LC-MS mass before
anything is promoted. See `PROGRESS.md` for the current findings.

The printed-SMILES column can be read by two interchangeable OCR routes
(see [Reading the printed SMILES](#reading-the-printed-smiles)):
`production` (the historical route, which produced the shipped CSV) and
`validated` (a route that reaches character-level accuracy on the patent
pages measured so far). They are selected with `--smiles-route`.

## Quick Start

```powershell
# 1. Install dependencies (one-time)
uv sync

# 2. Optional: install DECIMER for OCSR SMILES (pulls PyTorch ~2 GB).
#    Skip it if you only need the OCR SMILES column.
uv pip install --python .venv/Scripts/python.exe DECIMER

# 3. Run the pipeline (reads data/raw/WO2025162428/golden_tables by default)
$env:PYTHONIOENCODING = "utf-8"
uv run python main.py

# 4. Re-use cached OCR + DECIMER results (fast; only re-renders)
uv run python main.py --skip-ocr --skip-decimer

# 5. Read the printed SMILES with the validated OCR route
uv run python main.py --smiles-route validated

# 6. Run the test suite
uv run pytest
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
 |   +- chem.py                   # RDKit validation, repair, descriptors
 |   +- decimer.py                # Optional DECIMER (OCSR) wrapper
 |   +- merge.py                  # cmpd_id normalisation + row dedupe
 |   +- render.py                 # 2D structure render + macrocycle-intact gate
 |   +- macrocycle_repair.py      # Six post-hoc repair heuristics
 |   +- repair_audit.py           # Chemical invariants + confidence tiering
 |   +- ms_validation.py          # Check a SMILES against the reported m/z
 |   +- repair_policy.py          # Gated promotion into the deliverable CSV
|   +- mass_constrained_search.py# Atom-mass-constrained SMILES search
|   +- ocr_rewrite.py            # Deterministic OCR artefact rewrites
|   +- ocr_edit_model.py         # Edit-cost model for OCR repair
|   +- ocr_mass_gate.py          # Mass gate for OCR-derived candidates
|   +- ocr_repair_costs.py       # Cost tables for OCR repair strategies
|   +- render_polygonal.py       # ChemDraw-style polygonal macrocycle render
|   +- ocr/
|       +- __init__.py
|       +- printed_smiles.py     # The two printed-SMILES OCR routes
 |   +- parsers/
 |       +- __init__.py
 |       +- headers.py            # Header alias map (Cmpd # / Compound # / ...)
 |       +- table_extractor.py    # RapidOCR + OpenCV cell extractor
 |       +- grammar.py            # SMILES prefix grammar for beam pruning
 |       +- beam_search.py        # Top-k OCR beam search over SMILES prefixes
|       +- prefix_validator.py   # Prefix validity check for beam search
 +- tests/
 |   +- test_chem.py
 |   +- test_merge.py
 |   +- test_table_pipeline.py
 |   +- test_combined_pipeline.py
 |   +- test_render.py            # render.macrocycle_intact / audit / draw
 |   +- test_macrocycle_repair.py # digit-swap + terminal-bridge heuristics
 |   +- test_new_repair_strategies.py
 |   +- test_repair_audit.py      # invariants, tiers, LC-MS validation
|   +- test_ocr_repair_search.py
|   +- test_ocr_rewrite.py
|   +- test_prefix_validator.py
+- experiments/                  # One-off analysis scripts (not a package)
+- analysis/                     # Exploratory analysis scripts
 +- docs/
 |   +- archive/                 # Dated snapshots of earlier reports
 |       +- MORNING_BRIEF.md     #   (each carries an archive banner
 |       +- PRESENTATION.md      #    naming the numbers later audited)
 |       +- PROGRESS_REPORT.md
 |       +- REPORT_MATERIAL.md
 +- data/
 |   +- raw/WO2025162428/
 |   |   +- WO2025162428-PAMPH-20250807-0047.pdf
 |   |   +- golden_tables/        # 103 source pages used by the pipeline
 |   +- raw/_external_validation/ # probe logs for the second-patent search
 |   +- processed/                # regenerated by `main.py`; gitignored
 |       +- compounds_combined.csv
 |       +- cells/                # 406 cropped structure PNGs
 |       +- rendered/             # 406 re-rendered PNGs
 |   +- audit/                    # evidence: phase2 / phase3 / demos
 +- references/                   # Read-only upstream OCSR/OCR source clones
 |   +- RapidOCR/
 |   +- DECIMER-Image_Transformer/
 |   +- Hiro-OCSR/
 |   +- README.md
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
structure image. Two routes are implemented in
`src/cpd/ocr/printed_smiles.py` and selected with `--smiles-route`:

| | `production` | `validated` |
|---|---|---|
| recogniser | English PP-OCRv4 mobile | upstream default (PP-OCRv6 rec) |
| line segmentation | OCR detector boxes | our own ink-band projection |
| crop geometry | trim to content, 2x upscale | trim table rules, pad each ink band |
| cell column | fractional boundary | extended to the table's right-hand rule |
| status | produced the shipped CSV | verified on the pages measured so far |

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

Across all 336 cropped printed-SMILES cells (335 data cells plus one
header), the validated route gives:

| metric | `validated` | `production` (shipped CSV) |
|---|---:|---:|
| bracket-balanced | **228** | **0** |
| all printed rows recovered | **336** | -- |
| strict pass (balanced + rows + length 220-300) | 224 | 0 |
| cells with characters outside the SMILES alphabet | 62 | -- |

Bracket balance and row completeness are **text-level proxies, not
accuracy**: they say the string looks like a whole SMILES, not that
the structure is right.

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
