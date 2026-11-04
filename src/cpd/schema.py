"""The column schema of the combined CSV.

The combined CSV is assembled in layers, and every layer owns its
columns:

``pipeline``
    What the extraction and DECIMER stages produce.  Written by
    :mod:`main`.

``repair``
    What :mod:`cpd.repair_policy` adds: the DECIMER-side repair
    candidate, its audit tier and its mass verdict.

``search``
    What :mod:`cpd.mass_constrained_search` adds: the OCR-side
    mass-constrained repair and its trace.

``derived``
    What :mod:`cpd.postprocess` adds: the ring statistics of the
    DECIMER structure, the data-quality flags and the bracket
    deficit of the printed text.

The order of :data:`COMBINED_COLUMNS` is the order of the shipped
file, so a rebuild produces a file that can be compared column by
column with the previous one.
"""

from __future__ import annotations

from cpd.postprocess import DERIVED_COLUMNS as _DERIVED_COLUMNS

PIPELINE_COLUMNS: tuple[str, ...] = (
    "cmpd_id",
    "smiles",
    "canonical_smiles",
    "smiles_decimer",
    "canonical_smiles_decimer",
    "smiles_best",
    "smiles_best_source",
    "smiles_agree",
    "molecular_weight",
    "heavy_atom_count",
    "is_valid",
    "structure_image",
    "source_image",
    "kd_nm",
    "rt_min",
    "ms_mz",
    "lcms_method",
    "ms_polarity",
    "assay_source_image",
    "decimer_min_conf",
    "decimer_low_conf_count",
    "decimer_render_path",
    "ocr_avg_entropy",
    "ocr_min_top1",
    "ocr_low_conf_count",
)

REPAIR_COLUMNS: tuple[str, ...] = (
    "repaired_smiles",
    "is_repaired",
    "repair_method",
    "smiles_candidate_repair",
    "repair_confidence",
    "repair_review_status",
    "structure_confidence",
    "repair_atoms_invented",
    "repair_atoms_deleted",
    "ms_check_verdict",
    "ms_error_ppm",
)

SEARCH_COLUMNS: tuple[str, ...] = (
    "smiles_ocr_repaired",
    "ocr_repair_trace",
    "ocr_repair_formula",
    "ocr_repair_largest_ring",
    "ocr_repair_mass_ppm",
)

DERIVED_COLUMNS: tuple[str, ...] = _DERIVED_COLUMNS

COMBINED_COLUMNS: tuple[str, ...] = (
    PIPELINE_COLUMNS
    + ("data_quality_flags", "data_quality")
    + (
        "decimer_largest_ring_size",
        "decimer_macrocycle_intact",
        "decimer_num_rings",
    )
    + REPAIR_COLUMNS
    + SEARCH_COLUMNS
    + ("ocr_bracket_deficit", "ocr_difficulty")
)

__all__ = [
    "COMBINED_COLUMNS",
    "DERIVED_COLUMNS",
    "PIPELINE_COLUMNS",
    "REPAIR_COLUMNS",
    "SEARCH_COLUMNS",
]

