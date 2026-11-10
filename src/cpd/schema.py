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

from collections.abc import Iterable

from cpd.postprocess import DERIVED_COLUMNS as _DERIVED_COLUMNS
from cpd.parsers.profiles import CANONICAL_ASSAY_FIELDS

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

def combined_columns(extra_assay_fields: Iterable[str] = ()) -> tuple[str, ...]:
    """The combined-CSV columns for one patent.

    The canonical block always comes first and in the shipped order.
    Patent-specific activity columns (the letter grades of
    WO2022251372, for example) are appended after it, so the first
    patent -- which declares only the five canonical fields -- keeps
    exactly the shipped 48 columns and a byte-identical CSV.

    The order of the extra columns follows the patent's profile, so
    the header of a re-run is predictable.
    """
    seen: set[str] = set()
    extra: list[str] = []
    for name in extra_assay_fields:
        if name in COMBINED_COLUMNS or name in CANONICAL_ASSAY_FIELDS:
            continue
        if name in seen:
            continue
        seen.add(name)
        extra.append(name)
    return COMBINED_COLUMNS + tuple(extra)


__all__ = [
    "CANONICAL_ASSAY_FIELDS",
    "COMBINED_COLUMNS",
    "DERIVED_COLUMNS",
    "PIPELINE_COLUMNS",
    "REPAIR_COLUMNS",
    "SEARCH_COLUMNS",
    "combined_columns",
]

