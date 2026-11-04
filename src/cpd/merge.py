"""Compound-id normalisation and row dedupe (no joins — output is two tables).

Patent authors write the same compound number in different ways on
different pages (``1001``, ``Cmpd 1001``, ``Compound 1001``, ``01001``,
``1038:``...). Before we merge SMILES rows with activity rows in the
caller, we collapse everything to the canonical integer-string form so
the join key matches.

Why pure-stdlib
---------------
The dual-table pipeline keeps pandas out of the writer path; this
module has to work without it.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from cpd.parsers.table_extractor import AssayRecord, CompoundRecord


_PREFIX_RE = re.compile(r"^(?:compound|cmpd\.?|compd\.?|#)\s*#?\s*", re.IGNORECASE)


def normalise_cmpd_id(raw: str | int | None) -> str | None:
    """Return the canonical int-string for any ``cmpd_id`` writing style.

    Empty / unparseable input returns ``None``. Anything that doesn't
    look like a number falls through lowercased — useful for soft joins
    where the caller wants to be lenient.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = _PREFIX_RE.sub("", s)
    s = s.replace(",", "").rstrip(":.) ").strip()
    try:
        return str(int(s))
    except ValueError:
        return s.lower() or None


def dedupe_compounds(rows: Iterable[CompoundRecord]) -> dict[str, CompoundRecord]:
    """Keep the first valid (RDKit-validated) record per cmpd_id.

    Invalid rows are kept only when no valid one exists for the same id.
    """
    seen: dict[str, CompoundRecord] = {}
    fallback: dict[str, CompoundRecord] = {}
    for r in rows:
        cid = normalise_cmpd_id(r.cmpd_id)
        if cid is None or cid in seen:
            continue
        if r.is_valid is True and r.canonical_smiles:
            seen[cid] = r
        else:
            fallback.setdefault(cid, r)
    for cid, r in fallback.items():
        seen.setdefault(cid, r)
    return seen


def merge_compound_records(
    rows: Iterable[CompoundRecord],
    *,
    prefer_structure: bool = False,
) -> dict[str, CompoundRecord]:
    """Deduplicate compounds, optionally combining complementary records.

    Most patents keep SMILES and activity in separate tables, so the
    ordinary first-valid-wins rule is sufficient.  Some patents repeat a
    compound across two tables and put the structure image in only one of
    them.  ``prefer_structure`` keeps the RDKit-valid text but fills a
    missing structure crop from the complementary record.
    """
    grouped: dict[str, list[CompoundRecord]] = {}
    for row in rows:
        cid = normalise_cmpd_id(row.cmpd_id)
        if cid is None:
            continue
        grouped.setdefault(cid, []).append(row)

    result: dict[str, CompoundRecord] = {}
    for cid, records in grouped.items():
        valid = [record for record in records if record.is_valid]
        chosen = valid[0] if valid else records[0]
        if prefer_structure and not chosen.structure_image:
            for candidate in records:
                if candidate.structure_image:
                    chosen = replace(
                        chosen,
                        structure_image=candidate.structure_image,
                    )
                    break
        result[cid] = chosen
    return result


def dedupe_assays(rows: Iterable[AssayRecord]) -> dict[str, AssayRecord]:
    """Collapse duplicate assay rows per cmpd_id (first one wins)."""
    seen: dict[str, AssayRecord] = {}
    for r in rows:
        cid = normalise_cmpd_id(r.cmpd_id)
        if cid is None or cid in seen:
            continue
        seen[cid] = r
    return seen


__all__ = [
    "normalise_cmpd_id",
    "dedupe_compounds",
    "dedupe_assays",
    "merge_compound_records",
]
