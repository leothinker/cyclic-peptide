"""Inner-join ActivityRow + SmilesRow lists into a clean benchmark dataset.

Why this exists
---------------
After :mod:`cpd.parsers.table_finder` identifies which FullText JPGs are
activity / SMILES tables, and :mod:`cpd.parsers.table_extractor` pulls
rows out of them, the only thing left is to merge the two streams by
``Cmpd #`` so we end up with one row per cyclic peptide:

    cmpd_id, canonical_smiles, kd_nm, rt_min, ms_mz, lcms_method, ...

This module is intentionally pure-stdlib: no pandas, no RDKit. The
caller is expected to have already run
:func:`cpd.parsers.table_extractor.enrich_smiles_with_rdkit` so each
SmilesRow carries ``canonical_smiles`` and ``is_valid``.

Key normalisation rules
-----------------------
* Cmpd IDs are stripped of ``Compound `` / ``Cmpd `` / leading zeros and
  compared as integers (``"1001" == "Cmpd 1001" == "compound 01001"``).
* Duplicate SMILES rows for the same cmpd_id collapse to the first one
  whose ``is_valid`` is True (or just the first, if none validate).
* Duplicate activity rows collapse to the first; ``kd_nm`` wins unless
  the row carries a ``>`` / ``<`` / ``~`` relation that we want to keep
  on the side. (No relation parsing yet -- the patent's activity tables
  in WO2025162428 carry clean floats.)
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from cpd.parsers.table_extractor import ActivityRow, SmilesRow


# ---------- normalisation ----------


_PREFIX_RE = re.compile(r"^(?:compound|cmpd\.?|compd\.?|#)\s*#?\s*", re.IGNORECASE)


def normalise_cmpd_id(raw: str | int | None) -> str | None:
    """Strip ``Compound `` / ``Cmpd `` / leading zeros, return canonical int-string.

    Returns ``None`` when the input is empty or unparseable.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = _PREFIX_RE.sub("", s)
    s = s.replace(",", "").strip()
    # Drop trailing punctuation.
    s = s.rstrip(":.)")
    try:
        return str(int(s))
    except ValueError:
        # Fall back to the lowercased trimmed string -- caller may use it
        # as a soft join key.
        return s.lower()


# ---------- dedupe helpers ----------


def dedupe_smiles(rows: Iterable[SmilesRow]) -> dict[str, SmilesRow]:
    """Collapse duplicate SmilesRows per cmpd_id to one row.

    Preference order: first valid (canonical_smiles populated) > first seen.
    """
    seen: dict[str, SmilesRow] = {}
    fallback: dict[str, SmilesRow] = {}
    for r in rows:
        cid = normalise_cmpd_id(r.cmpd_id)
        if cid is None:
            continue
        if cid in seen:
            continue
        if r.is_valid is True and r.canonical_smiles:
            seen[cid] = r
        else:
            fallback.setdefault(cid, r)
    # Fill in any cmpd_id that never produced a valid row.
    for cid, r in fallback.items():
        seen.setdefault(cid, r)
    return seen


def dedupe_activity(rows: Iterable[ActivityRow]) -> dict[str, ActivityRow]:
    """Collapse duplicate ActivityRows per cmpd_id; first wins."""
    seen: dict[str, ActivityRow] = {}
    for r in rows:
        cid = normalise_cmpd_id(r.cmpd_id)
        if cid is None:
            continue
        seen.setdefault(cid, r)
    return seen


# ---------- join ----------


@dataclass
class JoinedRow:
    """One merged SMILES + activity row, ready for CSV / JSONL export."""

    cmpd_id: str
    canonical_smiles: str | None = None
    smiles_raw: str | None = None
    smiles_is_valid: bool | None = None
    kd_nm: float | None = None
    rt_min: float | None = None
    ms_mz: float | None = None
    lcms_method: str | None = None
    ms_polarity: str | None = None
    source_smiles_image: str | None = None
    source_activity_image: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


# Stable CSV column order.
COLUMNS: tuple[str, ...] = (
    "cmpd_id",
    "canonical_smiles",
    "smiles_raw",
    "smiles_is_valid",
    "kd_nm",
    "rt_min",
    "ms_mz",
    "lcms_method",
    "ms_polarity",
    "source_smiles_image",
    "source_activity_image",
)


def join(
    activity_rows: Iterable[ActivityRow],
    smiles_rows: Iterable[SmilesRow],
) -> list[JoinedRow]:
    """Inner join on cmpd_id.

    A cmpd_id is included only if it has BOTH a smiles row and an
    activity row. Use :func:`join_outer` for the lenient variant.
    """
    sm = dedupe_smiles(smiles_rows)
    ac = dedupe_activity(activity_rows)
    out: list[JoinedRow] = []
    for cid in sorted(sm.keys() & ac.keys()):
        s = sm[cid]
        a = ac[cid]
        out.append(JoinedRow(
            cmpd_id=cid,
            canonical_smiles=s.canonical_smiles,
            smiles_raw=s.smiles,
            smiles_is_valid=s.is_valid,
            kd_nm=a.kd_nm,
            rt_min=a.rt_min,
            ms_mz=a.ms_mz,
            lcms_method=a.lcms_method,
            ms_polarity=a.ms_polarity,
            source_smiles_image=s.source_image,
            source_activity_image=a.source_image,
        ))
    return out


def join_outer(
    activity_rows: Iterable[ActivityRow],
    smiles_rows: Iterable[SmilesRow],
) -> list[JoinedRow]:
    """Left-outer join: SMILES rows are the anchor.

    Activity columns stay ``None`` when only the SMILES row exists. Useful
    for diagnosing where the merge falls short (cmpd IDs present in one
    table but not the other).
    """
    sm = dedupe_smiles(smiles_rows)
    ac = dedupe_activity(activity_rows)
    out: list[JoinedRow] = []
    for cid in sorted(sm.keys()):
        s = sm[cid]
        a = ac.get(cid)
        out.append(JoinedRow(
            cmpd_id=cid,
            canonical_smiles=s.canonical_smiles,
            smiles_raw=s.smiles,
            smiles_is_valid=s.is_valid,
            kd_nm=a.kd_nm if a else None,
            rt_min=a.rt_min if a else None,
            ms_mz=a.ms_mz if a else None,
            lcms_method=a.lcms_method if a else None,
            ms_polarity=a.ms_polarity if a else None,
            source_smiles_image=s.source_image,
            source_activity_image=a.source_image if a else None,
        ))
    return out


def coverage_report(
    activity_rows: Iterable[ActivityRow],
    smiles_rows: Iterable[SmilesRow],
) -> dict:
    """Return counts useful for sanity-checking the merge."""
    sm = dedupe_smiles(smiles_rows)
    ac = dedupe_activity(activity_rows)
    only_smiles = sorted(set(sm) - set(ac))
    only_activity = sorted(set(ac) - set(sm))
    both = sorted(set(sm) & set(ac))
    return {
        "n_smiles": len(sm),
        "n_activity": len(ac),
        "n_both": len(both),
        "n_only_smiles": len(only_smiles),
        "n_only_activity": len(only_activity),
        "only_smiles_ids": only_smiles,
        "only_activity_ids": only_activity,
        "joined_ids": both,
    }


# ---------- writers ----------


def to_csv(rows: Iterable[JoinedRow], out_path: Path) -> int:
    """Write joined rows to a CSV file with the canonical column order.

    Returns the number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())
            n += 1
    return n


def to_jsonl(rows: Iterable[JoinedRow], out_path: Path) -> int:
    """Write joined rows to JSON Lines.

    Returns the number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            import json
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


__all__ = [
    "normalise_cmpd_id",
    "dedupe_smiles",
    "dedupe_activity",
    "JoinedRow",
    "COLUMNS",
    "join",
    "join_outer",
    "coverage_report",
    "to_csv",
    "to_jsonl",
]
