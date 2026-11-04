"""The mass-constrained OCR repair search.

What it is
----------
When the recogniser loses a bracket, the printed SMILES no longer
parses.  :mod:`cpd.mass_constrained_search` searches for the smallest
edit of the recognised text that makes the molecule parse *and*
reproduces the mass the patent measured.

Why it is not in the default pipeline
-------------------------------------
The search is a *proposal*, not a reading.  Its answer is chosen by
the mass gate, so a search that finds a string which reproduces the
m/z is not evidence that the string is the one that was printed -- it
is evidence that a plausible string with the right mass exists.  The
earlier audit of this project showed exactly that trap: a heuristic
that bolts a bond between two distant atoms always closes *some* ring,
so a macrocycle-shaped answer is not a correct answer.

The search therefore writes its own columns and never touches
``smiles_best``.  It is run on request (``--ocr-search``) and its
output has to be reviewed.

Column contract
---------------
``smiles_ocr_repaired``     the search's best SMILES, or empty
``ocr_repair_trace``        one-line summary: ``ok ...`` / ``no_answer`` /
                            ``failed_...`` / ``no_ocr`` / ``no_mass_reference``
``ocr_repair_formula``      formula if it parses
``ocr_repair_largest_ring`` largest ring size if it parses
``ocr_repair_mass_ppm``     signed ppm error against the patent's window
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from cpd.mass_constrained_search import (
    bracket_deficit,
    mass_window,
    repair_from_ocr,
)

PER_ROW_SECONDS = 5.0
PER_ROW_STATES = 500_000

#: Above this many lost brackets the search is not run: the space is
#: too large for the budget and the answer would be a guess.
SEARCHABLE_MAX_DEFICIT = 10

SEARCH_COLUMNS: tuple[str, ...] = (
    "smiles_ocr_repaired",
    "ocr_repair_trace",
    "ocr_repair_formula",
    "ocr_repair_largest_ring",
    "ocr_repair_mass_ppm",
)


def _num(value: object) -> Optional[float]:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _ppm(actual: float, expected: float) -> Optional[float]:
    if not expected:
        return None
    return (actual - expected) / expected * 1e6


def search_row(row: dict) -> dict:
    """Run the search on one row and return the search columns."""
    empty = {column: "" for column in SEARCH_COLUMNS}
    smiles = (row.get("smiles") or "").strip()
    if not smiles:
        return empty | {"ocr_repair_trace": "no_ocr"}

    deficit = bracket_deficit(smiles)
    total = deficit["total"]
    if total > SEARCHABLE_MAX_DEFICIT:
        return empty | {
            "ocr_repair_trace": f"unsearchable_deficit_{total}",
        }

    mz = _num(row.get("ms_mz"))
    polarity = (row.get("ms_polarity") or "").strip()
    if mz is None or not polarity:
        return empty | {"ocr_repair_trace": "no_mass_reference"}
    try:
        low, high = mass_window(mz, polarity)
    except (ValueError, KeyError):
        return empty | {"ocr_repair_trace": "bad_polarity"}
    target = (low + high) / 2.0

    started = time.time()
    result = repair_from_ocr(
        smiles,
        mz=mz,
        polarity=polarity,
        max_states=PER_ROW_STATES,
        max_seconds=PER_ROW_SECONDS,
    )
    elapsed = time.time() - started

    if not result or result.get("smiles") is None:
        trace = (
            result.get("trace", "no_answer")
            if isinstance(result, dict)
            else "no_answer"
        )
        return empty | {"ocr_repair_trace": trace}

    mass = result.get("mass")
    mass_ppm = _ppm(mass, target) if mass is not None else None
    return {
        "smiles_ocr_repaired": result["smiles"],
        "ocr_repair_trace": (
            f"ok states={result.get('states', 0)} "
            f"in={result.get('n_inserts', 0)} "
            f"subs={result.get('n_subs', 0)} "
            f"dels={result.get('n_dels', 0)} "
            f"t={elapsed:.1f}s"
        ),
        "ocr_repair_formula": result.get("formula", ""),
        "ocr_repair_largest_ring": str(result.get("largest_ring", "")),
        "ocr_repair_mass_ppm": f"{mass_ppm:.1f}" if mass_ppm is not None else "",
    }


def run_search(
    rows: Iterable[dict],
    progress: bool = True,
) -> list[dict]:
    """Run the search over every row and return the rows with the
    search columns filled in."""
    rows = list(rows)
    total = len(rows)
    n_ok = 0
    started = time.time()
    out: list[dict] = []
    for index, row in enumerate(rows, start=1):
        new = dict(row)
        for column in SEARCH_COLUMNS:
            new.pop(column, None)
        new.update(search_row(row))
        if (new.get("ocr_repair_trace") or "").startswith("ok "):
            n_ok += 1
        out.append(new)
        if progress and (index % 25 == 0 or index == total):
            elapsed = time.time() - started
            eta = elapsed * (total - index) / max(index, 1)
            print(
                f"      [{index:>4}/{total}] ok={n_ok} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    return out


__all__ = [
    "PER_ROW_SECONDS",
    "PER_ROW_STATES",
    "SEARCH_COLUMNS",
    "SEARCHABLE_MAX_DEFICIT",
    "run_search",
    "search_row",
]

