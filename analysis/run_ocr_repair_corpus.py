"""Run the mass-constrained OCR-repair search across the corpus.

For each row in compounds_combined.csv that has an OCR SMILES and a
usable m/z, run :func:`cpd.mass_constrained_search.repair_from_ocr`
with a tight per-row budget, and record the result in five new
columns independent of the existing best-SMILES pipeline:

* ``smiles_ocr_repaired``     -- the search's best SMILES, or empty
* ``ocr_repair_trace``        -- one-line summary: ok / no_answer / failed_X
* ``ocr_repair_formula``      -- formula if it parses
* ``ocr_repair_largest_ring`` -- largest ring size if it parses
* ``ocr_repair_mass_ppm``     -- signed ppm error vs the patent's window

Run from the repo root::

    .venv/Scripts/python.exe analysis/run_ocr_repair_corpus.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpd.mass_constrained_search import (bracket_deficit,
                                              mass_window,
                                              repair_from_ocr)
from cpd.ocr_repair_costs import expansion_cost  # noqa: F401  (warm import)

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

PER_ROW_SECONDS = 5.0
PER_ROW_STATES = 500_000

OUT_COLS = [
    "smiles_ocr_repaired",
    "ocr_repair_trace",
    "ocr_repair_formula",
    "ocr_repair_largest_ring",
    "ocr_repair_mass_ppm",
    "ocr_bracket_deficit",
    "ocr_difficulty",
]


def _ppm(actual: float, expected: float) -> float:
    if expected == 0:
        return float("nan")
    return (actual - expected) / expected * 1e6


def repair_row(row: dict) -> dict:
    """Run the search on a single row, return the new fields."""
    empty = {c: "" for c in OUT_COLS}
    smiles = (row.get("smiles") or "").strip()
    if not smiles:
        return {c: "" for c in OUT_COLS} | {"ocr_repair_trace": "no_ocr",
                                              "ocr_difficulty": "no_ocr"}
    deficit = bracket_deficit(smiles)
    total_d = deficit["total"]
    if total_d == 0:
        diff = "balanced"
    elif total_d <= 2:
        diff = "light"
    elif total_d <= 5:
        diff = "medium"
    elif total_d <= 10:
        diff = "heavy"
    else:
        diff = "extreme"

    base = {"ocr_bracket_deficit": str(total_d), "ocr_difficulty": diff}
    SEARCHABLE_MAX_DEFICIT = 10
    if total_d > SEARCHABLE_MAX_DEFICIT:
        return base | empty | {
            "ocr_repair_trace": f"unsearchable_deficit_{total_d}",
        }
    mz_raw = (row.get("ms_mz") or "").strip()
    pol = (row.get("ms_polarity") or "").strip()
    if not mz_raw or not pol:
        return empty | {"ocr_repair_trace": "no_mass_reference"}
    try:
        mz = float(mz_raw.replace(",", ""))
    except ValueError:
        return empty | {"ocr_repair_trace": "bad_mz"}
    try:
        lo, hi = mass_window(mz, pol)
    except (ValueError, KeyError):
        return empty | {"ocr_repair_trace": "bad_polarity"}
    target = (lo + hi) / 2.0

    t0 = time.time()
    result = repair_from_ocr(
        smiles,
        mz=mz,
        polarity=pol,
        max_states=PER_ROW_STATES,
        max_seconds=PER_ROW_SECONDS,
    )
    elapsed = time.time() - t0

    if result is None or result.get("smiles") is None:
        trace = (result.get("trace", "no_answer")
                 if isinstance(result, dict) else "no_answer")
        return base | empty | {"ocr_repair_trace": trace}

    mass = result.get("mass")
    mass_ppm = _ppm(mass, target) if mass is not None else None
    return base | {
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


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())
    for c in OUT_COLS:
        if c not in fieldnames:
            fieldnames.append(c)

    n_total = len(rows)
    n_with_ocr = sum(1 for r in rows if (r.get("smiles") or "").strip())
    n_empty = 0
    n_ok = 0
    t_start = time.time()

    for i, row in enumerate(rows, 1):
        # clear any existing repair columns before re-running
        for c in OUT_COLS:
            row.pop(c, None)

        if not (row.get("smiles") or "").strip():
            row["ocr_repair_trace"] = "no_ocr"
            n_empty += 1
            continue

        row.update(repair_row(row))
        if (row.get("ocr_repair_trace") or "").startswith("ok "):
            n_ok += 1

        if i % 25 == 0 or i == n_total:
            elapsed = time.time() - t_start
            eta = elapsed * (n_total - i) / max(i, 1)
            print(f"[{i:>4}/{n_total}] ok={n_ok} empty={n_empty} "
                  f"elapsed={elapsed:.1f}s eta={eta:.0f}s",
                  flush=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {n_total} rows; {n_ok}/{n_with_ocr} produced an "
          f"OCR-repaired SMILES.", flush=True)
    print(f"Total elapsed: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
