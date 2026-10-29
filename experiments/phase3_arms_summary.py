r"""One comparison table across every Phase 3 arm that was measured.

Why
---
The Phase 3 experiments live in separate output directories, each with
its own arm names and its own screen. That makes them impossible to
compare at a glance, which is exactly the comparison that decides
whether the production path should change.

This script reads the arm-level CSVs and puts them in one table with
one set of columns:

    n          cells scored
    balanced   parentheses balance over the joined string
    rowok      the printed row count was recovered
    strict     balanced AND rowok AND length in 220-300
    parse      RDKit accepts the joined string
    edit1036   character errors against the human-verified 1036 text
               (only where the arm was also scored on 1036)

The strict column is the honest headline number: a cell that lost a
printed row can still have balanced parentheses, so balance alone is
not enough.

All of this is TEXT level. It says nothing about whether the chemical
structure is correct.

Read-only. Writes to ``data/audit/phase3/arms_summary/``.

Run:
    .venv\Scripts\python.exe phase3_arms_summary.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_quality_screen import SMILES_CHARS  # noqa: E402

ROOT = REPO / "data" / "audit" / "phase3"
OUT = ROOT / "arms_summary"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")

LEN_MIN, LEN_MAX = 220, 300

# arm file -> (experiment label, source of the arm names)
SOURCES = [
    ("cell_engine_scale/cell_engine_partial.csv", "cell crop"),
    ("scale_binarise/scale_binarise.csv", "cell crop"),
    ("page_scale/page_scale.csv", "whole page"),
    ("page_det_line_rec/page_det_line_rec.csv", "page detect + line rec"),
]


def _median(vals):
    if not vals:
        return 0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2


def _read(path: Path):
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _expected_rows() -> dict[tuple[str, str], int]:
    """Printed row count per cell, as measured on the page by the audit.

    The page-scale experiments recorded how many rows the detector
    returned but not how many the page actually prints. Joining the
    manifest back in is what turns "balanced" into a claim that can be
    trusted: a cell that silently lost a printed row can still have
    balanced parentheses.
    """
    out: dict[tuple[str, str], int] = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no" or r["extraction_status"] != "ok":
                continue
            out[(r["page"], r["cmpd_id"])] = int(r["n_lines"])
    return out


def _row_ok(row: dict, expected: dict[tuple[str, str], int]) -> bool:
    """Did the arm recover the number of printed rows on the page?"""
    if "rows_match" in row and row["rows_match"] != "":
        return _truthy(row["rows_match"])
    exp = expected.get((row.get("page", ""), row.get("cmpd_id", "")))
    if exp is None or row.get("n_rows", "") == "":
        return True
    return int(row["n_rows"]) == exp


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref1036 = REFERENCES["1036"]["smiles"]
    expected = _expected_rows()

    table: list[dict] = []
    for rel, label in SOURCES:
        rows = _read(ROOT / rel)
        if not rows:
            continue
        arms = sorted({r.get("arm", "") for r in rows})
        for arm in arms:
            sub = [r for r in rows if r.get("arm", "") == arm]
            n = len(sub)
            bal = sum(1 for r in sub if _truthy(r.get("balanced")))
            rowok = sum(1 for r in sub
                        if _row_ok(r, expected))
            par = sum(1 for r in sub if _truthy(r.get("parses")))
            lens = [int(r["len"]) for r in sub if r.get("len")]
            strict = sum(
                1 for r in sub
                if _truthy(r.get("balanced"))
                and _row_ok(r, expected)
                and LEN_MIN <= int(r["len"]) <= LEN_MAX)
            edit = ""
            stray = sum(1 for r in sub
                        if r.get("text", "")
                        and any(c not in SMILES_CHARS for c in r["text"]))
            for r in sub:
                if r.get("cmpd_id") == "1036":
                    txt = r.get("text", "")
                    edit = edit_ops(ref1036, txt) if txt else ""
                    break
            table.append({
                "experiment": label,
                "arm": arm,
                "n": n,
                "balanced": bal,
                "rows_recovered": rowok,
                "strict": strict,
                "parses": par,
                "median_len": _median(lens),
                "edit_1036": edit,
                "stray_cells": stray,
            })

    order = {"cell crop": 0, "whole page": 1, "page detect + line rec": 2}
    table.sort(key=lambda t: (order.get(t["experiment"], 9), -t["strict"]))

    lines: list[str] = []
    lines.append("=== all Phase 3 arms, one table ===")
    lines.append("")
    lines.append(f"{'experiment':<24} {'arm':<26} {'n':>4} {'bal':>5} "
                 f"{'rowok':>6} {'strict':>7} {'parse':>6} {'medlen':>7} "
                 f"{'stray':>6} {'edit1036':>9}")
    for t in table:
        lines.append(
            f"{t['experiment']:<24} {t['arm']:<26} {t['n']:>4} "
            f"{t['balanced']:>5} {t['rows_recovered']:>6} "
            f"{t['strict']:>7} {t['parses']:>6} {t['median_len']:>7} "
            f"{t['stray_cells']:>6} {str(t['edit_1036']):>9}")
    lines.append("")
    lines.append("strict  = balanced AND rows recovered AND length 220-300")
    lines.append("edit1036 = character errors against the human-verified "
                 "1036 text (L3, text only)")
    lines.append("stray = cells containing a character that cannot occur in "
                 "a SMILES string")
    lines.append("This table is text level. No structure is verified.")

    with (OUT / "arms_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    io.open(OUT / "arms_summary.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
