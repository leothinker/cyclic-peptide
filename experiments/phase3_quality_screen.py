r"""Reference-free quality screen for printed-SMILES OCR output.

Why
---
Only two compounds in the whole patent have a human-verified reference
(1036, 1037). That is the bottleneck for evaluating ANY OCR change at
scale: a change can only be scored on n=2, which is how the preprocessing
lead from 2026-10-23 stayed ambiguous for a week.

This module supplies the missing instrument. It scores a set of printed
SMILES rows using only signals that need no gold standard, so it can be
applied to all 335 text cells.

Signals
-------
    row_len_ok       every printed row is long enough to be a SMILES row
    charset_ok       every character is legal in a SMILES string
    joined_balanced  parentheses and brackets balance over the JOINED string
    no_empty_branch  no '()' pair, which is not a legal SMILES branch
    joined_parses    RDKit accepts the joined string
    n_rows_plausible  row count matches the patent's printed layout prior

The balance check deliberately runs on the JOINED string, not per row.
Printed rows are split at arbitrary points, so a legal row routinely starts
with ')' or ends with '(' -- 1036 row 5 is '=O)N(C)CC(=O)N...' and row 7
ends with 'C('. Per-row balance would reject correct output.

Validation
----------
``validate_controls()`` scores two known cases:

    1036 raw-default   edit 0   -> must pass
    1037 raw-default   edit 38  -> must fail

The screen is only usable for ranking configurations if it accepts the
known-good case and rejects the known-bad one, so this runs as part of the
module rather than being left to a separate check.

Read-only. No pipeline, core module or official CSV is touched.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Characters that can occur in a printed SMILES string. Backslash covers the
# escaped bond notation used in some transcriptions ('C(=O)' is written
# '\(C(=O\)' in one of the patent's own reference transcriptions).
SMILES_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "@+-[]()=#$%/\\."
)

# Printed SMILES rows observed in this patent run 32-42 characters. The
# bounds below are deliberately wider than that so a legitimate unusual row
# is not rejected, but tight enough to catch a row that collapsed.
ROW_LEN_MIN = 20
ROW_LEN_MAX = 50

# Row-count prior measured over the 335 audited cells: 291 cells print 7
# rows, 38 print 8, a handful print 10 or 11.
EXPECTED_ROWS = {7, 8, 10, 11}


@dataclass
class ScreenResult:
    cmpd_id: str
    n_rows: int
    joined_len: int
    row_len_min: int
    row_len_max: int
    row_len_cv: float
    row_len_ok: bool
    charset_ok: bool
    joined_balanced: bool
    paren_excess: int
    bracket_excess: int
    no_empty_branch: bool
    joined_parses: bool
    n_rows_plausible: bool
    verdict: str
    reasons: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        d = self.__dict__.copy()
        d["reasons"] = ";".join(self.reasons)
        return d


def screen(cmpd_id: str, rows: list[str]) -> ScreenResult:
    """Score one compound's printed SMILES rows without a gold standard."""
    rows = [r for r in rows if r is not None]
    joined = "".join(rows)
    lens = [len(r) for r in rows]
    lo = min(lens) if lens else 0
    hi = max(lens) if lens else 0
    mean = sum(lens) / len(lens) if lens else 0.0
    var = sum((x - mean) ** 2 for x in lens) / len(lens) if lens else 0.0
    cv = math.sqrt(var) / mean if mean else 0.0

    row_len_ok = bool(lens) and all(ROW_LEN_MIN <= x <= ROW_LEN_MAX
                                    for x in lens)
    charset_ok = all(c in SMILES_CHARS for c in joined)
    joined_balanced = (joined.count("(") == joined.count(")")
                       and joined.count("[") == joined.count("]"))
    # Signed surplus. A positive paren_excess means closing brackets appear
    # without their opener -- the signature of a row whose LEADING bracket
    # was clipped away, because printed rows are split mid-string and a row
    # often begins with the closer that belongs to the previous row.
    paren_excess = joined.count(")") - joined.count("(")
    bracket_excess = joined.count("]") - joined.count("[")
    no_empty_branch = "()" not in joined
    joined_parses = False
    if joined:
        try:
            joined_parses = Chem.MolFromSmiles(joined) is not None
        except Exception:
            joined_parses = False
    n_rows_plausible = len(rows) in EXPECTED_ROWS

    reasons = []
    if not row_len_ok:
        bad = [x for x in lens if not (ROW_LEN_MIN <= x <= ROW_LEN_MAX)]
        reasons.append(f"row_len_out_of_range{bad}")
    if not charset_ok:
        badc = sorted({c for c in joined if c not in SMILES_CHARS})
        reasons.append(f"illegal_chars{badc}")
    if not joined_balanced:
        reasons.append(
            f"unbalanced(={joined.count('(')}/{joined.count(')')}"
            f",[={joined.count('[')}/{joined.count(']')})")
    if not no_empty_branch:
        reasons.append("empty_branch")
    if not joined_parses:
        reasons.append("rdkit_parse_fail")
    if not n_rows_plausible:
        reasons.append(f"n_rows={len(rows)}")

    # A row that is structurally impossible is a hard fail. A row count
    # outside the prior is a warning, because the patent does print some
    # cells with a different number of rows.
    hard = [r for r in reasons
            if not r.startswith("n_rows=")]
    if hard:
        verdict = "fail"
    elif reasons:
        verdict = "suspect"
    else:
        verdict = "pass"

    return ScreenResult(
        cmpd_id=cmpd_id, n_rows=len(rows), joined_len=len(joined),
        row_len_min=lo, row_len_max=hi, row_len_cv=round(cv, 4),
        row_len_ok=row_len_ok, charset_ok=charset_ok,
        joined_balanced=joined_balanced,
        paren_excess=paren_excess, bracket_excess=bracket_excess,
        no_empty_branch=no_empty_branch, joined_parses=joined_parses,
        n_rows_plausible=n_rows_plausible, verdict=verdict,
        reasons=reasons,
    )


def screen_joined(cmpd_id: str, text: str) -> ScreenResult:
    """Score one already-joined SMILES string.

    The official CSV stores the concatenated printed SMILES, so the row
    boundaries are gone and the row-level checks cannot be applied. The
    joined-level checks still carry most of the signal: a character set
    violation, an unbalanced bracket, an empty branch or an RDKit parse
    failure are all decidable without knowing where the rows were.
    """
    rows = [text] if text else []
    res = screen(cmpd_id, rows)
    # row_len is meaningless for a joined string
    res.row_len_ok = True
    res.row_len_min = res.row_len_max = len(text)
    res.row_len_cv = 0.0
    reasons = [r for r in res.reasons
               if not r.startswith("row_len_")
               and not r.startswith("n_rows=")]
    res.reasons = reasons
    hard = [r for r in reasons if not r.startswith("n_rows=")]
    if hard:
        res.verdict = "fail"
    elif reasons:
        res.verdict = "suspect"
    else:
        res.verdict = "pass"
    return res


# ---------------------------------------------------------------- controls
CONTROL_ROWS = {
    # 1036 under the raw-page configuration: edit distance 0 against the
    # human-verified reference, so it must pass.
    "1036_raw_default_pass": [
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCC",
        "C2)C(=O)N3[C@@H](CC3)C(=O)N(C)[C@@H]4C(",
        "=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C",
        "5)C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CC",
        "CC7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)[C@",
        "@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)=C",
        "(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C",
    ],
    # 1037 under the same configuration: edit distance 38. One row collapsed
    # into garbage, so the screen must reject it.
    "1037_raw_default_fail": [
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](CC2=CC=",
        "CC)()(=())(=(CH)",
        "4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F",
        ")=C5)C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(",
        "CCCC7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)[",
        "C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F",
        "=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C",
    ],
}


def validate_controls() -> tuple[bool, list[dict]]:
    """The screen must accept the known-good case and reject the known-bad."""
    out = []
    ok = True
    expect = {"1036_raw_default_pass": "pass",
              "1037_raw_default_fail": "fail"}
    for name, rows in CONTROL_ROWS.items():
        res = screen(name, rows)
        good = res.verdict == expect[name]
        ok = ok and good
        out.append({"control": name, "expected": expect[name],
                    "got": res.verdict, "as_expected": good,
                    "reasons": ";".join(res.reasons)})
    return ok, out


if __name__ == "__main__":
    ok, rows = validate_controls()
    for r in rows:
        print(f"{r['control']:<26} expected={r['expected']:<7} "
              f"got={r['got']:<7} {'OK' if r['as_expected'] else 'MISMATCH'}"
              f"  {r['reasons']}")
    print(f"\ncontrols {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
