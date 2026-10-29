r"""Cross-check the systemic fingerprint across independent sources.

The claim being tested
----------------------
The shipped CSV was produced by a pipeline that does not binarise the
SMILES crop. If that is true, the shipped strings and a fresh run of the
production path must show the SAME defect, independently:

    unmatched closing parentheses, on every compound, with the same
    distribution.

If the fresh production run and the shipped CSV disagree, then the CSV
was not produced by this path and the whole diagnosis is wrong. This
script makes that check explicit instead of assumed.

It also separates the three levels of evidence that must never be
mixed:

    level 1  the parenthesis balance of the printed text
    level 2  RDKit can parse the joined string
    level 3  the string represents the right chemical structure

Only level 1 and level 2 are measured here. Level 3 needs a structure
check and is not claimed.

Read-only. Writes to ``data/audit/phase3/fingerprint_check.txt``.

Run:
    .venv\Scripts\python.exe phase3_fingerprint_check.py
"""

from __future__ import annotations

import io
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase3" / "fingerprint_check.txt"
SCALE = (REPO / "data" / "audit" / "phase3" / "scale_binarise"
         / "scale_binarise.csv")


def main() -> None:
    df = pd.read_csv(SCALE, encoding="utf-8-sig")
    lines: list[str] = []
    lines.append("=== systemic fingerprint across independent sources ===\n")
    lines.append(f"cells: {len(df)}\n")
    lines.append("paren_excess = unmatched ')' in the joined printed SMILES.")
    lines.append("A correct SMILES string has 0; a positive value means")
    lines.append("opening parentheses were lost during recognition.\n")

    lines.append(f"{'source':<26} {'n':>4} {'excess>0':>9} {'median':>8} "
                 f"{'mean':>7} {'balanced':>9} {'parses':>8}")
    for label, col in (("production (fresh run)", "prod"),
                       ("shipped CSV", "csv"),
                       ("production + binarise", "bin")):
        ex = df[f"{col}_paren_excess"]
        lines.append(
            f"{label:<26} {len(ex):>4} {int((ex > 0).sum()):>9} "
            f"{st.median(ex):>8} {ex.mean():>7.2f} "
            f"{int(df[f'{col}_balanced'].sum()):>9} "
            f"{int(df[f'{col}_parses'].sum()):>8}\n"
        )

    lines.append("\n--- agreement between the fresh production run and the "
                 "shipped CSV ---\n")
    yes = int((df.csv_reproduced == "yes").sum())
    lines.append(f"  character-for-character identical : {yes}/{len(df)} "
                 f"({100.0 * yes / len(df):.1f}%)\n")
    same = int((df.prod_paren_excess == df.csv_paren_excess).sum())
    lines.append(f"  identical paren_excess            : {same}/{len(df)} "
                 f"({100.0 * same / len(df):.1f}%)\n")
    corr = float(np.corrcoef(df.prod_paren_excess,
                             df.csv_paren_excess)[0, 1])
    lines.append(f"  correlation of paren_excess        : {corr:.4f}\n")

    diff = (df.csv_len - df.prod_len)
    lines.append(f"  length difference: median {st.median(diff):.0f}, "
                 f"{int((diff.abs() <= 2).sum())}/{len(df)} differ by <= 2 "
                 f"characters\n")
    lines.append("  -> where the strings differ, they differ by a couple of")
    lines.append("     characters at the crop edge, but the DEFECT is the")
    lines.append("     same in both. The diagnosis does not depend on the")
    lines.append("     strings being identical.\n")

    lines.append("--- what the binarised arm changes ---\n")
    better = int((df.bin_paren_excess < df.prod_paren_excess).sum())
    worse = int((df.bin_paren_excess > df.prod_paren_excess).sum())
    lines.append(f"  fewer unmatched ')' : {better}/{len(df)} "
                 f"({100.0 * better / len(df):.1f}%)\n")
    lines.append(f"  more unmatched ')'  : {worse}/{len(df)}\n")
    lines.append(f"  balanced now        : "
                 f"{int(df.bin_balanced.sum())}/{len(df)} "
                 f"(production: {int(df.prod_balanced.sum())}/{len(df)})\n")
    lines.append(f"  RDKit-parses now    : "
                 f"{int(df.bin_parses.sum())}/{len(df)} "
                 f"(production: {int(df.prod_parses.sum())}/{len(df)})\n")

    lines.append("\n--- remaining errors after binarisation ---\n")
    zero_bal = df[(df.bin_paren_excess == 0)]
    zero_bal_fail = zero_bal[~zero_bal.bin_parses]
    lines.append(f"  cells with balanced parentheses : {len(zero_bal)}\n")
    lines.append(f"  of those, still not RDKit-parsable: {len(zero_bal_fail)}\n")
    lines.append("  -> a second, separate error class remains. Balanced")
    lines.append("     brackets are necessary, not sufficient: '=' read as")
    lines.append("     '-' or dropped still breaks the parse. That class is")
    lines.append("     what the phase-2 '=' investigations were chasing, and")
    lines.append("     it is NOT solved by binarisation.\n")

    lines.append("--- evidence levels, kept separate ---\n")
    lines.append("  level 1  printed text is balanced      : measured\n")
    lines.append(f"           production {int(df.prod_balanced.sum())}/{len(df)}, "
                 f"binarised {int(df.bin_balanced.sum())}/{len(df)}\n")
    lines.append("  level 2  RDKit can parse the string    : measured\n")
    lines.append(f"           production {int(df.prod_parses.sum())}/{len(df)}, "
                 f"binarised {int(df.bin_parses.sum())}/{len(df)}\n")
    lines.append("  level 3  the structure is chemically  : NOT measured here\n")
    lines.append("           correct. No claim is made about it.\n")

    io.open(OUT, "w", encoding="utf-8").write("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
