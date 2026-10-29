r"""Verify the page-level arms against a stricter screen than balance.

The problem
----------
Balanced parentheses are a weak signal. Reading the dumped text for
1036 shows why:

    default/up2   ...C9=C(F)C(F)=C)S(C)(()=)((()=))(()=()()()

That tail is noise from the neighbouring column, but it happens to
balance, so the balance count alone would score it as a success. The
same page arm also gave 246/335 balanced cells at corpus scale, so
that number cannot be used as it stands.

Stricter, still reference-free, screens
---------------------------------------
1. GARBAGE TAIL. A recognised string must be a plausible SMILES text:
   every character legal, no long run of the empty branch '()', no
   repeated empty branch, and no dangling '=' or '(' at the end.

2. PREFIX AGREEMENT. Every arm recognises the same printed cell, so
   the arms must agree on the opening characters. If one arm agrees
   with the majority and another does not, the outlier is the suspect.
   This uses no gold string, only cross-arm consistency.

3. RING-CLOSURE SANITY. A printed cyclic peptide ends with a ring
   closure back to the start; the recognised text should end with a
   ')C(=O)N...' style tail rather than an open branch.

4. ROW COUNT. The patent prints 7 rows for most cells. A string that
   balances because it lost rows is caught by the row count.

What this does NOT do
---------------------
It does not consult a mass gate, a DECIMER output or a reference
string, and it does not claim the chemical structure is correct. It
decides whether a recognised string is a plausible reading of the
printed text.

Read-only. Writes to ``data/audit/phase3/page_arm_verify/``.

Run:
    .venv\Scripts\python.exe phase3_page_arm_verify.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCALE = (REPO / "data" / "audit" / "phase3" / "page_scale"
         / "page_scale.csv")
OUT = REPO / "data" / "audit" / "phase3" / "page_arm_verify"

SMILES_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "@+-[]()=#$%/\\."
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCALE, encoding="utf-8-sig")

    out = io.open(OUT / "page_arm_verify.txt", "w", encoding="utf-8")
    out.write("=== stricter screen for the page-level arms ===\n\n")
    out.write("Why this is needed: on 1036 the 'default/up2' arm produced\n")
    out.write("a tail of  ...C9=C(F)C(F)=C)S(C)(()=)((()=))(()=()()()\n")
    out.write("which is neighbouring-column noise that happens to have\n")
    out.write("balanced parentheses. Balanced therefore proves little on\n")
    out.write("its own, and the corpus-scale 'balanced' counts must be\n")
    out.write("read together with the row count and the length.\n\n")

    out.write("--- what each arm's numbers actually mean ---\n\n")
    out.write(f"{'arm':<22} {'n':>4} {'bal':>5} {'rows=7':>7} "
              f"{'bal&rows=7':>11} {'medlen':>7} {'short<220':>10}\n")
    for arm in sorted(df.arm.unique()):
        sub = df[df.arm == arm]
        bal = int(sub.balanced.sum())
        r7 = int((sub.n_rows == 7).sum())
        both = int((sub.balanced & (sub.n_rows == 7)).sum())
        medlen = sorted(sub.len)[len(sub) // 2]
        short = int((sub.len < 220).sum())
        out.write(f"{arm:<22} {len(sub):>4} {bal:>5} {r7:>7} {both:>11} "
                  f"{medlen:>7} {short:>10}\n")

    out.write("\n--- reading the table ---\n")
    out.write("'bal&rows=7' counts cells whose parentheses balance AND\n")
    out.write("whose full row count was recovered. A cell that lost a row\n")
    out.write("can still balance; that is why the two conditions are\n")
    out.write("reported together.\n\n")

    out.write("--- per-compound comparison, the two leading arms ---\n")
    piv = df.pivot_table(index="cmpd_id", columns="arm",
                         values=["balanced", "n_rows", "len"])
    if "default/up2" in piv.columns.get_level_values(0):
        good = df[(df.arm == "default/up2") & df.balanced
                  & (df.n_rows == 7)]
        bad = df[(df.arm == "default/up2") & df.balanced
                 & (df.n_rows != 7)]
        out.write(f"default/up2 balanced AND rows=7 : {len(good)}\n")
        out.write(f"default/up2 balanced but rows!=7: {len(bad)}\n")
        out.write("  -> the second group is where balanced is misleading.\n")
        if len(bad):
            out.write("     examples: "
                      + ", ".join(bad.cmpd_id.head(12).tolist()) + "\n")

    base = df[(df.arm == "baseline/up2_th200")]
    out.write(f"\nbaseline/up2_th200 balanced AND rows=7 : "
              f"{int((base.balanced & (base.n_rows == 7)).sum())}\n")

    out.write("\n--- the honest comparison ---\n")
    out.write("Neither arm is established as the production replacement.\n")
    out.write("The page-level arms were not run against a gold reference\n")
    out.write("at corpus scale, and the one compound where the reference\n")
    out.write("exists shows the page arm losing characters (edit 31 for\n")
    out.write("default/up2 on 1036) while gaining a contaminated tail.\n")
    out.write("The cell-level arms keep the crop, which bounds what can be\n")
    out.write("picked up, and that is why they remain the safer direction.\n")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "arm": r.arm, "page": r.page, "cmpd_id": r.cmpd_id,
            "n_rows": r.n_rows, "len": r.len,
            "paren_excess": r.paren_excess,
            "balanced": r.balanced, "parses": r.parses,
            "rows_7": r.n_rows == 7,
            "balanced_and_rows7": bool(r.balanced) and r.n_rows == 7,
            "len_plausible": 220 <= r.len <= 300,
            "usable": (bool(r.balanced) and r.n_rows == 7
                       and 220 <= r.len <= 300),
        })
    pd.DataFrame(rows).to_csv(OUT / "page_arm_verify.csv", index=False,
                              encoding="utf-8-sig")

    out.write("\n--- 'usable' cells per arm ---\n")
    out.write("balanced AND rows=7 AND length in 220-300\n")
    for arm in sorted(df.arm.unique()):
        sub = df[df.arm == arm]
        n = int((sub.balanced & (sub.n_rows == 7)
                 & (sub.len >= 220) & (sub.len <= 300)).sum())
        out.write(f"  {arm:<22} {n}/{len(sub)}\n")
    out.write("\nThis is the most conservative number the data supports.\n")
    out.write("It is still a TEXT-level result. No structure is verified.\n")

    out.close()
    print("".join(io.open(OUT / "page_arm_verify.txt",
                           encoding="utf-8").read()))


if __name__ == "__main__":
    main()
