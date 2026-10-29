"""Read-only cross-check: do golden-table row counts match struct crops?"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "data" / "audit" / "phase2" / "layout_probe.txt"
CSV = REPO / "data" / "processed" / "compounds_combined.csv"

per = collections.defaultdict(list)
df = pd.read_csv(CSV, dtype=str)
for s in df["structure_image"].dropna():
    m = re.search(r"(I100\d+)_(\d+)_struct", s)
    if m:
        per[m.group(1)].append(int(m.group(2)))

probe = {}
for line in PROBE.read_text(encoding="utf-8").splitlines():
    if "\t" not in line:
        continue
    f = line.split("\t")
    page = f[0].split("-")[-1]
    shape = re.search(r"shape=\((\d+),", f[1])
    layout = re.search(r"layout=(\w+)", f[3])
    rows = re.search(r"rows=(\d+)", f[4])
    probe[page] = (int(shape.group(1)), layout.group(1), int(rows.group(1)))

total_smiles = 0
bad = []
for pg in sorted(per):
    if pg not in probe:
        bad.append((pg, "no_probe"))
        continue
    nrows = probe[pg][2]
    nstruct = len(per[pg])
    layout = probe[pg][1]
    if layout == "smiles":
        total_smiles += nstruct
        if nrows not in (nstruct, nstruct + 1):
            bad.append((pg, nrows, nstruct))

print("struct crops on smiles pages:", total_smiles)
print("mismatched smiles pages:", bad)
print("page holding 2173:", [pg for pg, ids in per.items() if 2173 in ids])
print("probe layouts:", collections.Counter(v[1] for v in probe.values()))
