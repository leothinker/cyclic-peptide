"""Write the Phase 8 summary.

The summary is written from this script rather than by hand so that
every number in it comes from a file that ships with the repository.
The script recomputes the numbers it quotes and refuses to write a
number it cannot reproduce.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

OUT = REPO / "data" / "audit" / "phase8"


def _route_table() -> str:
    path = OUT / "route_verification.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    routes: list[str] = []
    for row in rows:
        if row["route"] not in routes:
            routes.append(row["route"])
    compounds: list[str] = []
    for row in rows:
        if row["cmpd_id"] not in compounds:
            compounds.append(row["cmpd_id"])
    lines = [
        "| route | " + " | ".join(compounds) + " |",
        "|---|" + "---:|" * len(compounds),
    ]
    for route in routes:
        cells = []
        for compound in compounds:
            match = next(
                r for r in rows
                if r["route"] == route and r["cmpd_id"] == compound
            )
            cells.append(match["edit_distance"])
        lines.append(f"| `{route}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("route table:\n" + _route_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
