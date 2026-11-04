"""Where the WO2022251372 readings differ from the hand transcriptions.

The route comparison reports how many characters are wrong.  This
script shows *which* characters, so that a repair can be judged on
the error it actually has to fix rather than on a count.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_diff.py
"""

from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from wo2022251372_references import REFERENCE_LINES  # noqa: E402

OUT = REPO / "data" / "audit" / "phase9"


def _readings() -> dict[str, str]:
    """The strings the shipped WO2022251372 route produced."""
    path = REPO / "data" / "audit" / "phase8" / "patent2_vote_effect.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["cmpd_id"]: row["text_after"] for row in rows}


def main() -> int:
    readings = _readings()
    out_rows: list[dict[str, object]] = []
    for compound_id, lines in REFERENCE_LINES.items():
        reference = "".join(lines)
        text = readings.get(str(compound_id), "")
        matcher = difflib.SequenceMatcher(
            a=reference, b=text, autojunk=False
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            out_rows.append({
                "cmpd_id": compound_id,
                "operation": tag,
                "reference_fragment": reference[max(0, i1 - 12):i2 + 12],
                "read_fragment": text[max(0, j1 - 12):j2 + 12],
                "reference_position": i1,
                "reference_length": len(reference),
                "read_length": len(text),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "patent2_character_differences.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"-> {out_csv}")

    for row in out_rows:
        print(
            f"cmpd {row['cmpd_id']:>3}  {row['operation']:<8}"
            f"  printed ...{row['reference_fragment']}..."
            f"   read ...{row['read_fragment']}..."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
