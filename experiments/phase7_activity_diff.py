"""Show the activity cells that a policy does not reproduce.

``phase7_activity_policy.py`` counts the agreements; this script
shows the disagreements, so that each one can be looked at and
classified.  A disagreement is only a *difference from the archived
extraction* -- the archived extraction is the reference, not the
truth, and a difference has to be read before it is called an error.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_activity_diff.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from phase7_activity_policy import (  # noqa: E402
    ACTIVITY_FIELDS,
    POLICIES,
    _comparable,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", default="data/audit/phase7/activity_dump.csv"
    )
    parser.add_argument(
        "--archive",
        default=(
            "data/processed/archive/"
            "compounds_combined_production_route.csv"
        ),
    )
    parser.add_argument("--policy", default="line_first")
    parser.add_argument(
        "--field", default=None, help="Only this field."
    )
    parser.add_argument(
        "--out", default="data/audit/phase7/activity_diff.csv"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy = POLICIES[args.policy]
    with (REPO / args.dump).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (REPO / args.archive).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        archive = {r["cmpd_id"]: r for r in csv.DictReader(handle)}

    out_rows = []
    for row in rows:
        field = row["field"]
        if field not in ACTIVITY_FIELDS:
            continue
        if args.field and field != args.field:
            continue
        reference = archive.get(row["cmpd_id"])
        if reference is None:
            continue
        want = _comparable(field, reference.get(field))
        got = _comparable(field, policy(row))
        if want == got:
            continue
        out_rows.append({
            "cmpd_id": row["cmpd_id"],
            "field": field,
            "archived": reference.get(field) or "",
            "chosen": policy(row),
            "line": row.get("line_text") or "",
            "ch": row.get("ch_text") or "",
            "en": row.get("en_text") or "",
            "page": row.get("page") or "",
        })

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cmpd_id", "field", "archived", "chosen",
                "line", "ch", "en", "page",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"policy {args.policy}: {len(out_rows)} disagreements")
    by_field: dict[str, int] = {}
    for row in out_rows:
        by_field[row["field"]] = by_field.get(row["field"], 0) + 1
    for field, count in sorted(by_field.items()):
        print(f"  {field:<14} {count}")
    print(f"-> {out_path}")
    print()
    for row in out_rows[:40]:
        print(
            f"  {row['cmpd_id']:>5} {row['field']:<13}"
            f" archived={row['archived']!r}"
            f" chosen={row['chosen']!r}"
        )
        print(
            f"        line={row['line']!r} ch={row['ch']!r}"
            f" en={row['en']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
