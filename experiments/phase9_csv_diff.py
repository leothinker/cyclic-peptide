"""What changed between the two builds of the CSV?

The vote route was adopted as the default for WO2025162428, and the
CSV was rebuilt.  A rebuild is only an improvement if it changed what
it was supposed to change and nothing else, so the two files are
compared column by column.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

RDLogger.DisableLog("rdApp.*")

OLD = REPO / "data" / "processed" / "archive" / "compounds_combined_vote_no_bracket_rule.csv"
NEW = REPO / "data" / "processed" / "compounds_combined.csv"


def _load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["cmpd_id"]: row for row in csv.DictReader(handle)}


def _parses(text: str) -> bool:
    # RDKit returns an *empty molecule* for an empty string, not
    # ``None``, so an empty cell would otherwise be counted as a
    # well-formed SMILES.  71 rows have no printed SMILES at all.
    if not text.strip():
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def main() -> int:
    old = _load(OLD)
    new = _load(NEW)
    print(f"old {OLD.name}: {len(old)} rows")
    print(f"new {NEW.name}: {len(new)} rows")

    if set(old) != set(new):
        print("the compound numbers differ:")
        print("  only old:", sorted(set(old) - set(new))[:20])
        print("  only new:", sorted(set(new) - set(old))[:20])
        return 1

    columns = list(next(iter(new.values())).keys())
    changed: Counter[str] = Counter()
    changed_ids: dict[str, list[str]] = {}
    for compound_id, row in new.items():
        for column in columns:
            before = (old[compound_id].get(column) or "").strip()
            after = (row.get(column) or "").strip()
            if before != after:
                changed[column] += 1
                changed_ids.setdefault(column, []).append(compound_id)

    print("\n-- columns that changed --")
    if not changed:
        print("  (nothing)")
    for column, count in changed.most_common():
        print(f"  {column:<24} {count}")

    print("\n-- the printed SMILES --")
    before_valid = sum(
        1 for row in old.values() if _parses(row.get("smiles") or "")
    )
    after_valid = sum(
        1 for row in new.values() if _parses(row.get("smiles") or "")
    )
    print(f"  RDKit-valid before: {before_valid}")
    print(f"  RDKit-valid after : {after_valid}")

    became_worse = [
        compound_id
        for compound_id in new
        if _parses(old[compound_id].get("smiles") or "")
        and not _parses(new[compound_id].get("smiles") or "")
    ]
    became_better = [
        compound_id
        for compound_id in new
        if not _parses(old[compound_id].get("smiles") or "")
        and _parses(new[compound_id].get("smiles") or "")
    ]
    print(f"  became unparseable: {len(became_worse)} {became_worse[:10]}")
    print(f"  became parseable  : {len(became_better)}")

    print("\n-- the activity columns --")
    for field in ("kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity"):
        moved = changed_ids.get(field, [])
        print(f"  {field:<14} changed in {len(moved)} rows")
        for compound_id in moved[:5]:
            print(
                f"      {compound_id}: "
                f"{old[compound_id].get(field)!r} -> "
                f"{new[compound_id].get(field)!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
