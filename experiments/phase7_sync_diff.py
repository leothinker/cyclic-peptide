"""List the curated files that the delivery repository does not have yet.

``tools/sync_delivery.py`` reports how many files differ and how many
are missing, but not which.  This script prints the list, so the
content of the sync can be reviewed before it is applied.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_sync_diff.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.sync_delivery import _sha256, collect  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", default=str(REPO.parent / "drug_ocr")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    target = Path(args.target)
    if not target.exists():
        print(f"target not found: {target}")
        return 2

    missing: list[str] = []
    differ: list[str] = []
    same = 0
    for src, dst_rel in collect(REPO):
        dst = target / dst_rel
        if not dst.exists():
            missing.append(dst_rel.as_posix())
        elif _sha256(src) == _sha256(dst):
            same += 1
        else:
            differ.append(dst_rel.as_posix())

    print(f"identical : {same}")
    print(f"different : {len(differ)}")
    for name in differ:
        print(f"   {name}")
    print(f"missing   : {len(missing)}")
    for name in missing:
        print(f"   {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
