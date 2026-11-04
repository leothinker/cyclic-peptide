"""Show the character differences between a route and a reference.

``phase7_route_verify.py`` counts the differences; this script shows
where they are, so that each one can be traced to a printed line and
looked at in the image.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_route_diff.py \
        --route validated --cmpd 1037
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from phase7_route_verify import (  # noqa: E402
    PAGES,
    REFERENCES,
    _read_from_page,
)

from cpd.ocr.printed_smiles import ROUTES, PrintedSmilesOCR  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default="validated", choices=ROUTES)
    parser.add_argument("--cmpd", nargs="*", default=sorted(REFERENCES))
    parser.add_argument("--patent", default="WO2025162428")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reader = PrintedSmilesOCR(route=args.route)
    image_dir = REPO / "data" / "raw" / args.patent / "golden_tables"

    for cmpd in args.cmpd:
        reference = REFERENCES[cmpd]
        page = image_dir / PAGES[cmpd]
        text = _read_from_page(reader, page, cmpd)
        if text is None:
            print(f"{cmpd}: not read")
            continue
        print(f"\n=== {cmpd}  route {args.route} ===")
        print(f"  reference {len(reference)} chars, read {len(text)} chars")
        if text == reference:
            print("  identical")
            continue
        matcher = difflib.SequenceMatcher(
            None, reference, text, autojunk=False
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            left = max(0, i1 - 24)
            ref_part = reference[i1:i2]
            read_part = text[j1:j2]
            print(f"  {tag:<8} at reference[{i1}:{i2}] read[{j1}:{j2}]")
            print(f"      reference ...{reference[left:i1]}[{ref_part}]")
            print(f"      read      ...{text[max(0, j1 - 24):j1]}[{read_part}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
