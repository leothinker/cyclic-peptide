"""Show the exact differences behind the accuracy report.

``phase6_verify_accuracy.py`` reports the numbers; this script shows
*where* the characters differ, which is what is needed to decide
whether a difference is an OCR error or a transcription error.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_diff_detail.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402

from phase6_verify_accuracy import _CaptureExtractor  # noqa: E402
from wo2022251372_references import (  # noqa: E402
    REFERENCE_PAGES,
    REFERENCE_LINES,
    reference_text,
)

PAGES_DIR = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"


def _locate(lines: tuple[str, ...], offset: int) -> str:
    """Describe where ``offset`` falls inside the printed lines."""
    running = 0
    for index, line in enumerate(lines):
        if running + len(line) > offset:
            return (
                f"line {index + 1} char {offset - running + 1}"
                f"/{len(line)} ({line!r})"
            )
        running += len(line)
    return f"offset {offset} (past the end)"


def main() -> int:
    profile = get_profile("WO2022251372")
    extractor = _CaptureExtractor(profile=profile)
    prev_columns: list = []
    for path in sorted(PAGES_DIR.glob("*.tif")):
        _cmpds, _assays, columns = extractor.extract_records(
            path, cells_dir=None, prev_columns=prev_columns,
        )
        prev_columns = columns
    extractor.flush_pending()

    for compound_id in sorted(REFERENCE_PAGES):
        reference = reference_text(compound_id)
        variants = {
            layout: text
            for (layout, cid), text in extractor.raw_by_key.items()
            if cid == str(compound_id)
        }
        print("=" * 72)
        print(f"compound {compound_id}  reference length {len(reference)}")
        for layout, text in sorted(variants.items()):
            print(f"  [{layout}] length {len(text)}")
        distinct = set(variants.values())
        if len(distinct) > 1:
            names = sorted(variants)
            left, right = variants[names[0]], variants[names[1]]
            print(f"  !! the two tables disagree:")
            for op, a1, a2, b1, b2 in difflib.SequenceMatcher(
                None, left, right
            ).get_opcodes():
                if op != "equal":
                    print(
                        f"     {op:8s} {names[0]}={left[a1:a2]!r} "
                        f"{names[1]}={right[b1:b2]!r}"
                    )
        for layout, text in sorted(variants.items()):
            if text == reference:
                print(f"  [{layout}] == reference (exact)")
                continue
            print(f"  [{layout}] vs reference:")
            printed = REFERENCE_LINES[compound_id]
            for op, a1, a2, b1, b2 in difflib.SequenceMatcher(
                None, reference, text
            ).get_opcodes():
                if op != "equal":
                    print(
                        f"     {op:8s} reference={reference[a1:a2]!r} "
                        f"read={text[b1:b2]!r} @{a1}"
                    )
                    print(
                        f"              position: "
                        f"{_locate(printed, a1)}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
