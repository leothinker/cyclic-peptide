"""Find table pages in the source PDF that the golden set does not cover.

The 103 stored golden tables were cropped out of this PDF, so the PDF
holds more characterisation tables than the golden set -- pages the
OCR route has never been run on.  Those pages are a free
generalisation test: no new patent and no new reference strings are
needed, only the structural observation "this page carries a printed
SMILES column".

The script cross-references two existing measurements:

* ``pdf_table_pages_only.json`` -- which pages look like tables at all.
* ``pdf_to_golden_match.json``   -- which golden crop each page matches.

and prints the table pages that are *not* claimed by a strong golden
match, so the interesting ones can be rendered and inspected.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase3" / "external"


def main() -> int:
    table_pages = json.loads(
        (OUT / "pdf_table_pages_only.json").read_text(encoding="utf-8")
    )
    matches = json.loads(
        (OUT / "pdf_to_golden_match.json").read_text(encoding="utf-8")
    )

    table_ids = {int(r["pdf_page"]) for r in table_pages}
    by_page = {int(r["pdf_page"]): r for r in matches}

    # Which golden crop is the best match for each table page, and how
    # many table pages point at the same golden crop?
    claims: dict[str, list[int]] = defaultdict(list)
    for page in sorted(table_ids):
        entry = by_page.get(page)
        if entry is None:
            continue
        claims[entry["best_golden"]].append(page)

    print("=" * 78)
    print("golden crops claimed by the table pages")
    print("=" * 78)
    for name, pages in sorted(claims.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:45s} <- pdf pages {pages}")

    # A golden crop that several pages claim is suspicious: the pages
    # are not actually showing that crop.  Those are the gaps.
    print()
    print("=" * 78)
    print("table pages with a weak golden match (generalisation candidates)")
    print("=" * 78)
    candidates = []
    for page in sorted(table_ids):
        entry = by_page.get(page)
        if entry is None:
            continue
        rivals = claims[entry["best_golden"]]
        if entry["score"] < 0.6 or len(rivals) > 1:
            candidates.append(
                {
                    "pdf_page": page,
                    "best_golden": entry["best_golden"],
                    "score": entry["score"],
                    "shared_with": len(rivals),
                }
            )
    for row in candidates:
        print(
            f"  pdf page {row['pdf_page']:4d}: best={row['best_golden']:45s} "
            f"score={row['score']:.4f} shared={row['shared_with']}"
        )
    print(f"\n{len(candidates)} candidate pages")

    (OUT / "external_gap_candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT / 'external_gap_candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
