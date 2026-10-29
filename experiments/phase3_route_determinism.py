r"""Is the page route deterministic?

Why
---
Any OCR configuration that is going to replace a shipped one has to
produce the same text on every run. The page route is a multi-stage
pipeline (detect on the page, crop each row, recognise each crop), and
each stage could in principle be unstable, so the claim has to be
tested rather than assumed.

The check repeats the winning arm on the same pages and compares the
recognised strings character by character. Anything that is not
identical is reported.

Read-only. Writes to ``data/audit/phase3/route_determinism/``.

Run:
    .venv\Scripts\python.exe phase3_route_determinism.py --repeats 3
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import edit_ops  # noqa: E402
from phase3_page_det_line_rec import _load_manifest  # noqa: E402
from phase3_page_det_line_rec import _process_page  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "route_determinism"

ARM = "pdet_def_rec_def_2x"
PAGES = ["I100297", "I100298", "I100300"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pages", nargs="*", default=PAGES)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _load_manifest()
    lines = []
    lines.append("=== determinism of the page route ===")
    lines.append("")
    lines.append(f"arm    : {ARM}")
    lines.append(f"pages  : {', '.join(args.pages)}")
    lines.append(f"repeats: {args.repeats}")
    lines.append("")
    lines.append("A run is identical when every compound on the page has")
    lines.append("the same recognised text as the first run.")
    lines.append("")

    n_cells = 0
    n_diff = 0
    worst = 0
    for page in args.pages:
        cells = cells_by_page.get(page, [])
        runs = []
        for _ in range(args.repeats):
            rows = _process_page(page, cells, arms=(ARM,))
            runs.append({r["cmpd_id"]: r["text"] for r in rows if r["arm"] == ARM})
        base = runs[0]
        lines.append(f"--- {page}: {len(base)} compounds")
        for i, run in enumerate(runs[1:], start=2):
            for cid, txt in base.items():
                n_cells += 1
                other = run.get(cid, "")
                if other == txt:
                    continue
                n_diff += 1
                d = edit_ops(txt, other)
                worst = max(worst, d)
                lines.append(f"    run {i} differs for {cid}: edit {d}")
        lines.append("")

    lines.append(f"compound-runs compared : {n_cells}")
    lines.append(f"non-identical          : {n_diff}")
    lines.append(f"worst edit distance    : {worst}")
    lines.append("")
    lines.append("VERDICT: deterministic" if n_diff == 0 else "VERDICT: NOT deterministic")

    io.open(OUT / "route_determinism.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[-10:]))


if __name__ == "__main__":
    main()
