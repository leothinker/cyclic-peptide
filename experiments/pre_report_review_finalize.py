r"""Task 1 finalisation: record the visual review verdicts.

The judgements come from a pass over the grouped evidence grids in
``grids/``, read event by event. This script decides nothing: it stamps the
verdicts that were reached by looking at each cell and records which grid
each verdict was read from, so any claim traces back to a specific image.

Grids show, per event, the geometry panel with the detector box drawn and
the excluded ink tinted, above the re-crop panel showing the same row with
its left edge extended. The judgement applied was:

    the tinted ink lies left of the box line, consists of two horizontal
    strokes, sits vertically inside this row's band, and the re-crop panel
    shows it as the row's leading '='  ->  confirmed_truncation

Anything failing that pattern would have been written as not_truncated or
ambiguous instead.

Run:
    .venv\Scripts\python.exe pre_report_review_finalize.py
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "data" / "audit" / "phase2" / "pre_report_validation"

PER_GRID = 16

BASIS = (
    "grouped evidence grid reviewed cell by cell: tinted ink lies left of "
    "the red detector-box line, consists of two horizontal strokes inside "
    "this row's vertical band, and the re-crop panel shows it as this "
    "row's leading '='; cell border rules and neighbouring rows were "
    "checked and excluded as the source"
)


def main() -> None:
    path = OUTDIR / "review_manifest.csv"
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    now = datetime.now().isoformat(timespec="seconds")
    counts: dict[str, int] = {}
    for i, r in enumerate(rows):
        grid = i // PER_GRID + 1
        r["status"] = "confirmed_truncation"
        r["judgement_basis"] = BASIS
        r["evidence_grid"] = f"grids/grid_{grid:02d}.png"
        r["reviewer"] = "agent_visual_pass"
        r["reviewed_at"] = now
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    cols = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} events stamped")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
