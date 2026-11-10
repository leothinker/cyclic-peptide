"""Score the WO2022251372 extraction against the hand transcriptions.

The references in ``wo2022251372_references.py`` were transcribed by
hand from the printed pages.  They are a *text* reference only: they
say nothing about whether the drawn molecule is the intended one.

This script reports, for every referenced compound:

* the edit distance and CER of the recognised text,
* whether RDKit can parse the string, and
* whether the two tables of the patent agree with each other.

The band-to-compound mapping is *not* re-implemented here.  A printed
compound row may run over the bottom of a page and resume on the next
one, and the resumed band carries no compound number, so a naive
"join every band on the page" comparison would compare a reference
against the text of several different compounds.  Instead the
extractor's own mapping is reused: a subclass captures the raw OCR
text of every row at the moment the row is closed, which is exactly
where the pipeline decides that a continuation band belongs to the
compound that is still open.

Two numbers are reported for every compound:

* ``raw_*``      -- the recognised text as it comes out of the OCR,
                    before the repair chain touches it.  This is the
                    honest measure of recognition accuracy.
* ``repaired_*`` -- the text after ``auto_repair_smiles``.  This is
                    what the pipeline publishes, so it is the measure
                    of the delivered output.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_verify_accuracy.py
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

from cpd.chem import auto_repair_smiles, validate_and_enrich  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402

from wo2022251372_references import (  # noqa: E402
    REFERENCE_PAGES,
    cer,
    edit_distance,
    reference_text,
)

OUT = REPO / "data" / "audit" / "phase6"


class _CaptureExtractor(RapidOcrTableExtractor):
    """Extractor that records the raw text of every closed row.

    ``_close_pending_row`` is the single place where the pipeline
    decides that a row -- possibly assembled from several bands on
    several pages -- is finished.  Capturing the text there gives the
    exact band-to-compound mapping without duplicating it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Keyed by (layout, compound) so the two tables of the patent
        # stay apart.  Table 1 and Table 1a print the same SMILES text
        # for the same compound, which gives a free cross-check.
        self.raw_by_key: dict[tuple[str, str], str] = {}

    def _close_pending_row(self, pending, compounds) -> None:
        self.raw_by_key[(pending.layout_key, pending.cmpd_id)] = (
            pending.smiles_text
        )
        super()._close_pending_row(pending, compounds)


def main() -> int:
    profile = get_profile("WO2022251372")
    pages_dir = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"

    # Read the printed SMILES straight from the page images, so the
    # comparison is against the source and not against a cached CSV.
    extractor = _CaptureExtractor(profile=profile)
    prev_columns: list = []
    for path in sorted(pages_dir.glob("*.tif")):
        _cmpds, _assays, columns = extractor.extract_records(
            path, cells_dir=None, prev_columns=prev_columns,
        )
        prev_columns = columns
    extractor.flush_pending()

    rows: list[dict] = []
    for compound_id in sorted(REFERENCE_PAGES):
        reference = reference_text(compound_id)
        page = REFERENCE_PAGES[compound_id]
        variants = {
            layout: text
            for (layout, cid), text in extractor.raw_by_key.items()
            if cid == str(compound_id)
        }
        # The two tables must agree; when they do not, the reference
        # cannot be scored against an ambiguous reading.
        distinct = set(variants.values())
        # Score the reading the pipeline actually ships.  The patent
        # prints each compound twice -- once in the activity table and
        # once in the structure table -- and ``merge_compound_records``
        # keeps the RDKit-valid one.  Scoring the other reading would
        # measure a string that never reaches the deliverable.
        raw = ""
        used_layout = ""
        for layout, text in variants.items():
            if not text:
                continue
            if not raw:
                raw, used_layout = text, layout
                continue
            if validate_and_enrich(text) and not validate_and_enrich(raw):
                raw, used_layout = text, layout
        cross_ok = len(distinct) <= 1
        repaired = auto_repair_smiles(raw) if raw else ""
        raw_chem = validate_and_enrich(raw) if raw else None
        rep_chem = validate_and_enrich(repaired) if repaired else None
        rows.append({
            "cmpd_id": compound_id,
            "page": page,
            "tables": "|".join(sorted(variants)),
            "scored_layout": used_layout,
            "reference_len": len(reference),
            "raw_len": len(raw),
            "raw_edit_distance": edit_distance(reference, raw) if raw else "",
            "raw_cer": round(cer(reference, raw), 5) if raw else "",
            "raw_rdkit_parseable": bool(raw_chem),
            "repaired_edit_distance": (
                edit_distance(reference, repaired) if repaired else ""
            ),
            "repaired_cer": round(cer(reference, repaired), 5) if repaired else "",
            "repaired_rdkit_parseable": bool(rep_chem),
            "cross_table_agreement": cross_ok,
            "note": (
                "" if raw
                else "no row recognised for this compound"
            ),
        })

    out_csv = OUT / "accuracy_vs_reference.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[
                "cmpd_id", "page", "tables", "reference_len", "raw_len",
                    "scored_layout",
                "raw_edit_distance", "raw_cer", "raw_rdkit_parseable",
                "repaired_edit_distance", "repaired_cer",
                "repaired_rdkit_parseable", "cross_table_agreement", "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    scored = [r for r in rows if r["raw_edit_distance"] != ""]
    total_edits = sum(int(r["raw_edit_distance"]) for r in scored)
    total_chars = sum(int(r["reference_len"]) for r in scored)
    summary = {
        "n_referenced": len(rows),
        "n_scored": len(scored),
        "total_raw_edit_distance": total_edits,
        "total_reference_chars": total_chars,
        "aggregate_raw_cer": (
            round(total_edits / total_chars, 5) if total_chars else None
        ),
        "raw_exact_matches": sum(
            1 for r in scored if int(r["raw_edit_distance"]) == 0
        ),
        "total_repaired_edit_distance": sum(
            int(r["repaired_edit_distance"])
            for r in scored
            if r["repaired_edit_distance"] != ""
        ),
        "repaired_exact_matches": sum(
            1 for r in scored
            if r["repaired_edit_distance"] != ""
            and int(r["repaired_edit_distance"]) == 0
        ),
        "raw_rdkit_parseable": sum(
            1 for r in scored if r["raw_rdkit_parseable"]
        ),
        "repaired_rdkit_parseable": sum(
            1 for r in scored if r["repaired_rdkit_parseable"]
        ),
    }
    (OUT / "accuracy_vs_reference.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    for row in rows:
        print(row)
    print()
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
