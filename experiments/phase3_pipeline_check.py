"""End-to-end check of the validated route through the real pipeline.

The route is exercised through ``RapidOcrTableExtractor`` (not through
the standalone module), so this covers the column geometry, the cell
crop and the recognition in one go.

For every page it prints the recognised SMILES per compound and, when
a reference string is available, the edit distance against it.

References are text-level only.  1038 is the researcher-supplied
string; 1039-1041 are the phase-3 transcriptions and are still
awaiting a human check.  No mass gate, DECIMER output or RDKit
parseability is used to choose anything.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import ROUTE_VALIDATED  # noqa: E402
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402

PAGES = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
OUT = REPO / "data" / "audit" / "phase3" / "pipeline_check"

REFERENCES: dict[str, tuple[str, str]] = {
    "1038": (
        "researcher_supplied_held_out",
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)"
        "[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H]"
        "(CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)"
        "[C@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N(C)C",
    ),
    "1039": (
        "phase3_transcription_awaiting_human_check",
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)"
        "[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H]"
        "(CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)"
        "[C@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N9CCCC9",
    ),
    "1040": (
        "phase3_transcription_awaiting_human_check",
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CC3)C(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)CC)C(=O)N(C)[C@@H](C)"
        "C(=O)N6[C@@H](CC6)C(=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)"
        "C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C",
    ),
    "1041": (
        "phase3_transcription_awaiting_human_check",
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CCCC3)C(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)CC)C(=O)N(C)[C@@H](C)"
        "C(=O)N6[C@@H](CC6)C(=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)"
        "C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C",
    ),
}


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    extractor = RapidOcrTableExtractor(smiles_route=ROUTE_VALIDATED)

    rows = []
    for stem in (
        "PCTCN2025075389-ftappb-I100297",
        "PCTCN2025075389-ftappb-I100298",
        "PCTCN2025075389-ftappb-I100299",
        "PCTCN2025075389-ftappb-I100300",
        "PCTCN2025075389-ftappb-I100377",
    ):
        path = PAGES / f"{stem}.jpg"
        if not path.exists():
            print(f"missing {path.name}")
            continue
        compounds, assays, columns = extractor.extract_records(path)
        print(f"\n=== {stem} ({len(compounds)} compounds) ===")
        for rec in compounds:
            cmpd = rec.cmpd_id
            text = rec.smiles or ""
            reference = REFERENCES.get(cmpd)
            if reference:
                status, ref = reference
                edits = edit_distance(text, ref)
                cer = edits / len(ref)
                print(
                    f"  {cmpd}: len={len(text):4d} edit={edits:3d} "
                    f"cer={cer:.5f}  [{status}]"
                )
            else:
                edits = ""
                print(f"  {cmpd}: len={len(text):4d} (no reference)")
            rows.append(
                {
                    "source_image": path.name,
                    "cmpd_id": cmpd,
                    "reference_status": reference[0] if reference else "",
                    "ocr_len": len(text),
                    "edit": edits,
                    "text": text,
                }
            )

    with (OUT / "pipeline_check.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "pipeline_check.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
