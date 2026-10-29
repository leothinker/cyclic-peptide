"""Compare the validated route against a fresh transcription of 2173.

Compound 2173 is the generalisation test that the patent itself
provides.  It sits on page I100376 (= PDF page 240 of
WO2025/162428), a page that was never part of the configuration
search, and the pipeline has never managed to read it: the shipped CSV
carries a bracket-less string for it and an empty ``smiles_best``.

The printed SMILES occupies eight lines in the right-hand cell of the
2173 row.  Each line was cropped from the golden table at 3x and read
character by character; the crops are kept next to this script so the
transcription can be re-checked.

The transcription status is ``printed_transcription_awaiting_human_check``
-- the same status as the 1039-1041 references.  It is a *text*
reference, not a verified chemical structure.
"""

from __future__ import annotations

import csv
import difflib
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase3" / "external"

# Read line by line from data/audit/phase3/external/_2173_line{1..8}.png
# (3x crops of the golden table I100376, columns x=1084..1912).
PRINTED_LINES = [
    "FC(F)CO[C@@H]1C[C@H]2C(=O)N(C)C3(CCC3)C",
    "(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)[C@@H](",
    "CC(=O)N(CC)[C@H]5C(=O)N[C@@H]([C@@H](C)",
    "CC)C(=O)N(C)[C@@H](C)C(=O)N6[C@@H](CC6)",
    "C(=O)N(CCC(C)C)[C@@H](CC7=CC=C(C=C7)C8=",
    "CN(N=N8)CC5)C(=O)N(C)CC(=O)N[C@@H](CCC9",
    "=CC(F)=C(C(F)=C9)C(F)(F)F)C(=O)N2C1)C(=O)N(C",
    "C)CC",
]

REFERENCE = "".join(PRINTED_LINES)


def edit_ops(a: str, b: str) -> list[tuple[str, str, str]]:
    """Character-level differences between two strings."""
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops: list[tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ops.append((tag, a[i1:i2], b[j1:j2]))
    return ops


def rdkit_info(text: str) -> dict[str, object]:
    info: dict[str, object] = {"parses": False}
    if not text:
        return info
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return info
        info["parses"] = True
        info["formula"] = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
        info["exact_mass"] = round(Descriptors.ExactMolWt(mol), 4)
        info["heavy_atoms"] = mol.GetNumHeavyAtoms()
    except Exception:
        pass
    return info


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    cell = CELLS / "I100376_2173_text.png"
    image = cv2.imread(str(cell))
    if image is None:
        print(f"no cell crop at {cell}")
        return 1

    production = PrintedSmilesOCR(route="production").read_cell(image) or ""
    validated = PrintedSmilesOCR(route="validated").read_cell(image) or ""

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        csv_row = {}
        for row in csv.DictReader(handle):
            if str(row.get("cmpd_id", "")).strip() == "2173":
                csv_row = row
                break
    shipped = (csv_row.get("smiles") or "").strip()
    decimer = (csv_row.get("smiles_decimer") or "").strip()

    print("=== compound 2173, printed SMILES on I100376 ===")
    print(f"reference length          : {len(REFERENCE)}")
    print(f"validated route length    : {len(validated)}")
    print(f"production route length   : {len(production)}")
    print(f"shipped CSV smiles length : {len(shipped)}")
    print(f"DECIMER length            : {len(decimer)}")
    print()

    for label, text in (
        ("validated", validated),
        ("production", production),
        ("shipped", shipped),
        ("decimer", decimer),
    ):
        ops = edit_ops(REFERENCE, text)
        info = rdkit_info(text)
        print(
            f"{label:10s} edit_vs_reference={sum(1 for _ in ops):3d} "
            f"brackets={text.count('(')}/{text.count(')')} "
            f"rdkit={info.get('parses')} "
            f"formula={info.get('formula', '')}"
        )
        for tag, got, want in ops[:8]:
            print(f"             {tag:8s} got={got!r} expected={want!r}")
        if len(ops) > 8:
            print(f"             ... {len(ops) - 8} more")
        print()

    rows = [
        {
            "cmpd_id": "2173",
            "source_page": "I100376 (PDF page 240)",
            "reference_length": len(REFERENCE),
            "validated_length": len(validated),
            "production_length": len(production),
            "shipped_length": len(shipped),
            "validated_edit_vs_reference": len(edit_ops(REFERENCE, validated)),
            "production_edit_vs_reference": len(edit_ops(REFERENCE, production)),
            "shipped_edit_vs_reference": len(edit_ops(REFERENCE, shipped)),
            "validated_brackets": f"{validated.count('(')}/{validated.count(')')}",
            "production_brackets": f"{production.count('(')}/{production.count(')')}",
            "shipped_brackets": f"{shipped.count('(')}/{shipped.count(')')}",
            "validated_rdkit": rdkit_info(validated).get("parses", False),
            "validated_formula": rdkit_info(validated).get("formula", ""),
            "reference_status": "printed_transcription_awaiting_human_check",
        }
    ]
    out_csv = OUT / "external_2173_check.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT / "external_2173_reference.json").write_text(
        json.dumps(
            {
                "cmpd_id": "2173",
                "source_page": "I100376 (PDF page 240 of WO2025/162428)",
                "printed_lines": PRINTED_LINES,
                "reference": REFERENCE,
                "status": "printed_transcription_awaiting_human_check",
                "line_crops": [
                    f"_2173_line{i}.png" for i in range(1, len(PRINTED_LINES) + 1)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_csv}")
    print(f"wrote {OUT / 'external_2173_reference.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
