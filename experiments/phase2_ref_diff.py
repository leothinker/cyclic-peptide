"""Diagnose transcription slips: diff my reference against DECIMER output.

Used ONLY to locate where an agent transcription is wrong. The DECIMER
string is never used to build a reference.
"""

from __future__ import annotations

from pathlib import Path

import csv
import difflib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
from cpd.ms_validation import validate_against_ms  # noqa: E402
from phase2_ocr_compare import REFERENCES  # noqa: E402

rows = {
    r["cmpd_id"]: r
    for r in csv.DictReader(
        open("data/processed/compounds_combined.csv", encoding="utf-8-sig")
    )
}

for cid, ref in REFERENCES.items():
    mine = ref["smiles"]
    r = rows[cid]
    dec = (r.get("smiles_decimer") or "").strip()
    mz = float(r["ms_mz"])
    pol = r["ms_polarity"]
    vm = validate_against_ms(mine, mz, pol)
    vd = validate_against_ms(dec, mz, pol) if dec else {}
    print("=" * 78)
    print(
        f"{cid}: mine={len(mine)}ch {vm.get('formula')} "
        f"mass={vm.get('observed_mass')}  |  "
        f"decimer={len(dec)}ch {vd.get('formula')} mass={vd.get('observed_mass')}"
    )
    print(f"  patent mz={mz} {pol}")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=mine, b=dec
    ).get_opcodes():
        if tag == "equal":
            continue
        print(
            f"  {tag:8s} mine[{i1}:{i2}]={mine[i1:i2]!r:28s} "
            f"decimer[{j1}:{j2}]={dec[j1:j2]!r}"
        )
