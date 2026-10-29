"""Sanity-check the Phase-2 reference strings (read-only)."""

from __future__ import annotations

from pathlib import Path

import csv
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

hdr = (
    "cmpd", "len", "parse", "formula", "MW", "m/z", "pol",
    "dDa", "ppm", "tol_ppm", "verdict",
)
print("{:>5s} {:>4s} {:>6s} {:>13s} {:>9s} {:>9s} {:>10s} {:>8s} {:>8s} {:>8s} {:>18s}".format(*hdr))
for cid, ref in REFERENCES.items():
    s = ref["smiles"]
    mol = Chem.MolFromSmiles(s)
    r = rows[cid]
    mz = float(r["ms_mz"])
    pol = r["ms_polarity"]
    v = validate_against_ms(s, mz, pol)
    obs = v.get("observed_mass") or 0.0
    exp = v.get("expected_mass") or 0.0
    print(
        "{:>5s} {:>4d} {:>6s} {:>13s} {:>9.3f} {:>9.1f} {:>10s} {:>8.3f} {:>8s} {:>8s} {:>18s}".format(
            cid,
            len(s),
            "yes" if mol else "NO",
            str(v.get("formula")),
            exp,
            mz,
            pol,
            obs - exp,
            str(v.get("error_ppm")),
            str(v.get("tolerance_ppm")),
            str(v.get("verdict")),
        )
    )
