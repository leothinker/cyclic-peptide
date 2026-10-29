"""Independent cross-check of the OCR output against PubChem.

The external-validation question -- "does the pipeline still work on a
document it has never seen?" -- was answered in two parts:

1. *Within the patent*: compound 2173, which the pipeline never
   managed to read (the shipped CSV carries a bracket-less string and
   an empty ``smiles_best``).  The validated route reads it.
2. *Outside the patent*: a second patent document could not be
   downloaded from this environment (Google Patents answers 503, EPO
   500, Espacenet 403 -- see ``fetch_WO2022234852.json``).  Instead,
   PubChem is used as an **independent third-party chemical reference**:
   a structure that PubChem can recognise, and whose formula and
   exact mass it agrees with, is evidence that the recognised string is
   a real chemical entity rather than a plausible-looking string.

What this does and does not establish
--------------------------------------

* It is a **text/structure sanity check against an outside authority**.
  It is *not* proof that the OCR output is the structure drawn in the
  patent; only the drawing can settle that.
* A PubChem hit for the *same* structure as the patent's DECIMER
  output, with the OCR output agreeing with it, is the strongest form
  of evidence available without a human check.
* A miss is **not** a failure of the OCR: these are novel patent
  compounds and most of them are simply not deposited in PubChem.
  Misses are recorded as such and never counted as errors.
"""

from __future__ import annotations

import csv
import json
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase3" / "external"

USER_AGENT = "cyclic-peptide-pipeline/1.0 (patent SMILES OCR validation)"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_json(url: str, timeout: int = 40) -> tuple[int, dict | None]:
    context = ssl.create_default_context()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            body = response.read()
        return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return 0, None


def query_by_smiles(smiles: str) -> dict:
    """Look a SMILES up in PubChem by identity."""
    quoted = urllib.parse.quote(smiles, safe="")
    url = (
        f"{PUG}/compound/smiles/property/"
        "MolecularFormula,ExactMass,IUPACName,InChIKey/JSON?smiles="
        + quoted
    )
    status, data = fetch_json(url)
    if status != 200 or not data:
        return {"found": False, "status": status}
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props or not props[0].get("CID"):
        return {"found": False, "status": status}
    first = props[0]
    return {
        "found": True,
        "status": status,
        "cid": first.get("CID"),
        "formula": first.get("MolecularFormula"),
        "exact_mass": first.get("ExactMass"),
        "inchikey": first.get("InChIKey"),
        "iupac": first.get("IUPACName"),
    }


def formula_of(smiles: str) -> tuple[str | None, float | None]:
    """Local formula and exact mass, used to compare with PubChem."""
    if not smiles:
        return None, None
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        return (
            Descriptors.rdMolDescriptors.CalcMolFormula(mol),
            Descriptors.ExactMolWt(mol),
        )
    except Exception:
        return None, None


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    targets = argv or ["2173", "1036", "1037", "1038"]
    rows: list[dict[str, object]] = []

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = {
            str(r.get("cmpd_id", "")).strip(): r for r in csv.DictReader(handle)
        }

    for cmpd_id in targets:
        row = csv_rows.get(cmpd_id, {})
        ocr = (row.get("smiles") or "").strip()
        decimer = (row.get("smiles_decimer") or "").strip()

        entry: dict[str, object] = {
            "cmpd_id": cmpd_id,
            "ocr_from_csv": ocr,
            "decimer_from_csv": decimer,
        }
        for label, smiles in (("ocr", ocr), ("decimer", decimer)):
            formula, mass = formula_of(smiles)
            entry[f"{label}_local_formula"] = formula
            entry[f"{label}_local_exact_mass"] = round(mass, 4) if mass else ""
            hit = query_by_smiles(smiles) if smiles else {"found": False}
            entry[f"{label}_pubchem_found"] = hit.get("found", False)
            entry[f"{label}_pubchem_cid"] = hit.get("cid", "")
            entry[f"{label}_pubchem_formula"] = hit.get("formula", "")
            entry[f"{label}_pubchem_exact_mass"] = hit.get("exact_mass", "")
            entry[f"{label}_pubchem_inchikey"] = hit.get("inchikey", "")
            print(
                f"{cmpd_id} {label:8s}: local={formula} "
                f"pubchem_found={hit.get('found', False)} "
                f"{hit.get('formula', '')} {hit.get('exact_mass', '')}"
            )
        rows.append(entry)
        print()

    out_csv = OUT / "pubchem_crosscheck.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
