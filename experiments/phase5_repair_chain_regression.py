"""Regression check for the two repair-chain changes.

Two guards were added to the repair chain:

1. ``cpd.ocr_rewrite.rewrite_oxo`` no longer rewrites ``<digit>=O)``.
   The lactam of a cyclic peptide is printed as ``C1=O)``, where the
   ring-closure digit sits between the carbon and the ``=O``.  The
   shipped rule inserted a spurious bracket there.
2. ``cpd.chem.auto_repair_smiles`` reads a lone ``=0`` as ``=O``.
   Zero is not an element, and a ring-closure digit always has a
   partner, so a string with exactly one ``0`` has a misread oxygen.

This script measures both, on the real OCR text of both patents:
``routes_before.json`` holds the pre-change OCR output of the
``production`` and ``validated`` routes for all 364 printed cells.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_repair_chain_regression.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles, close_unbalanced_brackets  # noqa: E402
from cpd.ocr_rewrite import rewrite_oxo  # noqa: E402

from wo2022251372_references import REFERENCE_LINES, edit_distance  # noqa: E402

ROUTES = REPO / "data" / "audit" / "phase5" / "routes_before.json"
OLD_REFERENCES = REPO / "data" / "audit" / "phase2" / "ocr_references.json"
OLD_META = (
    REPO / "data" / "audit" / "phase2" / "cells_text" / "cells_text_meta.csv"
)
OUT = REPO / "data" / "audit" / "phase5" / "repair_chain_regression"


# The shipped rule, before the ring-closure guard was added.
_OXO = re.compile(r"(.)[-=](O\))")


def shipped_rewrite_oxo(text: str) -> tuple:
    """``rewrite_oxo`` without the ring-closure-digit guard."""
    count = 0

    def sub(match: "re.Match") -> str:
        nonlocal count
        if match.group(1) == "(":
            return match.group(0)
        count += 1
        return f"{match.group(1)}(=O)"

    return _OXO.sub(sub, text), count


def shipped_auto_repair(raw: str) -> str:
    """``auto_repair_smiles`` without the ``=0`` -> ``=O`` rule."""
    s = re.sub(r"[^\x00-\x7F]+", "", raw)
    s = re.sub(r"C\(O\)N", "C(=O)N", s)
    s = re.sub(r"C\(=[0O]\)?N", "C(=O)N", s)
    s = re.sub(r"C\(=0\)", "C(=O)", s)
    s = re.sub(r"(\[C@{1,2}H)\(", r"\1](", s)
    if s.startswith("H](") or s.startswith("@H]("):
        s = "CC[C@" + s
    elif s.startswith("[C@H](") and not s.startswith("CC[C@H]("):
        s = "CC" + s
    return s.strip()


def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _chain(raw: str, rewrite, repair) -> str:
    oxo, _ = rewrite(raw)
    repaired = repair(oxo)
    return close_unbalanced_brackets(repaired) or repaired


def _load_old_refs() -> dict[str, str]:
    data = json.loads(OLD_REFERENCES.read_text(encoding="utf-8"))
    out = {}
    for key, value in data.items():
        text = value if isinstance(value, str) else value.get("smiles")
        if text:
            out[key] = text
    return out


def _load_old_ids() -> dict[str, str]:
    mapping = {}
    with OLD_META.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cmpd_id = (row.get("cmpd_id") or "").strip()
            stem = Path(row.get("crop_path") or "").stem
            if cmpd_id and stem:
                mapping[stem] = cmpd_id
    return mapping


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    snapshot = json.loads(ROUTES.read_text(encoding="utf-8"))
    old_refs = _load_old_refs()
    old_ids = _load_old_ids()

    rows = []
    for key, values in sorted(snapshot.items()):
        patent, stem = key.split("/", 1)
        if patent == "new":
            compound_id = int(stem.split("cmpd")[-1])
            reference = "".join(REFERENCE_LINES.get(compound_id, ()))
        else:
            compound_id = old_ids.get(stem, "")
            reference = old_refs.get(compound_id, "")

        for route in ("production", "validated"):
            raw = values.get(route, "") or ""
            old_text = _chain(
                raw, shipped_rewrite_oxo, shipped_auto_repair
            )
            new_text = _chain(raw, rewrite_oxo, auto_repair_smiles)

            rows.append(
                {
                    "patent": patent,
                    "cell": stem,
                    "compound_id": compound_id,
                    "route": route,
                    "has_reference": bool(reference),
                    "shipped_parses": _parses(old_text),
                    "guarded_parses": _parses(new_text),
                    "shipped_edit": (
                        edit_distance(reference, old_text)
                        if reference
                        else ""
                    ),
                    "guarded_edit": (
                        edit_distance(reference, new_text)
                        if reference
                        else ""
                    ),
                    "shipped_text": old_text,
                    "guarded_text": new_text,
                }
            )

    with (OUT / "repair_chain_regression.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {}
    for patent in ("old", "new"):
        summary[patent] = {}
        for route in ("production", "validated"):
            subset = [
                r
                for r in rows
                if r["patent"] == patent and r["route"] == route
            ]
            with_ref = [r for r in subset if r["has_reference"]]
            entry = {
                "cells": len(subset),
                "shipped_parses": sum(1 for r in subset if r["shipped_parses"]),
                "guarded_parses": sum(
                    1 for r in subset if r["guarded_parses"]
                ),
                "cells_with_reference": len(with_ref),
                "shipped_edit": [r["shipped_edit"] for r in with_ref],
                "guarded_edit": [r["guarded_edit"] for r in with_ref],
            }
            if with_ref:
                entry["total_edit_shipped"] = sum(entry["shipped_edit"])
                entry["total_edit_guarded"] = sum(entry["guarded_edit"])
            summary[patent][route] = entry

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
