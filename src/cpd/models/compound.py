"""Compound + bioactivity schemas with JSONL / CSV serialization."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Measurement = Literal["Kd", "IC50", "EC50", "Ki", "MIC", "Kd_app"]
Unit = Literal["pM", "nM", "uM", "mM"]


class Bioactivity(BaseModel):
    """One bioactivity measurement, e.g. target=KRAS G12V-GDP, Kd=2.3 nM."""

    target: str = Field(description="e.g. 'KRAS G12V-GDP'")
    measurement: Measurement
    value: float
    unit: Unit


class Compound(BaseModel):
    """A single molecule entry from a patent.

    `compound_id` is whatever the patent uses (e.g. 'I-42', 'Compound 12').
    `smiles` is canonical when produced by RDKit; `sequence` is the cyclic
    peptide linear form (e.g. 'cyclo(-Ala-D-Pro-Phe-)') when applicable.
    """

    patent_id: str
    compound_id: str
    smiles: str | None = None
    sequence: str | None = None
    bioactivity: list[Bioactivity] = Field(default_factory=list)
    source_page: int | None = None
    notes: str | None = None


# ---------- serialization helpers ----------

def write_jsonl(compounds: list[Compound], path: Path) -> None:
    """Write one JSON object per line. Path parent dirs are created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in compounds:
            f.write(c.model_dump_json())
            f.write("\n")


def read_jsonl(path: Path) -> list[Compound]:
    with path.open("r", encoding="utf-8") as f:
        return [Compound.model_validate_json(line) for line in f if line.strip()]


def write_csv(compounds: list[Compound], path: Path) -> None:
    """Flat CSV: one row per (compound, bioactivity) pair.

    Compounds without bioactivity get a single row with empty activity fields.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patent_id", "compound_id", "smiles", "sequence", "source_page", "notes",
        "activity_target", "activity_measurement", "activity_value", "activity_unit",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in compounds:
            base = {
                "patent_id": c.patent_id, "compound_id": c.compound_id,
                "smiles": c.smiles or "", "sequence": c.sequence or "",
                "source_page": c.source_page if c.source_page is not None else "",
                "notes": c.notes or "",
                "activity_target": "", "activity_measurement": "",
                "activity_value": "", "activity_unit": "",
            }
            if not c.bioactivity:
                w.writerow(base)
                continue
            for b in c.bioactivity:
                row = dict(base)
                row["activity_target"] = b.target
                row["activity_measurement"] = b.measurement
                row["activity_value"] = b.value
                row["activity_unit"] = b.unit
                w.writerow(row)
