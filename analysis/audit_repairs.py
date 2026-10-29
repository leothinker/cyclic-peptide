"""Audit every macrocycle repair in the corpus and emit a tiered report.

Run from the repo root::

    .venv/Scripts/python.exe analysis/audit_repairs.py

Writes two files next to this script:

* ``repair_audit.json``   -- one record per compound with its repair tier,
  invariants and reason, plus the aggregate counts.
* ``repair_audit.csv``   -- the same records flattened for spreadsheets.

The point of this script is to replace the single headline
"macrocycle recovery rate" with a defensible breakdown.  A repair that
invents atoms is counted separately from one that only balances a bracket,
because only the latter is evidence that DECIMER's graph was correct.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# RDKit logs every failed parse attempt to stderr.  The repair strategies
# deliberately try hundreds of malformed candidates per compound, so this
# script would otherwise emit tens of thousands of lines of noise. The
# audit cares about the *outcome*, not about each rejected attempt.
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from cpd.macrocycle_repair import repair_macrocycle_with_method  # noqa: E402
from cpd.render import MACROCYCLE_MIN_SIZE, is_macrocycle_intact  # noqa: E402
from cpd.repair_audit import (  # noqa: E402
    MIN_AMIDE_COUNT,
    PLAUSIBLE_MACROCYCLE_RANGE,
    audit_repair,
    fingerprint,
)

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _clean(value: str | None) -> str:
    return (value or "").strip()


def audit_row(row: dict) -> dict:
    """Audit one CSV row. Returns a flat record."""
    cmpd = _clean(row.get("cmpd_id"))
    decimer = _clean(row.get("smiles_decimer"))
    cmpd_id = int(cmpd) if cmpd.isdigit() else None

    record: dict = {
        "cmpd_id": cmpd,
        "decimer_len": len(decimer),
        "decimer_parses": bool(decimer) and fingerprint(decimer) is not None,
        "decimer_macrocycle_intact": is_macrocycle_intact(decimer),
        "repair_smiles": "",
        "repair_method": "",
        "tier": "not_attempted",
        "reason": "",
        "atoms_delta": "",
        "atoms_invented": "",
        "atoms_deleted": "",
        "amides_delta": "",
        "stereo_delta": "",
        "formula_before": "",
        "formula_after": "",
        "largest_ring": "",
        "num_rings": "",
        "amides": "",
        "stereo_centres": "",
    }

    if not decimer:
        record["tier"] = "no_decimer_output"
        record["reason"] = "empty_smiles_decimer"
        return record

    if is_macrocycle_intact(decimer):
        record["tier"] = "decimer_intact"
        record["reason"] = "macrocycle_>=%d" % MACROCYCLE_MIN_SIZE
        fp = fingerprint(decimer)
        if fp:
            record.update(
                formula_after=fp["formula"],
                largest_ring=fp["largest_ring"],
                num_rings=fp["rings"],
                amides=fp["amides"],
                stereo_centres=fp["stereo_centres"],
            )
        return record

    repaired, method = repair_macrocycle_with_method(decimer)
    if repaired is None:
        record["tier"] = "unrepaired"
        record["reason"] = "all_6_strategies_failed"
        return record

    audit = audit_repair(decimer, repaired, method or "", cmpd_id)
    record["repair_smiles"] = repaired
    record["repair_method"] = method or ""
    record["tier"] = audit.tier
    record["reason"] = audit.reason or "accepted"
    record["atoms_delta"] = audit.atoms_delta
    record["atoms_invented"] = audit.atoms_invented
    record["atoms_deleted"] = audit.atoms_deleted
    record["amides_delta"] = audit.amides_delta
    record["stereo_delta"] = audit.stereo_delta
    if audit.before:
        record["formula_before"] = audit.before["formula"]
    if audit.after:
        record.update(
            formula_after=audit.after["formula"],
            largest_ring=audit.after["largest_ring"],
            num_rings=audit.after["rings"],
            amides=audit.after["amides"],
            stereo_centres=audit.after["stereo_centres"],
        )
    if audit.notes:
        record["reason"] = (record["reason"] + ";" + ",".join(audit.notes)).strip(";")
    return record


def summarise(records: list[dict]) -> dict:
    total = len(records)
    tiers = Counter(r["tier"] for r in records)
    methods = Counter(
        r["repair_method"] for r in records if r["repair_method"]
    )
    validated = sum(1 for r in records if r["tier"] in ("high", "medium"))
    accepted = sum(
        1 for r in records if r["tier"] in ("high", "medium", "low", "experimental")
    )
    intact = tiers.get("decimer_intact", 0)
    raw_recovery = (intact + accepted) / total if total else 0.0
    strict_recovery = (intact + validated) / total if total else 0.0
    return {
        "total_compounds": total,
        "decimer_intact": intact,
        "repair_validated": validated,
        "repair_accepted_not_validated": accepted - validated,
        "still_unrepaired": tiers.get("unrepaired", 0),
        "no_decimer_output": tiers.get("no_decimer_output", 0),
        "tier_counts": dict(tiers),
        "method_counts": dict(methods),
        "apparent_recovery_rate": round(raw_recovery, 4),
        "strict_recovery_rate": round(strict_recovery, 4),
        "gates": {
            "macrocyle_min_size": MACROCYCLE_MIN_SIZE,
            "plausible_macrocyle_range": list(PLAUSIBLE_MACROCYCLE_RANGE),
            "min_amide_count": MIN_AMIDE_COUNT,
        },
    }


def print_report(summary: dict, records: list[dict]) -> None:
    total = summary["total_compounds"]
    print("=" * 72)
    print("MACROCYCLE REPAIR AUDIT")
    print("=" * 72)
    print(f"total compounds          : {total}")
    print(f"DECIMER macrocycle intact: {summary['decimer_intact']}")
    print(
        f"repair validated         : {summary['repair_validated']}"
        f"   (tier high/medium)"
    )
    print(
        f"repair accepted, unproven: {summary['repair_accepted_not_validated']}"
        f"   (tier low/experimental)"
    )
    print(f"still unrepaired         : {summary['still_unrepaired']}")
    print(f"no DECIMER output        : {summary['no_decimer_output']}")
    print()
    print(
        f"apparent recovery (all accepted) : "
        f"{summary['apparent_recovery_rate'] * 100:.1f}%"
    )
    print(
        f"STRICT recovery (validated only) : "
        f"{summary['strict_recovery_rate'] * 100:.1f}%"
    )
    print()
    print("-- tier counts --")
    for tier, n in sorted(
        summary["tier_counts"].items(), key=lambda kv: -kv[1]
    ):
        print(f"  {tier:16s} {n:4d}  ({n * 100 / total:5.1f}%)")
    print("-- repair method counts --")
    for method, n in sorted(
        summary["method_counts"].items(), key=lambda kv: -kv[1]
    ):
        print(f"  {method:24s} {n:4d}")
    print()
    print("-- validated repairs by method --")
    for method, n in sorted(Counter(
        r["repair_method"]
        for r in records
        if r["repair_method"] and r["tier"] in ("high", "medium")
    ).items(), key=lambda kv: -kv[1]):
        print(f"  {method:24s} {n:4d}")
    print("=" * 72)


def main() -> int:
    rows = _load_rows(CSV_PATH)
    records = [audit_row(row) for row in rows]
    summary = summarise(records)

    out_json = Path(__file__).resolve().parent / "repair_audit.json"
    out_json.write_text(
        json.dumps({"summary": summary, "records": records}, indent=1),
        encoding="utf-8",
    )

    out_csv = Path(__file__).resolve().parent / "repair_audit.csv"
    fields = list(records[0].keys()) if records else []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print_report(summary, records)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
