"""Summarise the deliverable CSV of every configured patent.

Prints the numbers that go into the progress report, computed from the
file itself, so that nothing in the report is a remembered figure.

The patents are found through the profile registry rather than through
a hard-coded list, so a third patent is covered as soon as its
profile exists and its CSV has been built.

Usage::

    .venv\\Scripts\\python.exe analysis/verify_deliverable.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from cpd.parsers.profiles import available_profiles, get_profile
from cpd.postprocess import bracket_deficit

ACTIVITY_FIELDS = ("kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity")


def _balanced(smiles: str) -> bool:
    """Whether both bracket kinds are balanced.

    ``cpd.postprocess.bracket_deficit`` counts the surplus of *opening*
    brackets and clamps it at zero, because the failure it was written
    for is a bracket the recogniser lost.  A string that is one closing
    bracket too many therefore scores zero there -- which is not the
    same as being balanced.  This check looks at both directions.
    """
    if not smiles:
        return False
    return (
        smiles.count("(") == smiles.count(")")
        and smiles.count("[") == smiles.count("]")
    )


def _patent_csv(patent: str) -> Path:
    """The deliverable CSV of one patent."""
    if patent == "WO2025162428":
        return ROOT / "data" / "processed" / "compounds_combined.csv"
    return ROOT / "data" / "processed" / patent / "compounds_combined.csv"


def main() -> int:
    for patent in available_profiles():
        print("=" * 72)
        print(f"### {patent}")
        print("=" * 72)
        _verify_one(patent)
        print()
    return 0


def _verify_one(patent: str) -> int:
    path = _patent_csv(patent)
    if not path.exists():
        print(f"no deliverable for {patent}: {path}")
        return 0
    archive_path = (
        ROOT / "data" / "processed" / "archive"
        / "compounds_combined_production_route.csv"
    )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0].keys())
    profile = get_profile(patent)
    grade_fields = list(profile.extra_assay_fields)

    with_text = [r for r in rows if (r.get("smiles") or "").strip()]
    print(f"file            : {path}")
    print(f"rows            : {len(rows)}")
    print(f"columns         : {len(columns)}")
    print(f"unique cmpd_id  : {len({r['cmpd_id'] for r in rows})}")
    print(f"route           : {profile.smiles_route}")

    for field in ("kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity"):
        filled = sum(1 for r in rows if (r.get(field) or "").strip())
        print(f"  {field:<14} filled {filled}/{len(rows)}")
    for field in grade_fields:
        filled = sum(1 for r in rows if (r.get(field) or "").strip())
        print(f"  {field:<14} filled {filled}/{len(rows)}")

    print()
    print("printed SMILES")
    print(f"  read                : {len(with_text)}")
    print(f"  RDKit-parseable     : {sum(1 for r in with_text if r['is_valid'] == 'True')}")
    print(
        "  bracket-balanced    : "
        f"{sum(1 for r in with_text if _balanced(r['smiles']))}"
    )
    print(
        "  opener-deficit dist.: "
        + ", ".join(
            f"{d}:{n}"
            for d, n in sorted(Counter(bracket_deficit(r["smiles"]) for r in with_text).items())
        )
    )

    print()
    print("smiles_best source   :")
    for key, count in Counter(r.get("smiles_best_source") or "-" for r in rows).most_common():
        print(f"  {key:<20} {count:4d}")
    print("structure_confidence :")
    for key, count in Counter(r.get("structure_confidence") or "-" for r in rows).most_common():
        print(f"  {key:<20} {count:4d}")
    print("ms_check_verdict     :")
    for key, count in Counter(r.get("ms_check_verdict") or "-" for r in rows).most_common():
        print(f"  {key:<20} {count:4d}")
    print("data_quality         :")
    for key, count in Counter(r.get("data_quality") or "-" for r in rows).most_common():
        print(f"  {key:<20} {count:4d}")
    print("ocr_difficulty       :")
    for key, count in Counter(r.get("ocr_difficulty") or "-" for r in rows).most_common():
        print(f"  {key:<20} {count:4d}")
    print("repair_review_status :")
    for key, count in Counter(r.get("repair_review_status") or "-" for r in rows).most_common():
        print(f"  {key:<20} {count:4d}")

    print()
    decimer_rows = [r for r in rows if (r.get("smiles_decimer") or "").strip()]
    print(f"DECIMER output       : {len(decimer_rows)} rows")
    agree = sum(1 for r in rows if r.get("smiles_agree") == "True")
    print(f"OCR/DECIMER agree    : {agree}")

    # ---- regression check against the archived extraction -------------
    #
    # The archived CSV is the extraction that was shipped before the
    # printed-SMILES route changed.  Its activity fields are the
    # reference: a rebuild must not lose any of them, and must not
    # change a value that the reference already had.  A drop here is a
    # regression in the reader, not in the data.
    #
    # The archive belongs to the first patent only; the second patent
    # never had a production-route CSV.
    if patent != "WO2025162428":
        return 0
    if not archive_path.exists():
        print()
        print(f"no archive to compare with: {archive_path}")
        return 0
    with archive_path.open(newline="", encoding="utf-8-sig") as handle:
        archive = {r["cmpd_id"]: r for r in csv.DictReader(handle)}
    current = {r["cmpd_id"]: r for r in rows}
    print()
    print("activity fields vs the archived extraction")
    for field in ACTIVITY_FIELDS:
        ref_filled = {
            cid for cid, r in archive.items() if (r.get(field) or "").strip()
        }
        cur_filled = {
            cid for cid, r in current.items() if (r.get(field) or "").strip()
        }
        lost = sorted(ref_filled - cur_filled)
        changed = sorted(
            cid
            for cid in ref_filled & cur_filled
            if (archive[cid].get(field) or "").strip()
            != (current[cid].get(field) or "").strip()
        )
        print(
            f"  {field:<14} archived {len(ref_filled):>4}  "
            f"now {len(cur_filled):>4}  "
            f"lost {len(lost):>4}  changed {len(changed):>4}"
        )
        for cid in lost[:5]:
            print(f"      lost: {cid}")
        for cid in changed[:5]:
            print(
                f"      changed: {cid} "
                f"{archive[cid].get(field)!r} -> {current[cid].get(field)!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

