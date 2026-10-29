"""Task 1 - separate three different 'status' flags that were conflated.

The audit report ``ocr_compare_summary.txt`` prints a column headed
"CER usable" whose value is ``yes`` for cmpd 1036 and 1037 while the same
row also prints ``verdict=mismatch``. That reads as a contradiction and
invites the conclusion that the mass gate passed. It did not.

The column is a TRANSCRIPTION QC flag, not a mass-gate result:

* ``transcription_admissible``  |delta| <= 5 Da under the mass gate. A
  larger delta means roughly a whole residue (100-200 Da) is missing or
  misread, so the string cannot serve as CER ground truth. It says nothing
  about whether the mass gate accepts the structure.
* ``mass_gate_verdict``         the patent's own m/z check at the tolerance
  implied by the printed decimal places.
* ``structure_verified``        never set by this project. Requires a human
  to confirm the printed cell character by character AND the chemistry
  independently.

This script recomputes all three for every reference and writes a corrected
table. It changes no official CSV and adjusts no tolerance.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from cpd.ms_validation import validate_against_ms  # noqa: E402
from phase2_ocr_compare import REFERENCES  # noqa: E402

PHASE2 = REPO / "data" / "audit" / "phase2"
OUT = PHASE2 / "recognizer_candidates"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"

TRANSCRIPTION_QC_TOL_DA = 5.0

# Only 1036 was transcribed by the researcher and read back against the
# printed cell. The others are agent transcriptions still awaiting a human
# reader. This flag is about the CHARACTERS only, never about chemistry.
HUMAN_VERIFIED_TEXT = {"1036"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = {r["cmpd_id"]: r for r in csv.DictReader(fh)}

    recs: list[dict] = []
    for cid in sorted(REFERENCES):
        info = REFERENCES[cid]
        ref = info["smiles"]
        if cid not in rows:
            continue
        row = rows[cid]
        mz = float(row["ms_mz"])
        pol = (row.get("ms_polarity") or "").strip()
        gate = validate_against_ms(ref, mz, pol)
        # validate_against_ms reports the two masses, not the difference:
        #   observed_mass = mass computed from the SMILES
        #   expected_mass = neutral mass implied by the printed m/z
        obs = gate.get("observed_mass")
        exp = gate.get("expected_mass")
        delta = (
            float(obs) - float(exp)
            if obs is not None and exp is not None
            else None
        )
        admissible = (
            delta is not None and abs(delta) <= TRANSCRIPTION_QC_TOL_DA
        )
        recs.append(
            {
                "cmpd_id": cid,
                "reference_len": len(ref),
                "ms_mz": mz,
                "ms_polarity": pol,
                "formula": gate.get("formula", ""),
                "expected_mass": gate.get("expected_mass", ""),
                "observed_mass": gate.get("observed_mass", ""),
                "delta_da": round(delta, 4) if delta is not None else "",
                "error_ppm": gate.get("error_ppm", ""),
                "tolerance_ppm": gate.get("tolerance_ppm", ""),
                "mass_gate_verdict": gate.get("verdict", ""),
                "transcription_admissible": "yes" if admissible else "no",
                "text_reference_human_verified": (
                    "yes" if cid in HUMAN_VERIFIED_TEXT else "no"
                ),
                "rdkit_parses": "yes" if Chem.MolFromSmiles(ref) else "no",
                "structure_verified": "no",
                "reference_provenance": info.get("verified_by", ""),
            }
        )

    with (OUT / "reference_status_clarified.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {OUT/'reference_status_clarified.csv'}")

    L: list[str] = []
    add = L.append
    add("Phase 2 Task 1 - reference status, three flags kept separate")
    add("")
    add("These three columns answer different questions and must never be")
    add("collapsed into one another:")
    add("")
    add("  transcription_admissible  can this string be used as CER ground")
    add("                            truth? (|delta| <= 5 Da)")
    add("  text_reference_human_ver  did a person read the printed cell back")
    add("                            character by character? (CHARACTERS only)")
    add("  mass_gate_verdict         does the patent m/z check accept it?")
    add("  structure_verified        has a human confirmed the chemistry?")
    add("                            (always 'no' in this project)")
    add("")
    add(
        f"{'cmpd':>5s} {'len':>4s} {'dDa':>9s} {'ppm':>8s} {'tol ppm':>8s} "
        f"{'mass gate':>10s} {'transcr':>8s} {'txt-ok':>7s} {'struct':>7s}"
    )
    for r in recs:
        add(
            f"{r['cmpd_id']:>5s} {r['reference_len']:4d} "
            f"{str(r['delta_da']):>9s} {str(r['error_ppm']):>8s} "
            f"{str(r['tolerance_ppm']):>8s} "
            f"{r['mass_gate_verdict']:>10s} "
            f"{r['transcription_admissible']:>8s} "
            f"{r['text_reference_human_verified']:>7s} "
            f"{r['structure_verified']:>7s}"
        )
    add("")
    add("CMPD 1037 - the specific question")
    add("")
    r37 = next((r for r in recs if r["cmpd_id"] == "1037"), None)
    if r37:
        add(f"  ms_mz in CSV          = {r37['ms_mz']}  (not 1592.8)")
        add(f"  adduct                = {r37['ms_polarity']}")
        add(f"  formula of reference  = {r37['formula']}")
        add(f"  expected neutral mass = {r37['expected_mass']}")
        add(f"  observed neutral mass = {r37['observed_mass']}")
        add(f"  delta                 = {r37['delta_da']} Da")
        add(f"  error                 = {r37['error_ppm']} ppm")
        add(f"  tolerance             = {r37['tolerance_ppm']} ppm")
        add(f"  MASS GATE VERDICT     = {r37['mass_gate_verdict'].upper()}")
        add("")
        add("  1037 was never recorded as mass-gate passing. The 'yes' in")
        add("  the earlier 'CER usable' column is transcription_admissible,")
        add("  a |delta| <= 5 Da criterion, and is unrelated to the verdict.")
        add("")
        add("  Any earlier claim of a mass-gate pass for 1037 is retracted.")
        add("  No tolerance was changed and no CSV value was modified.")
    add("")
    add("PROVENANCE - kept distinct from chemistry")
    add("")
    for r in recs:
        add(f"  {r['cmpd_id']}: {r['reference_provenance']}")
    add("")
    add("  A reference may be 'transcribed from the printed cell' while")
    add("  structure_verified is still 'no'. Character-level transcription")
    add("  and chemical verification are separate claims.")
    (OUT / "reference_status_clarified.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT/'reference_status_clarified.txt'}")
    print()
    print("\n".join(L))


if __name__ == "__main__":
    main()
