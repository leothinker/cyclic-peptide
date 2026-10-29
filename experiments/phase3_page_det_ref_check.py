r"""Character-level check of the page-detection arms against references.

The corpus screen is reference-free: it can show that a configuration
recovers the printed rows and balances the parentheses, but it cannot
say whether the characters are right. This script closes that gap for
the references that exist, and is explicit about which references may
be used for what.

Reference status
----------------
    1036           human-verified transcription. Usable as a text
                   reference.
    1037, 1038,
    1040, 2173     agent transcriptions. They carry known misreads, so
                   a non-zero edit distance against them is not
                   evidence that the OCR is wrong; they are reported
                   for completeness only.

No candidate is ever selected by its mass gate, by RDKit
parseability, or by DECIMER. The reference strings decide.

Read-only. Writes to ``data/audit/phase3/page_det_ref_check/``.

Run:
    .venv\Scripts\python.exe phase3_page_det_ref_check.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402

CSV_IN = (REPO / "data" / "audit" / "phase3" / "page_det_line_rec"
          / "page_det_line_rec.csv")
OUT = REPO / "data" / "audit" / "phase3" / "page_det_ref_check"

HUMAN_VERIFIED = {"1036"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8-sig")))

    lines: list[str] = []
    lines.append("=== character-level check against the printed "
                 "references ===")
    lines.append("")
    lines.append("edit distance and CER against the reference text.")
    lines.append("CER = edit / len(reference).")
    lines.append("")
    lines.append("1036 is a human-verified transcription. The others are")
    lines.append("agent transcriptions with known misreads, so their")
    lines.append("numbers are indicative only and are marked as such.")
    lines.append("")

    table: list[dict] = []
    for cmpd_id, ref_obj in REFERENCES.items():
        ref = ref_obj["smiles"]
        verified = cmpd_id in HUMAN_VERIFIED
        for arm in sorted({r["arm"] for r in rows}):
            hit = [r for r in rows
                   if r["cmpd_id"] == cmpd_id and r["arm"] == arm]
            if not hit:
                continue
            r = hit[0]
            txt = r["text"]
            edit = edit_ops(ref, txt)
            table.append({
                "cmpd_id": cmpd_id,
                "reference_status": "human_verified" if verified
                                   else "agent_transcription",
                "arm": arm,
                "ref_len": len(ref),
                "ocr_len": len(txt),
                "edit": edit,
                "cer": round(edit / len(ref), 5),
                "rows": r["n_rows"],
                "expected_rows": r["expected_rows"],
                "rows_match": r["rows_match"],
                "balanced": r["balanced"],
                "parses": r["parses"],
                "strict_pass": r["strict_pass"],
            })

    hdr = (f"{'cmpd':<6} {'ref status':<19} {'arm':<24} "
           f"{'edit':>5} {'cer':>8} {'rows':>7} {'bal':>5} {'strict':>7}")
    lines.append(hdr)
    for t in table:
        lines.append(
            f"{t['cmpd_id']:<6} {t['reference_status']:<19} "
            f"{t['arm']:<24} {t['edit']:>5} {t['cer']:>8.5f} "
            f"{t['rows']:>3}/{t['expected_rows']:<3} "
            f"{'yes' if t['balanced'] else 'no':>5} "
            f"{'yes' if t['strict_pass'] else 'no':>7}")

    with (OUT / "page_det_ref_check.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    io.open(OUT / "page_det_ref_check.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
