r"""Where the page route still goes wrong, and why.

The page route with the default recogniser reproduces both
human-verified references character for character, but only 190 of 335
cells come out with balanced parentheses. A correct reading of a
printed SMILES always balances, so the 145 remaining cells contain
real reading errors. This script says what kind.

Three strings are compared per cell:

    csv     the shipped pipeline output (read from the official CSV)
    page    whole-page recognition, default engine, rows attributed
            inside the production SMILES column
    recrop  page detection, then every row cropped out of the 2x page
            and recognised on its own

and each cell is classified by which pair agrees and which strings
balance. Example diffs are printed so the error pattern can be read
rather than guessed.

Read-only. Writes to ``data/audit/phase3/residual_errors/``.

Run:
    .venv\Scripts\python.exe phase3_route_residual_errors.py
"""

from __future__ import annotations

import csv
import difflib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import edit_ops  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "residual_errors"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
RECROP = (REPO / "data" / "audit" / "phase3" / "page_det_line_rec"
          / "page_det_line_rec.csv")
PAGE = (REPO / "data" / "audit" / "phase3" / "page_route_ref"
        / "page_route_ref.csv")

RECROP_ARM = "pdet_def_rec_def_2x"


def _read(path):
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def _bal(t):
    return t.count("(") == t.count(")") and t.count("[") == t.count("]")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = list(csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")))
    csv_map = {r["cmpd_id"]: (r.get("smiles") or "") for r in df}

    recrop = {r["cmpd_id"]: r for r in _read(RECROP)
              if r["arm"] == RECROP_ARM}
    page = {r["cmpd_id"]: r for r in _read(PAGE)}

    lines = []
    lines.append("=== residual errors of the page route ===")
    lines.append("")
    lines.append(f"recrop arm: {RECROP_ARM}")
    lines.append("")
    lines.append(f"{'group':<34} {'n':>4}")
    groups = {}
    rows = []
    for cid in sorted(recrop):
        rc = recrop[cid]
        rt = rc["text"]
        ct = csv_map.get(cid, "")
        pt = page.get(cid, {}).get("text", "")
        same_rc = rt == ct
        same_rp = bool(pt) and rt == pt
        if _bal(rt) and _bal(ct):
            g = "both balanced"
        elif _bal(rt) and not _bal(ct):
            g = "recrop balanced, csv not"
        elif not _bal(rt) and _bal(ct):
            g = "csv balanced, recrop not"
        else:
            g = "neither balanced"
        groups[g] = groups.get(g, 0) + 1
        rows.append({"cmpd_id": cid, "group": g, "recrop_len": len(rt), "recrop_open": rt.count("("), "recrop_close": rt.count(")"), "recrop_balanced": _bal(rt), "csv_len": len(ct), "csv_balanced": _bal(ct), "page_len": len(pt), "page_balanced": _bal(pt) if pt else "", "recrop_eq_csv": same_rc, "recrop_eq_page": same_rp, "edit_recrop_csv": edit_ops(rt, ct)})

    for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        lines.append(f"{g:<34} {n:>4}")
    lines.append("")
    agree = sum(1 for r in rows if r["recrop_eq_csv"])
    lines.append(f"recrop identical to the shipped CSV : {agree}/{len(rows)}")
    agree_p = sum(1 for r in rows if r["recrop_eq_page"])
    have_page = sum(1 for r in rows if r["page_len"])
    lines.append(f"recrop identical to the whole-page route: {agree_p}/{have_page} (reference compounds only)")
    lines.append("")
    lines.append("--- examples: neither balanced (recrop errors) ---")
    shown = 0
    for r in rows:
        if r["group"] != "neither balanced":
            continue
        cid = r["cmpd_id"]
        rt = recrop[cid]["text"]
        ct = csv_map.get(cid, "")
        lines.append(f"[{cid}] recrop len {len(rt)} ({r['recrop_open']}/{r['recrop_close']})  csv len {len(ct)}")
        bal = 0
        first_bad = None
        for i, ch in enumerate(rt):
            if ch == "(":
                bal += 1
            elif ch == ")":
                bal -= 1
            if bal < 0 and first_bad is None:
                first_bad = i
        if first_bad is not None:
            lo = max(0, first_bad - 30)
            lines.append(f"    balance first breaks at {first_bad}: {rt[lo:first_bad + 30]!r}")
        else:
            lines.append(f"    closing brackets exceed opening ones by {r['recrop_close'] - r['recrop_open']}")
        lines.append(f"    tail: {rt[-40:]!r}")
        shown += 1
        if shown >= 5:
            break
    lines.append("")
    lines.append("The csv column is the shipped pipeline output, which uses")
    lines.append("the English PP-OCRv4 recogniser; the recrop column uses the")
    lines.append("default recogniser. A difference here is a recogniser")
    lines.append("difference, not a detection difference.")
    lines.append("")
    lines.append("Text level. No structure is verified.")

    with (OUT / "residual_errors.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    io.open(OUT / "residual_errors.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
