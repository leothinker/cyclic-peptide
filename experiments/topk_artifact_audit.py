r"""Audit the orphaned demo_rapidocr_topk artifact.

topk_printed_smiles.json is the only surviving record of a page-level
detection + per-character top-k extraction that was never evaluated against
the human-verified references. The script that produced it is gone, so the
first job is to establish what the file actually contains and whether its
claims can be trusted.

Read-only. Writes one report file next to the pre-report validation set.
"""

from __future__ import annotations

import difflib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402

ART = REPO / "demo_rapidocr_topk" / "topk_printed_smiles.json"
OUT = (REPO / "data" / "audit" / "phase2" / "pre_report_validation"
       / "topk_artifact_audit.txt")

# 1036 printed in 7 rows, 1037 in 7 rows; the artifact stores page-level
# line numbers, not row indices.
GROUPS = {"1036": slice(0, 7), "1037": slice(7, 13)}


def main() -> None:
    data = json.load(ART.open(encoding="utf-8"))
    engs = list(data)
    out = io.open(OUT, "w", encoding="utf-8")

    out.write("=== orphaned topk artifact audit ===\n\n")

    # ---- 1. are the two engines actually independent runs?
    a, b = data["ch_PP-OCRv4"], data["en_PP-OCRv4"]
    same_text = all(x["text"] == y["text"] for x, y in zip(a, b))
    same_char = all(
        json.dumps(x["chars"], sort_keys=True)
        == json.dumps(y["chars"], sort_keys=True)
        for x, y in zip(a, b))
    out.write("1. ENGINE INDEPENDENCE\n")
    out.write(f"   engines present          : {engs}\n")
    out.write(f"   texts identical          : {same_text}\n")
    out.write(f"   per-char top3 identical  : {same_char}\n")
    stray = [(x["line"], x["engine"]) for x in a if "engine" in x]
    out.write(f"   stray 'engine' keys      : {stray}\n")
    if same_text and same_char:
        out.write("   => the two sections are byte-identical; they cannot be\n"
                  "      two independent recognition runs. Treat whichever\n"
                  "      engine label is present as UNVERIFIED.\n")

    # ---- 2. per-compound comparison against the verified references
    out.write("\n2. RECONSTRUCTION vs HUMAN-VERIFIED REFERENCE\n")
    base = {}
    import csv
    fb = (REPO / "data/audit/phase2/pre_report_validation"
          / "ab_fullstring.csv")
    for r in csv.DictReader(fb.open(encoding="utf-8-sig")):
        if r["cmpd_id"] == "1036" and r["pad_px"] == "0":
            base = r
    for cid, sl in GROUPS.items():
        rows = a[sl]
        txt = "".join(r["text"] for r in rows)
        ref = REFERENCES.get(cid, {}).get("smiles")
        if not ref:
            continue
        ed = edit_ops(ref, txt)
        out.write(f"\n   cmpd {cid}: rows={len(rows)} joined={len(txt)} "
                  f"ref={len(ref)}\n")
        out.write(f"   edit_distance = {ed}   CER = {ed / len(ref):.5f}\n")
        out.write(f"   page lines present : {[r['line'] for r in rows]}\n")
        gaps = [n for n in range(min(r['line'] for r in rows),
                                 max(r['line'] for r in rows) + 1)
                if n not in [r["line"] for r in rows]]
        out.write(f"   page lines absent  : {gaps}\n")
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                a=ref, b=txt).get_opcodes():
            if tag == "equal":
                continue
            out.write(f"     {tag:<8} ref[{i1}:{i2}]={ref[i1:i2]!r} "
                      f"got={txt[j1:j2]!r}\n")
    if base:
        out.write(f"\n   production baseline, cmpd 1036 (pad 0): "
                  f"len={base['text_len']} "
                  f"edit={base['edit_distance_vs_ref']} "
                  f"CER={base['cer_vs_ref']}\n")

    # ---- 3. does the row-start clip hit other characters than '='?
    out.write("\n3. WHICH CHARACTERS GET CLIPPED AT ROW START\n")
    for cid, sl in GROUPS.items():
        ref = REFERENCES.get(cid, {}).get("smiles")
        rows = a[sl]
        if not ref:
            continue
        out.write(f"   cmpd {cid}: row-by-row alignment vs reference is only\n"
                  "   meaningful while the row count matches; 1037 is missing a\n"
                  "   whole row so every offset after it is shifted and any\n"
                  "   per-row diff here would be an artefact, not evidence.\n")
        break

    # ---- 4. internal consistency: does top3[0] agree with the chosen char?
    out.write("\n4. INTERNAL CONSISTENCY OF THE PER-CHAR top3 FIELD\n")
    out.write("   a trustworthy row must satisfy top3[0].char == text[i]\n")
    bad = []
    for r in a:
        chars = r.get("chars", [])
        mism = [c["pos"] for c in chars
                if not c.get("top3")
                or c["top3"][0]["char"] != c["char"]]
        agree = len(chars) - len(mism)
        flag = "" if not mism else f"  <-- {len(mism)} MISMATCHED"
        out.write(f"   pageline {r['line']:>2}: {agree}/{len(chars)} "
                  f"positions agree{flag}\n")
        if mism:
            bad.append(r["line"])
            # whose top3 is it, really?
            donor = [q["text"][:8] for q in a
                     if q["line"] != r["line"]
                     and q["chars"]
                     and q["chars"][0]["top3"][0]["char"]
                     == chars[0]["top3"][0]["char"]]
            out.write(f"      pos0 chosen={chars[0]['char']!r} but "
                      f"top3[0]={chars[0]['top3'][0]['char']!r}; "
                      f"rows starting with that char: {donor}\n")
    out.write(f"   rows with inconsistent top3: {bad}\n")
    out.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
