r"""Character-level check of the binarisation fix on the human reference.

The corpus-scale screen in ``phase3_scale_binarise.py`` is
reference-free: it can prove that the parenthesis balance is restored,
but it cannot say whether the text is right. This script closes that gap
for the one compound with a human-verified reference, 1036.

It measures, on the identical crop:

    production      trim -> 2x -> recognise          (the shipped path)
    production+th  trim -> 2x -> threshold 200 -> recognise

and reports the edit distance and CER against the reference for each,
together with the character-level diff, so that a character error that
merely moved (for instance a '(' replaced by another symbol) is visible
and not hidden inside an aggregate score.

The reference is a TEXT reference. Passing this check means the printed
string was read correctly; it does not mean the chemical structure was
verified, and it does not mean the mass gate passes (1036 fails it).

Read-only. Writes to ``data/audit/phase3/cer_1036/``.

Run:
    .venv\Scripts\python.exe phase3_cer_1036.py
"""

from __future__ import annotations

import csv
import difflib
import io
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import GOLD, baseline_engine  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "cer_1036"


def _recognise_joined(engine, img) -> tuple[str, list[str]]:
    result = engine(img)
    if not result:
        return "", []
    boxes = list(result.boxes)
    txts = list(result.txts)
    items = sorted(zip(boxes, txts),
                   key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
    parts = [t.strip() for _, t in items if t and t.strip()]
    return "".join(parts), parts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from cpd.parsers.table_extractor import RapidOcrTableExtractor

    ex = RapidOcrTableExtractor(rec_top_k=0)
    engine = baseline_engine()

    ref = REFERENCES["1036"]["smiles"]
    img = cv2.imread(str(GOLD / "PCTCN2025075389-ftappb-I100297.jpg"))

    # Production crop: img[row_y1:row_y2, smiles_col.x1:smiles_col.x2]
    cols = ex._detect_columns(img)
    col = next(c for c in cols if c.name == "smiles")
    rows = ex._detect_row_boundaries(img)

    lines: list[str] = []
    lines.append("=== 1036: character accuracy, production vs +binarise "
                 "===\n")
    lines.append("Reference: human-verified transcription of the printed "
                 "cell in I100297.")
    lines.append("This is a TEXT check. It does not verify the chemical "
                 "structure and does not affect the mass gate.\n")

    results = {}
    for band_idx, (y1, y2) in enumerate(rows):
        crop = img[y1:y2, col.x1:col.x2]
        if crop.size == 0:
            continue
        trimmed = ex._trim_to_content(crop)
        up = ex._upscale(trimmed)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        prod_txt, prod_parts = _recognise_joined(engine, up)
        bin_txt, bin_parts = _recognise_joined(
            engine, cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))

        results[band_idx] = {
            "y1": y1, "y2": y2,
            "prod": prod_txt, "prod_parts": prod_parts,
            "bin": bin_txt, "bin_parts": bin_parts,
        }

    # The SMILES column holds several compounds; pick the band whose
    # production text starts the same way the reference starts.
    target = None
    for band_idx, d in results.items():
        if d["prod"].startswith("CC[C@H]") or d["bin"].startswith("CC[C@H]"):
            target = (band_idx, d)
            break
    if target is None:
        lines.append("could not locate the 1036 band")
        io.open(OUT / "cer_1036.txt", "w",
                encoding="utf-8").write("\n".join(lines))
        print("\n".join(lines))
        return

    band_idx, d = target
    prod, binr = d["prod"], d["bin"]
    lines.append(f"1036 band index   : {band_idx} "
                 f"(page rows y={d['y1']}..{d['y2']})\n")
    lines.append(f"reference length  : {len(ref)}\n")
    lines.append(f"production length : {len(prod)}\n")
    lines.append(f"binarised length  : {len(binr)}\n\n")

    for label, txt in (("production", prod), ("binarised", binr)):
        ed = edit_ops(ref, txt)
        parts = d["prod_parts"] if label == "production" else d["bin_parts"]
        lines.append(f"{label:<11} edit={ed:<4} "
                     f"CER={ed / len(ref):.5f} "
                     f"open={txt.count('(')} close={txt.count(')')} "
                     f"rows={len(parts)}\n")

    lines.append("\n--- character-level diff, production ---\n")
    sm = difflib.SequenceMatcher(None, ref, prod)
    ops = sm.get_opcodes()
    n_diff = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag != "equal")
    lines.append(f"differing blocks: {sum(1 for op in ops if op[0] != 'equal')}"
                 f", differing chars: {n_diff}\n")
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        lines.append(f"  ref[{i1}:{i2}]={ref[i1:i2]!r} -> "
                     f"got[{j1}:{j2}]={prod[j1:j2]!r}\n")

    lines.append("\n--- character-level diff, binarised ---\n")
    sm = difflib.SequenceMatcher(None, ref, binr)
    ops = sm.get_opcodes()
    n_diff = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag != "equal")
    lines.append(f"differing blocks: {sum(1 for op in ops if op[0] != 'equal')}"
                 f", differing chars: {n_diff}\n")
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        lines.append(f"  ref[{i1}:{i2}]={ref[i1:i2]!r} -> "
                     f"got[{j1}:{j2}]={binr[j1:j2]!r}\n")

    lines.append("\n--- recognised rows ---\n")
    lines.append("production rows:\n")
    for i, p in enumerate(d["prod_parts"]):
        lines.append(f"  {i}: {p}\n")
    lines.append("binarised rows:\n")
    for i, p in enumerate(d["bin_parts"]):
        lines.append(f"  {i}: {p}\n")

    with (OUT / "cer_1036.csv").open("w", newline="",
                                     encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "length", "edit", "cer", "open", "close",
                    "n_rows", "text"])
        for label, txt in (("production", prod), ("binarised", binr)):
            ed = edit_ops(ref, txt)
            parts = (d["prod_parts"] if label == "production"
                     else d["bin_parts"])
            w.writerow([label, len(txt), ed, round(ed / len(ref), 6),
                        txt.count("("), txt.count(")"),
                        len(parts), txt])

    io.open(OUT / "cer_1036.txt", "w",
            encoding="utf-8").write("".join(lines))
    print("".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
