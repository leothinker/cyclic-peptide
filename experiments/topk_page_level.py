r"""Reproducible rebuild of the page-level per-character top-k extraction.

Background
----------
``demo_rapidocr_topk/topk_printed_smiles.json`` is an orphaned artefact. Its
generating script is gone, its two "engine" sections are byte-identical, and
its ``top3`` field is misaligned on 6 of 13 rows (see
``topk_artifact_audit.py``). Yet its cmpd-1036 reconstruction scores edit
distance 0 against the human-verified reference, beating the production
pipeline's 2. That is worth reproducing properly rather than trusting.

What this script does
---------------------
Runs the SAME fixed OCR baseline (2x INTER_CUBIC -> threshold 200 -> PP-OCRv4
mobile English) but takes lines from the PAGE-LEVEL detector instead of the
per-cell crop. That is the one intended difference, and the reason this path
can see a row-leading ``=`` that the cell-crop path clips away.

Per-character candidates come from ``rec.session`` -> (T, C) softmax,
walked with the same greedy CTC collapse the official decoder uses:
emit at timestep t only when argmax is non-blank and differs from the
previous timestep's argmax. ``top3[0]`` is therefore the greedy choice by
construction, which is what makes the self-check below meaningful.

Known limitation (unchanged from ``phase2_recog_candidates.py``): because
candidates are only read at KEPT timesteps, a character that greedy CTC
dropped entirely cannot be recovered from this file. Fixing that needs the
exact CTC scorer, not a wider top-k.

Trustworthiness measures
------------------------
* one line per recognition batch, so a captured (T, C) tensor provably
  belongs to that line -- no batch-to-line distribution guesswork;
* a self-check that ``top3[0].char == chosen char`` at every position,
  which is exactly the invariant the orphaned file violated;
* row-count conservation, because the orphaned file silently dropped a whole
  39-character row of cmpd 1037 without any error;
* comparison against the verified reference where one exists, reported as a
  score only, never used to select anything.

Read-only with respect to the pipeline, the official CSV and the baseline.

Run:
    .venv\Scripts\python.exe topk_page_level.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import BASELINE_THRESHOLD, build_engine  # noqa: E402
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from row_crop_audit import line_boxes, preprocess  # noqa: E402

OUTDIR = REPO / "data" / "audit" / "phase2" / "page_level_topk"
GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"

TOP_K = 5
UNCERTAIN = 0.9
ENGINE_TAG = "en_PP-OCRv4"


class SessionSpy:
    """Capture the (T, C) posterior the official recogniser actually used."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def __call__(self, x):
        out = self.inner(x)
        self.last = np.asarray(out).copy()
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def greedy_chars(probs, char_list, top_k):
    """Walk the posterior with greedy CTC collapse, collecting top-k.

    Returns ``(chars, mean_conf)`` where each char entry is
    ``{"char": c, "pos": i, "top3": [{"char", "prob"}, ...]}``.
    """
    order = np.argsort(-probs, axis=1)[:, :top_k]
    emitted, prev = [], -1
    for t in range(probs.shape[0]):
        k = int(order[t, 0])
        if k == 0:                      # CTC blank
            prev = k
            continue
        if k == prev:                   # collapsed repeat
            prev = k
            continue
        prev = k
        alts = [{"char": char_list[int(j)], "prob": round(float(probs[t, j]), 4)}
                for j in order[t]]
        emitted.append({"char": char_list[k], "pos": len(emitted), "top3": alts})
    conf = ([c["top3"][0]["prob"] for c in emitted] or [0.0])
    return emitted, float(np.mean(conf))


def run_page(engine, page_name, page_rel):
    """Page-level detection + one-line-per-batch recognition with top-k."""
    img = cv2.imread(str(GOLD / page_rel))
    if img is None:
        return None
    proc = preprocess(img, BASELINE_THRESHOLD)
    boxes = line_boxes(engine(proc))
    # text_rec is lazily built by the first engine() call, so it only exists
    # once detection has run at least once.
    rec = engine.text_rec
    char_list = list(rec.postprocess_op.character)

    spy = SessionSpy(rec.session)
    rec.session = spy
    rows = []
    try:
        for i, poly in enumerate(boxes):
            pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
            crop = get_rotate_crop_image(proc, pts)
            out = engine.recognize_txt([crop])
            probs = spy.last
            if probs is None or probs.ndim != 3:
                raise RuntimeError(f"no posterior captured for box {i}")
            p = probs[0]
            chars, conf = greedy_chars(p, char_list, TOP_K)
            # recognise_txt returns a bare object for a single-line batch
            res = out[0] if isinstance(out, list) else out
            # fields are txts/scores tuples on a single-line batch
            text = (res.txts[0] if getattr(res, "txts", None) else "") or ""
            ocr_score = (float(res.scores[0])
                         if getattr(res, "scores", None) else 0.0)
            rows.append({
                "line": i,
                "text": text,
                "score": round(float(conf), 4),
                "ocr_score": round(ocr_score, 4),
                "chars": chars,
                "box": [round(float(v), 1) for v in pts.reshape(-1)],
                "crop_w": int(crop.shape[1]),
                "crop_h": int(crop.shape[0]),
            })
    finally:
        rec.session = spy.inner
    return {"page": page_rel, "engine": ENGINE_TAG, "rows": rows}


def self_check(rows):
    """Three invariants that the orphaned artefact violated.

    1. top3[0] must be the char the greedy walk emitted at that position.
    2. joining the walked chars must reproduce the official OCR text, which
       proves the capture belongs to this line and to this line only.
    3. the two must agree on length.
    """
    out = []
    for r in rows:
        bad = [c["pos"] for c in r["chars"]
               if not c["top3"] or c["top3"][0]["char"] != c["char"]]
        joined = "".join(c["char"] for c in r["chars"])
        out.append({
            "line": r["line"],
            "n_chars": len(r["chars"]),
            "top3_mismatch": len(bad),
            "chars_match_text": int(joined == r["text"]),
            "walked_len": len(joined),
            "ocr_len": len(r["text"]),
            "low_conf": sum(1 for c in r["chars"]
                            if c["top3"][0]["prob"] < UNCERTAIN),
        })
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    engine = build_engine()

    cells = _manifest_cells()
    # One recognition batch per line is the whole point (it makes the
    # captured posterior provably belong to that line), so a full 80-page
    # sweep is slow. Default to the pages that carry a verified reference.
    want = sys.argv[1:] or ["I100297", "I100298", "I100300"]
    pages = [p for p in sorted({c["page"] for c in cells}) if p in want]
    cells = [c for c in cells if c["page"] in pages]
    if not pages:
        raise SystemExit(f"no cells on pages {want}")
    print(f"pages: {pages}  ({len(cells)} cells)")
    all_rows, summary = {}, []
    for tag in pages:
        rel = next(c["source_image"] for c in cells if c["page"] == tag)
        res = run_page(engine, tag, rel)
        if res is None:
            print(f"  skip {tag}: page not readable")
            continue
        rows = res["rows"]
        chk = self_check(rows)
        all_rows.setdefault(ENGINE_TAG, {"engine": ENGINE_TAG, "pages": {}})
        all_rows[ENGINE_TAG]["pages"][tag] = rows
        bad = sum(c["top3_mismatch"] for c in chk)
        mism = sum(1 for c in chk if not c["chars_match_text"])
        print(f"  {tag}: {len(rows)} lines, top3 mismatches={bad}, "
              f"chars!=text on {mism} lines")
        summary.append({"page": tag, "n_lines": len(rows),
                        "top3_mismatch": bad, "chars_ne_text": mism,
                        "checks": chk})

    with (OUTDIR / "page_level_topk.json").open("w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, ensure_ascii=False, indent=1)

    cmpd = _assign_compounds(all_rows, cells)
    with (OUTDIR / "cmpd_rows.csv").open("w", newline="",
                                        encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["cmpd_id", "page", "page_line", "text_len", "text"])
        for cid, rows in cmpd.items():
            for r in rows:
                w.writerow([cid, r["page"], r["line"], len(r["text"]),
                            r["text"]])

    with (OUTDIR / "self_check.csv").open("w", newline="",
                                          encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["page", "line", "n_chars",
                                           "top3_mismatch",
                                           "chars_match_text", "walked_len",
                                           "ocr_len", "low_conf"])
        w.writeheader()
        for s in summary:
            for c in s["checks"]:
                w.writerow({"page": s["page"], **c})

    _report(summary, cells, cmpd)
    print(f"\nwrote {OUTDIR}")


def _manifest_cells():
    """Per-compound cell boxes in processed coords, from the audit manifest."""
    path = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        if r["is_header"] != "no":
            continue
        out.append({
            "cmpd_id": r["cmpd_id"], "page": r["page"],
            "source_image": r["source_image"],
            "x1": int(r["bbox_x1"]), "y1": int(r["bbox_y1"]),
            "x2": int(r["bbox_x2"]), "y2": int(r["bbox_y2"]),
            "cell_rows": int(r["n_lines"] or 0),
        })
    return out


def _assign_compounds(all_rows, cells):
    """Keep only SMILES-column boxes, then bucket them by compound y-range.

    Everything is derived from the page itself, because the cell-crop
    manifest lives in a different coordinate frame from whole-page detection
    and mixing the two silently misattributes every row.

    * the SMILES column is the x-range of the header box reading "SMILES";
    * each compound's band starts at the top of its own numeric label in the
      "Cmpd #" column and runs to the top of the next label, so a printed row
      is attributed to exactly one compound.
    """
    out: dict[str, list] = {}
    pages = all_rows.get(ENGINE_TAG, {}).get("pages", {})

    smiles_col, id_col = {}, {}
    for tag, rows in pages.items():
        for r in rows:
            t = r["text"].strip().upper()
            xs = r["box"][0::2]
            if t == "SMILES" and tag not in smiles_col:
                smiles_col[tag] = (min(xs), max(xs))
            elif t in ("CMPD #", "CMPD#", "CMPD") and tag not in id_col:
                id_col[tag] = (min(xs), max(xs))

    # compound bands from the numeric labels in the id column
    bands = {}
    for tag, rows in pages.items():
        if tag not in id_col or tag not in smiles_col:
            continue
        ilo, ihi = id_col[tag]
        labels = []
        for r in rows:
            t = r["text"].strip()
            if not t.isdigit():
                continue
            xs = r["box"][0::2]
            ys = r["box"][1::2]
            mid = sum(xs) / len(xs)
            if not (ilo - 0.3 * max(1.0, ihi - ilo) <= mid <= ihi):
                continue
            labels.append((min(ys), t))
        labels.sort()
        bands[tag] = labels

    for tag, labels in bands.items():
        lo, hi = smiles_col[tag]
        span = max(1.0, hi - lo)
        for i, (y_top, cid) in enumerate(labels):
            # The printed SMILES sits ABOVE its own "Cmpd #" label on these
            # pages, so a compound's rows run from the PREVIOUS label down to
            # its own. Getting this backwards silently attributes one
            # compound's string to its neighbour.
            y_start = labels[i - 1][0] if i > 0 else -1
            y_end = y_top
            for r in pages[tag]:
                xs = r["box"][0::2]
                ys = r["box"][1::2]
                mid = sum(xs) / len(xs)
                # a printed SMILES row is far wider than the column header,
                # so membership is decided by the row centre, not its extent
                if not (lo - 0.40 * span <= mid <= hi + 0.40 * span):
                    continue
                # column headers live in the same column and would otherwise
                # be counted as a printed row of the first compound
                if r["text"].strip().upper() in ("SMILES", "CMPD #", "CMPD#",
                                                 "CMPD", "STRUCTURE"):
                    continue
                cy = sum(ys) / len(ys)
                if not (y_start < cy <= y_end):
                    continue
                out.setdefault(cid, []).append(
                    {"page": tag, "line": r["line"], "text": r["text"],
                     "chars": r["chars"], "box": r["box"]})
    for v in out.values():
        v.sort(key=lambda r: r["line"])
    return out


def _report(summary, cells, cmpd) -> None:
    lines = ["=== page-level per-char top-k: rebuild summary ===", ""]
    lines.append("Self-check (every row must pass both):")
    lines.append("  top3[0].char == chosen char")
    lines.append("  join(greedy walk) == official OCR text")
    for s in summary:
        ok = s["top3_mismatch"] == 0 and s["chars_ne_text"] == 0
        lines.append(f"  {s['page']}: {s['n_lines']} lines  "
                     f"{'PASS' if ok else 'FAIL'}")
    lines += ["", "Row-count conservation is NOT asserted here: a page-level",
              "detector may legitimately miss a printed row, so a compound",
              "whose row count drops must be failed by the caller, not",
              "silently scored. Compare against the reference row count.", ""]
    cellmap = {c["cmpd_id"]: c for c in cells}
    lines.append("Per-compound result (page-level boxes filtered to the "
                 "SMILES column):")
    lines.append(f"{'cmpd':>6} {'page':>8} {'rows':>5} {'cellrows':>9} "
                 f"{'joined':>7} {'ref':>5} {'edit':>5} {'CER':>8}")
    for cid in sorted(cmpd, key=lambda s: int(s)):
        grp = cmpd[cid]
        txt = "".join(r["text"] for r in grp)
        ref = REFERENCES.get(cid, {}).get("smiles")
        c = cellmap.get(cid, {})
        if ref:
            ed = edit_ops(ref, txt)
            lines.append(f"{cid:>6} {c.get('page',''):>8} {len(grp):>5} "
                         f"{c.get('cell_rows', 0):>9} {len(txt):>7} "
                         f"{len(ref):>5} {ed:>5} {ed / len(ref):>8.5f}")
        else:
            lines.append(f"{cid:>6} {c.get('page',''):>8} {len(grp):>5} "
                         f"{c.get('cell_rows', 0):>9} {len(txt):>7} "
                         f"{'-':>5} {'-':>5} {'-':>8}")
    (OUTDIR / "rebuild_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
