"""Phase 2: binarisation-threshold sweep on the printed-SMILES text cells.

One variable only: the fixed OpenCV threshold applied to the grayscale
cell before recognition. Crop, 2x cubic upscale, interpolation, strip,
recogniser model and recognition parameters are all held fixed so any
CER change is attributable to the threshold.

Scope is deliberately narrow. Only cmpd 1036 has a reference supplied by
the researcher; 1037 is still awaiting a human re-read and 1038/1040/2173
failed transcription QC, so none of them are scored here. Threshold 200
reproduces the earlier ``binarize_up2_en`` baseline exactly, which doubles
as a reproducibility check on this script.

Writes only under ``data/audit/phase2/threshold_scan/``. The official CSV
and every core column are read-only here.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from cpd.ms_validation import validate_against_ms  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402

PHASE2 = REPO / "data" / "audit" / "phase2"
CELLS = PHASE2 / "cells_text"
META = CELLS / "cells_text_meta.csv"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = PHASE2 / "threshold_scan"

# Fixed for every run. Threshold is the only thing that changes.
UPSCALE = 2
INTERP = cv2.INTER_CUBIC
STRIP_PX = 0
REC_LANG = "en"
THRESHOLDS = (150, 180, 200, 220, 240)
BASELINE_THRESHOLD = 200


def load_meta() -> dict[str, dict]:
    with META.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return {
        r["cmpd_id"]: r
        for r in rows
        if r["extraction_status"] == "ok" and r["crop_path"]
    }


def load_rows() -> dict[str, dict]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        return {r["cmpd_id"]: r for r in csv.DictReader(fh)}


def build_engine() -> object:
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import (
        EngineType, ModelType, OCRVersion, TaskType,
    )

    return RapidOCR(
        params={
            "Det.lang_type": "en",
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.task_type": TaskType.DET,
            "Rec.lang_type": "en",
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.0,
            "Global.use_cls": False,
        }
    )


def preprocess(img: np.ndarray, threshold: int) -> np.ndarray:
    """Strip -> 2x cubic upscale -> grayscale -> fixed threshold.

    The order matters and must match ``phase2_ocr_compare.preprocess``:
    that baseline binarises the *upscaled* image, and binarising before
    the upscale instead yields a ~150k-pixel-different image (INTER_CUBIC
    on continuous grey is not the same as on a 0/255 mask) and a
    different recognised string at the same threshold. ``verify_pipeline``
    asserts the two agree so this cannot silently drift again.
    """
    out = img
    if STRIP_PX:
        pad = STRIP_PX
        if out.shape[1] > 2 * pad:
            out = out[:, pad:out.shape[1] - pad]
    if UPSCALE != 1:
        h, w = out.shape[:2]
        out = cv2.resize(
            out, (w * UPSCALE, h * UPSCALE), interpolation=INTERP
        )
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def verify_pipeline(img: np.ndarray) -> None:
    """Fail loudly if this script's preprocessing drifts from the baseline."""
    from phase2_ocr_compare import preprocess as baseline_pre

    mine = preprocess(img, BASELINE_THRESHOLD)
    base = baseline_pre(
        img, upscale=UPSCALE, binarize=True, strip=STRIP_PX
    )
    if mine.shape != base.shape or not bool((mine == base).all()):
        raise SystemExit(
            "preprocessing does not match the binarize_up2_en baseline; "
            "the sweep would not be comparable to the earlier numbers"
        )
    print("pipeline check: identical to binarize_up2_en baseline")


def recognise(engine: object, img: np.ndarray) -> str:
    result = engine(img)
    if not result:
        return ""
    txts = result.txts if hasattr(result, "txts") else [r[1] for r in result]
    return "".join(t for t in txts if t).strip()


def diff_ops(ref: str, got: str) -> list[dict]:
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=ref, b=got
    ).get_opcodes():
        if tag == "equal":
            continue
        out.append(
            {
                "op": tag,
                "ref_pos": i1,
                "ref_text": ref[i1:i2],
                "ocr_text": got[j1:j2],
                "context": ref[max(0, i1 - 12):i2 + 12],
                "in_carbonyl": "C(=O)" in ref[
                    max(0, i1 - 3):i2 + 3
                ],
            }
        )
    return out


def run(cmpds: list[str], thresholds: tuple[int, ...]) -> dict:
    meta = load_meta()
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT / "raw_ocr"
    raw_dir.mkdir(exist_ok=True)

    targets = [
        c
        for c in cmpds
        if c in meta and c in REFERENCES and c in rows
    ]
    skipped = [c for c in cmpds if c not in targets]
    if skipped:
        print(f"skipped (no usable reference this round): {skipped}")
    if not targets:
        raise SystemExit("no compound has both a crop and an admitted reference")

    engine = build_engine()
    summary: list[dict] = []
    per_cmpd: dict[str, dict[int, dict]] = {}

    for cid in targets:
        ref = REFERENCES[cid]["smiles"]
        img = cv2.imread(str(PHASE2 / meta[cid]["crop_path"]))
        if img is None:
            print(f"WARNING: unreadable crop for {cid}")
            continue
        if cid == targets[0]:
            verify_pipeline(img)
        row = rows[cid]
        mz = float(row["ms_mz"])
        pol = (row.get("ms_polarity") or "").strip()

        per_cmpd[cid] = {}
        for th in thresholds:
            proc = preprocess(img, th)
            raw = recognise(engine, proc)
            edits = edit_ops(ref, raw) if raw else None
            cer = round(edits / len(ref), 6) if edits is not None else None
            gate = (
                validate_against_ms(raw, mz, pol)
                if raw
                else {"verdict": "empty"}
            )
            obs = gate.get("observed_mass")
            exp = gate.get("expected_mass")
            delta = round(obs - exp, 4) if obs is not None and exp else None
            ops = diff_ops(ref, raw) if raw else []

            txt_path = raw_dir / f"{cid}_th{th}.txt"
            txt_path.write_text(raw + "\n", encoding="utf-8")
            (raw_dir / f"{cid}_th{th}_diff.json").write_text(
                json.dumps(ops, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if th == thresholds[0]:
                cv2.imwrite(str(raw_dir / f"{cid}_th{th}_proc.png"), proc)

            rec = {
                "cmpd_id": cid,
                "threshold": th,
                "upscale": UPSCALE,
                "strip_px": STRIP_PX,
                "rec_lang": REC_LANG,
                "source_image": meta[cid]["source_image"],
                "crop_path": meta[cid]["crop_path"],
                "reference_len": len(ref),
                "raw_len": len(raw),
                "edit_distance": edits if edits is not None else "",
                "cer": cer if cer is not None else "",
                "n_diff_ops": len(ops),
                "n_ops_in_carbonyl": sum(1 for o in ops if o["in_carbonyl"]),
                "rdkit_parse": "yes" if raw and Chem.MolFromSmiles(raw) else "no",
                "ms_mz": mz,
                "ms_polarity": pol,
                "mass_gate_verdict": gate.get("verdict"),
                "mass_delta_da": delta if delta is not None else "",
                "mass_error_ppm": gate.get("error_ppm")
                if gate.get("error_ppm") is not None
                else "",
                "raw_ocr_file": f"raw_ocr/{txt_path.name}",
                "diff_file": f"raw_ocr/{cid}_th{th}_diff.json",
            }
            summary.append(rec)
            per_cmpd[cid][th] = rec
            print(
                f"{cid} th={th:3d} len={len(raw):4d} edits={edits} "
                f"CER={cer} parse={rec['rdkit_parse']} "
                f"dDa={delta}"
            )

    with (OUT / "threshold_scan.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    write_report(per_cmpd, targets, thresholds)
    return per_cmpd


def write_report(
    per_cmpd: dict[str, dict[int, dict]],
    targets: list[str],
    thresholds: tuple[int, ...],
) -> None:
    L: list[str] = []
    add = L.append
    add("Phase 2 - binarisation threshold sweep")
    add("")
    add("PRELIMINARY: scored on a single reference (cmpd 1036). This shows")
    add("threshold stability for one string, NOT an accuracy rate for the")
    add("dataset. 1037 awaits a human re-read; 1038/1040/2173 failed")
    add("transcription QC and are excluded.")
    add("")
    add(f"fixed    : upscale={UPSCALE}x interpolation=cubic strip={STRIP_PX}px "
        f"rec_lang={REC_LANG} threshold=VARIED")
    add(f"baseline : threshold {BASELINE_THRESHOLD} == earlier "
        f"'binarize_up2_en' config")
    add("")

    for cid in targets:
        add(f"=== cmpd {cid} ===")
        add(
            f"{'thresh':>6s} {'len':>5s} {'edits':>6s} {'CER':>9s} "
            f"{'carbonyl_ops':>12s} {'parse':>6s} {'dDa':>9s}  verdict"
        )
        for th in thresholds:
            r = per_cmpd[cid].get(th)
            if not r:
                continue
            add(
                f"{r['threshold']:6d} {r['raw_len']:5d} "
                f"{str(r['edit_distance']):>6s} {str(r['cer']):>9s} "
                f"{r['n_ops_in_carbonyl']:12d} {r['rdkit_parse']:>6s} "
                f"{str(r['mass_delta_da']):>9s}  {r['mass_gate_verdict']}"
            )
        base = per_cmpd[cid].get(BASELINE_THRESHOLD)
        best = min(
            (per_cmpd[cid][t] for t in thresholds if cid in per_cmpd
             and t in per_cmpd[cid]),
            key=lambda r: (r["edit_distance"] is None,
                           r["edit_distance"] or 10**9),
            default=None,
        )
        if base and best:
            add("")
            add(
                f"best threshold = {best['threshold']} "
                f"(CER {best['cer']}) vs baseline {BASELINE_THRESHOLD} "
                f"(CER {base['cer']})"
            )
            gain = (
                (base["edit_distance"] or 0) - (best["edit_distance"] or 0)
            )
            add(
                f"edit distance {base['edit_distance']} -> "
                f"{best['edit_distance']} (change {gain:+d} chars)"
            )
            if base["edit_distance"] == 0:
                add("baseline already exact; sweep cannot improve it")
        add("")
        add("--- residual differences vs reference (best threshold) ---")
        if best:
            dfile = OUT / f"raw_ocr/{cid}_th{best['threshold']}_diff.json"
            ops = json.loads(dfile.read_text(encoding="utf-8"))
            if not ops:
                add("  (none - exact character match)")
            for o in ops:
                flag = "  <-- in C(=O)" if o["in_carbonyl"] else ""
                add(
                    f"  {o['op']:7s} @{o['ref_pos']:>3d} "
                    f"ref={o['ref_text']!r} -> ocr={o['ocr_text']!r}{flag}"
                )
                add(f"           context ...{o['context']}...")
        add("")

    (OUT / "threshold_scan_report.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {OUT/'threshold_scan.csv'}")
    print(f"wrote {OUT/'threshold_scan_report.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmpds", default="1036")
    ap.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in THRESHOLDS),
    )
    args = ap.parse_args()
    ths = tuple(
        int(t) for t in args.thresholds.split(",") if t.strip()
    )
    run([c.strip() for c in args.cmpds.split(",") if c.strip()], ths)


if __name__ == "__main__":
    main()
