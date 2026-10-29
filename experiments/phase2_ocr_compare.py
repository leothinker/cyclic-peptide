"""Task 4: OCR configuration sweep on the printed-SMILES text cells.

Reads the Task-2 crops from ``data/audit/phase2/cells_text/`` and runs a
small matrix of preprocessing x recogniser settings over the same pixels.
For every (cmpd, config) it records the raw recognised string, the
character error rate against the hand-verified reference where one
exists, RDKit parse status, and the mass-gate verdict against that row's
own ``ms_mz`` / ``ms_polarity``.

Nothing here writes to ``data/processed/``. The only reference strings are
the ones a chemist typed by hand and confirmed against the patent page;
no reference is synthesised from DECIMER output or from a mass match.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
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

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
META = CELLS / "cells_text_meta.csv"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT_DIR = REPO / "data" / "audit" / "phase2"

# ---------------------------------------------------------------------------
# Hand-verified references. Only compounds whose printed SMILES a chemist
# transcribed character-by-character from the golden table belong here.
# ``ms_gate_note`` records whether the reference passes the patent m/z gate;
# a failing reference is still a valid CER reference but must not be
# treated as a validated chemical truth.
# ---------------------------------------------------------------------------
REFERENCES: dict[str, dict] = {
    "1036": {
        "smiles": (
            "CC[C@H]"
            "(C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
            "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
            "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)"
            "[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
            "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
        ),
        "verified_by": (
            "supplied verbatim by the researcher, transcribed from I100297 "
            "and confirmed against the printed cell"
        ),
        "ms_gate_note": (
            "reference FAILS the patent m/z gate (see "
            "positive_tests.csv); CER reference only, not chemical truth"
        ),
    },
    "1037": {
        "smiles": (
            "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](CC2=CC=CC=C2)C(=O)N3[C@@H](CC3)"
            "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
            "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H]"
            "(C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)="
            "C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
        ),
        "verified_by": (
            "agent transcription from I100297 zoomed 1.6x bands; "
            "NOT yet confirmed by a human reader"
        ),
        "ms_gate_note": "reference status not yet established",
    },
    "1038": {
        "smiles": (
            "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N"
            "(CC(F)(F)F)[C@@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)"
            "C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H]"
            "(C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C8=C(F)C(F)="
            "C(SC3)C(F)=C8F)C(=O)N1)C(=O)N(C)C"
        ),
        "verified_by": (
            "agent transcription from I100298 zoomed 1.6x bands; "
            "NOT yet confirmed by a human reader"
        ),
        "ms_gate_note": "reference status not yet established",
    },
    "1040": {
        "smiles": (
            "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CC3)C(=O)N(C)[C@@H](C4CCCCC4)"
            "C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)CCC)"
            "CC(=O)N(C)[C@@H](C)C(=O)N6[C@@H](CC6)C(=O)N(C)[C@H](CSC7=C(F)"
            "C(F)=C(C(=S)NCC5)C(F)=C7F)C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)="
            "C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C"
        ),
        "verified_by": (
            "agent transcription from I100298 zoomed 1.6x bands; "
            "NOT yet confirmed by a human reader"
        ),
        "ms_gate_note": "reference status not yet established",
    },
    "2173": {
        "smiles": (
            "FC(F)CO[C@@H]1C[C@H]2C(=O)N(C)C3(CCC3)C(=O)N(C)[C@@H](C4CCCCC4)"
            "C(=O)N(C)[C@@H](CC(=O)N(CC)[C@H]5C(=O)N[C@@H]([C@@H](C)CCC)"
            "CC(=O)N(C)[C@@H](C)C(=O)N6[C@@H](CC6)C(=O)N(CCC(C)C)[C@@H]"
            "(CC7=CC=C(C=C7)C8=CN(N=N8)CC5)C(=O)N(C)CC(=O)N[C@@H](CCC9="
            "CC(F)=C(C(F)=C9)C(F)(F)F)C(=O)N2C1)C(=O)N(C)C"
        ),
        "verified_by": (
            "agent transcription from I100376 leading row, zoomed 1.6x "
            "bands; NOT yet confirmed by a human reader"
        ),
        "ms_gate_note": "reference status not yet established",
    },
}


def edit_ops(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(a=a, b=b)
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    )


def preprocess(img: np.ndarray, *, upscale: int, binarize: bool,
                strip: int) -> np.ndarray:
    out = img
    if strip:
        pad = strip
        if out.shape[1] > 2 * pad:
            out = out[:, pad:out.shape[1] - pad]
    if upscale and upscale != 1:
        h, w = out.shape[:2]
        out = cv2.resize(
            out, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC
        )
    if binarize:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        out = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return out


CONFIGS = [
    # name,           upscale, binarize, strip_px, rec_lang
    ("baseline_en",        1, False,   0, "en"),
    ("baseline_ch",        1, False,   0, "ch"),
    ("up2_en",             2, False,   0, "en"),
    ("up4_en",             4, False,   0, "en"),
    ("strip20_en",         1, False,  20, "en"),
    ("strip20_up2_en",     2, False,  20, "en"),
    ("binarize_en",        1, True,    0, "en"),
    ("binarize_up2_en",    2, True,    0, "en"),
    ("strip20_up2_bin_en", 2, True,   20, "en"),
]


def build_engines(engines: dict[str, object]) -> dict[str, object]:
    """Instantiate only the recogniser variants the sweep actually needs."""
    from rapidocr import RapidOCR

    engines["ch"] = RapidOCR()
    from rapidocr.utils.typings import (
        EngineType, ModelType, OCRVersion, TaskType,
    )

    engines["en"] = RapidOCR(
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
    return engines


def recognise(engine: object, img: np.ndarray) -> str:
    result = engine(img)
    if not result:
        return ""
    txts = result.txts if hasattr(result, "txts") else [r[1] for r in result]
    return "".join(t for t in txts if t)


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


def reference_admissibility(
    rows: dict[str, dict], tol_da: float = 5.0
) -> dict[str, dict]:
    """Classify each reference by its mass-gate delta.

    A large |delta| against the patent m/z is the signature of a
    *transcription* slip rather than a chemistry problem: one residue in
    this series is 100-200 Da, so a 14-56 Da error means whole residues
    were misread or dropped. Only references whose delta is small enough
    to be a single-character slip are admitted as CER ground truth.
    Anything larger is reported but excluded from CER scoring, because
    scoring OCR against a known-bad reference is meaningless.

    This is a *transcription QC* device, not a chemistry verdict: it says
    "this string probably matches the printed cell", never "this molecule
    is correct".
    """
    out: dict[str, dict] = {}
    for cid, ref in REFERENCES.items():
        row = rows.get(cid, {})
        try:
            mz = float(row.get("ms_mz") or "")
        except ValueError:
            mz = None
        pol = (row.get("ms_polarity") or "").strip()
        delta = None
        ppm = None
        verdict = "no_ms_data"
        if ref["smiles"] and mz and pol:
            v = validate_against_ms(ref["smiles"], mz, pol)
            obs = v.get("observed_mass")
            exp = v.get("expected_mass")
            verdict = v.get("verdict")
            ppm = v.get("error_ppm")
            if obs is not None and exp is not None:
                delta = round(obs - exp, 4)
        delta = delta if delta is not None else float("inf")
        out[cid] = {
            "delta_da": delta,
            "error_ppm": ppm,
            "mass_gate_verdict": verdict,
            "cer_admissible": abs(delta) <= tol_da,
            "reason": (
                f"|delta|={abs(delta):.2f} Da vs tol {tol_da} Da"
            ),
        }
    return out


def run(cmpds: list[str]) -> None:
    meta = load_meta()
    rows = load_rows()
    adm = reference_admissibility(rows)
    targets = [c for c in cmpds if c in meta]
    missing = [c for c in cmpds if c not in meta]
    if missing:
        print(f"WARNING: no text crop for {missing}")
    if not targets:
        print("nothing to run")
        return

    engines: dict[str, object] = {}
    out_rows: list[dict] = []
    t0 = time.time()
    for name, up, binz, strip, lang in CONFIGS:
        if lang not in engines:
            print(f"[{time.time()-t0:6.1f}s] loading '{lang}' engine ...")
            build_engines(engines)
        engine = engines[lang]
        for cid in targets:
            img = cv2.imread(str(CELLS.parent / meta[cid]["crop_path"]))
            if img is None:
                continue
            proc = preprocess(
                img, upscale=up, binarize=binz, strip=strip
            )
            raw = recognise(engine, proc).strip()
            row = rows.get(cid, {})
            mz = row.get("ms_mz") or ""
            pol = (row.get("ms_polarity") or "").strip()
            try:
                mz_f = float(mz) if mz else None
            except ValueError:
                mz_f = None

            gate = (
                validate_against_ms(raw, mz_f, pol)
                if raw and mz_f
                else {"verdict": "no_input"}
            )
            delta = None
            if gate.get("observed_mass") and gate.get("expected_mass"):
                delta = round(
                    gate["observed_mass"] - gate["expected_mass"], 4
                )

            ref = REFERENCES.get(cid)
            cer = None
            if ref and adm.get(cid, {}).get("cer_admissible"):
                cer = round(
                    edit_ops(ref["smiles"], raw) / len(ref["smiles"]), 4
                ) if raw else None

            out_rows.append(
                {
                    "cmpd_id": cid,
                    "config": name,
                    "rec_lang": lang,
                    "upscale": up,
                    "binarize": binz,
                    "strip_px": strip,
                    "source_image": meta[cid]["source_image"],
                    "crop_path": meta[cid]["crop_path"],
                    "raw_ocr_smiles": raw,
                    "raw_len": len(raw),
                    "has_reference": "yes" if ref else "no",
                    "reference_cer_admissible": (
                        "yes" if adm.get(cid, {}).get("cer_admissible") else "no"
                    ),
                    "reference_delta_da": adm.get(cid, {}).get("delta_da"),
                    "reference_error_ppm": adm.get(cid, {}).get("error_ppm"),
                    "cer_vs_reference": cer if cer is not None else "",
                    "reference_verified_by": ref["verified_by"] if ref else "",
                    "rdkit_parse": "yes" if raw and Chem.MolFromSmiles(raw) else "no",
                    "ms_mz": mz,
                    "ms_polarity": pol,
                    "mass_gate_verdict": gate.get("verdict"),
                    "mass_delta_da": delta if delta is not None else "",
                    "mass_error_ppm": gate.get("error_ppm")
                    if gate.get("error_ppm") is not None
                    else "",
                }
            )
        print(
            f"[{time.time()-t0:6.1f}s] config {name:20s} done "
            f"({len(targets)} cells)"
        )

    out_csv = OUT_DIR / "ocr_compare.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    with (OUT_DIR / "ocr_references.json").open("w", encoding="utf-8") as fh:
        json.dump(REFERENCES, fh, indent=2, ensure_ascii=False)

    write_summary(out_rows, targets, adm)
    print(f"\nwrote {out_csv}")


def write_summary(
    out_rows: list[dict], targets: list[str], adm: dict[str, dict]
) -> None:
    lines: list[str] = []
    add = lines.append
    add("Phase 2 Task 4 - OCR configuration comparison")
    add(f"cells compared : {len(targets)} ({', '.join(targets)})")
    add(f"configs        : {len(CONFIGS)}")
    add("")
    add("IMPORTANT: a config that raises the RDKit parse rate is NOT proof of")
    add("correct structure. Parse rate measures bracket balance, not chemistry.")
    add("")
    add("--- reference transcription QC ---")
    add("A reference is admitted as CER ground truth only when its mass-gate")
    add("delta is small enough to be a character slip (tol 5 Da). A large delta")
    add("means the transcription itself is wrong (a whole residue is 100-200 Da),")
    add("so that reference is reported but excluded from CER.")
    add("")
    add(
        f"{'cmpd':>5s} {'len':>4s} {'dDa':>9s} {'ppm':>10s} "
        f"{'verdict':>9s} {'CER usable':>11s}"
    )
    for cid in REFERENCES:
        a = adm.get(cid, {})
        d = a.get("delta_da")
        add(
            "{:>5s} {:>4d} {:>9.3f} {:>10s} {:>9s} {:>11s}".format(
                cid,
                len(REFERENCES[cid]["smiles"]),
                d if d is not None and d != float("inf") else float("nan"),
                str(a.get("error_ppm")),
                str(a.get("mass_gate_verdict")),
                "yes" if a.get("cer_admissible") else "NO",
            )
        )
    add("")

    by_cmpd: dict[str, list[dict]] = {}
    for r in out_rows:
        by_cmpd.setdefault(r["cmpd_id"], []).append(r)

    for cid in targets:
        rows = by_cmpd.get(cid, [])
        if not rows:
            continue
        add(f"=== cmpd {cid} ===")
        ref = REFERENCES.get(cid)
        if ref:
            add(f"reference ({len(ref['smiles'])} chars) verified by: "
                f"{ref['verified_by']}")
            add(f"mass-gate note: {ref['ms_gate_note']}")
            if not adm.get(cid, {}).get("cer_admissible"):
                add(
                    "CER EXCLUDED: transcription QC failed "
                    f"({adm.get(cid, {}).get('reason')}) - this string is "
                    "missing or misreading part of the printed cell"
                )
        else:
            add("reference: NONE - CER cannot be computed for this compound")
        add(f"{'config':22s} {'len':>4s} {'CER':>7s} {'RDKit':>6s} "
            f"{'dDa':>9s}  verdict")
        for r in sorted(rows, key=lambda x: x["config"]):
            cer = r["cer_vs_reference"]
            cer_s = f"{cer:.4f}" if cer != "" else "n/a"
            add(
                f"{r['config']:22s} {r['raw_len']:4d} {cer_s:>7s} "
                f"{r['rdkit_parse']:>6s} {str(r['mass_delta_da']):>9s}  "
                f"{r['mass_gate_verdict']}"
            )
        add("")

    add("--- aggregate over configs ---")
    for cid in targets:
        rows = by_cmpd.get(cid, [])
        if not rows:
            continue
        parsers = sum(1 for r in rows if r["rdkit_parse"] == "yes")
        within = sum(
            1
            for r in rows
            if r["mass_gate_verdict"] in ("match", "within_tolerance")
        )
        cers = [r["cer_vs_reference"] for r in rows
                if r["cer_vs_reference"] != ""]
        add(
            f"cmpd {cid}: RDKit-parse {parsers}/{len(rows)} configs, "
            f"mass-gate pass {within}/{len(rows)} configs, "
            f"CER range "
            + (f"{min(cers):.4f}-{max(cers):.4f}" if cers else "n/a")
        )

    add("")
    add("--- per-config aggregate (all compared cells) ---")
    add(f"{'config':22s} {'RDKit-parse':>12s} {'mass-gate pass':>15s}")
    for name, *_ in CONFIGS:
        rows = [r for r in out_rows if r["config"] == name]
        if not rows:
            continue
        parsers = sum(1 for r in rows if r["rdkit_parse"] == "yes")
        within = sum(
            1
            for r in rows
            if r["mass_gate_verdict"] in ("match", "within_tolerance")
        )
        add(f"{name:22s} {parsers:>7d}/{len(rows):<4d} "
            f"{within:>10d}/{len(rows):<4d}")

    path = OUT_DIR / "ocr_compare_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-14:]))
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cmpds",
        default=(
            "1036,1037,1038,1039,1040,1041,2170,2171,2172,2173"
        ),
        help=(
            "comma-separated cmpd ids to sweep. Only ids present in "
            "REFERENCES get a CER; the rest still contribute parse-rate "
            "and mass-gate columns."
        ),
    )
    args = ap.parse_args()
    run([c.strip() for c in args.cmpds.split(",") if c.strip()])


if __name__ == "__main__":
    main()
