"""Demo: per-character top-k OCR on a real printed SMILES cell.

The SMILES text lives in the table JPG (cells/ only has rendered
structure PNGs). We OCR the full page, pick the longest text row that
looks like a SMILES fragment (contains ``C(N)...``), and dump the
per-character top-3 alternatives with softmax probabilities. Characters
where the chosen top-1 has probability < 0.9 are marked as "uncertain";
for those we try a greedy one-swap with the top-2/top-3 candidate and
report whether any swap fixes the SMILES parse.

Run::

    .venv\\Scripts\\python.exe -u _demo_topk.py

Writes to ``demo_rapidocr_topk/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from rapidocr import RapidOCR

PAGE = "data/raw/WO2025162428/golden_tables/PCTCN2025075389-ftappb-I100297.jpg"
OUT = Path("demo_rapidocr_topk")
TOP_K = 5
UNCERTAIN_THRESHOLD = 0.9
MIN_LEN = 25


def _run(engine_label: str, lang: str) -> list[dict]:
    engine = RapidOCR(params={})
    engine.cfg.Det["lang_type"] = lang
    engine.cfg.Rec["lang_type"] = lang
    engine.return_top_k = TOP_K
    engine.cfg.Global.return_top_k = TOP_K
    engine.cfg.Rec["top_k"] = TOP_K

    result = engine(PAGE)
    if result is None:
        return []
    txts = list(getattr(result, "txts", []) or [])
    scores = list(getattr(result, "scores", []) or [])
    top_k_lines = list(getattr(result, "top_k_lines", []) or [])

    lines: list[dict] = []
    for line_idx, text in enumerate(txts):
        if not text or len(text) < MIN_LEN:
            continue
        score = float(scores[line_idx]) if line_idx < len(scores) else 0.0
        line_top_k = top_k_lines[line_idx] if line_idx < len(top_k_lines) else None
        chars: list[dict] = []
        if line_top_k:
            for i, ch in enumerate(text):
                alts = (
                    line_top_k[i]
                    if i < len(line_top_k) and line_top_k[i]
                    else [(ch, score)]
                )
                top_three = [
                    {"char": a[0], "prob": round(float(a[1]), 4)}
                    for a in list(alts)[:3]
                ]
                chars.append({"pos": i, "char": ch, "top3": top_three})
        lines.append(
            {"line": line_idx, "text": text, "score": round(score, 4), "chars": chars}
        )
    return lines


def _pick_smi_row(engine_label: str, lines: list[dict]) -> dict | None:
    smi_rows = [
        lp for lp in lines
        if "(" in lp["text"] and ")" in lp["text"] and "C" in lp["text"]
    ]
    candidates = smi_rows or lines
    if not candidates:
        return None
    chosen = max(candidates, key=lambda x: len(x["text"]))
    chosen["engine"] = engine_label
    return chosen


def _render(line: dict | None, header: str) -> str:
    if line is None:
        return f"### {header}\n(no SMILES-like row found)"
    text = line["text"]
    chars = line["chars"]
    score = line["score"]

    out = [
        f"### {header}",
        f"page : {PAGE}",
        f"line : {line['line']}  (engine={line.get('engine', '?')})",
        f"score: {score}",
        f"text : {text}",
        "",
        "pos | chosen     | p(chosen) | top-3 alternatives (char, prob)",
        "----+------------+-----------+--------------------------------------",
    ]
    for ch in chars:
        top3 = ch["top3"]
        chosen_p = top3[0]["prob"] if top3 else 0.0
        marker = "" if chosen_p >= UNCERTAIN_THRESHOLD else "  <- uncertain"
        top = ", ".join(f"{t['char']!r:<5} {t['prob']:.3f}" for t in top3)
        out.append(
            f"{ch['pos']:>3} | {ch['char']!r:<10} | {chosen_p:.3f}    | {top}{marker}"
        )

    n_uncertain = sum(1 for ch in chars if ch["top3"] and ch["top3"][0]["prob"] < UNCERTAIN_THRESHOLD)
    out.append("")
    out.append(f"# {n_uncertain}/{len(chars)} chars have top-1 prob < {UNCERTAIN_THRESHOLD}")

    seen: set[str] = set()
    rescued = []
    chars_list = list(text)
    for pos, ch_entry in enumerate(chars):
        alts = ch_entry["top3"]
        if not alts or alts[0]["prob"] >= UNCERTAIN_THRESHOLD:
            continue
        for alt in alts[1:]:
            cand_text = "".join(chars_list[:pos] + [alt["char"]] + chars_list[pos + 1 :])
            if cand_text in seen:
                continue
            seen.add(cand_text)
            mol = Chem.MolFromSmiles(cand_text)
            if mol is not None:
                rescued.append({"pos": pos, "from": text[pos], "to": alt["char"], "from_p": alts[0]["prob"], "to_p": alt["prob"]})
                break

    out.append(
        f"# greedy one-swap rescue: {len(rescued)} successful char swap(s) that make the row RDKit-valid"
    )
    for r in rescued[:5]:
        out.append(
            f"   pos {r['pos']:>3}: {r['from']!r:<3} ({r['from_p']:.3f}) -> {r['to']!r:<3} ({r['to_p']:.3f})"
        )

    return "\n".join(out)


def main() -> int:
    if not Path(PAGE).exists():
        print(f"[!] page missing: {PAGE}")
        return 1
    OUT.mkdir(exist_ok=True)

    sections: dict[str, list[dict] | dict] = {}
    chosen_per_engine: dict[str, dict] = {}
    for label, lang in [("ch_PP-OCRv4", "ch"), ("en_PP-OCRv4", "en")]:
        try:
            lines = _run(label, lang)
            sections[label] = lines
            pick = _pick_smi_row(label, lines)
            if pick is not None:
                chosen_per_engine[label] = pick
        except Exception as exc:
            sections[label] = {"error": str(exc)}

    (OUT / "topk_printed_smiles.json").write_text(
        json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    txt_parts = [
        f"# RapidOCR per-char top-k on printed SMILES rows from page {PAGE}",
        f"# rec_top_k = {TOP_K}, uncertain threshold = {UNCERTAIN_THRESHOLD}",
        "",
    ]
    for label in ("ch_PP-OCRv4", "en_PP-OCRv4"):
        if label not in chosen_per_engine:
            err = sections.get(label, {})
            err_msg = err.get("error", "(no row)") if isinstance(err, dict) else "(no row)"
            txt_parts.append(f"### engine: {label}\n{err_msg}")
        else:
            txt_parts.append(_render(chosen_per_engine[label], f"engine: {label}"))
        txt_parts.append("")
    (OUT / "topk_printed_smiles.txt").write_text(
        "\n".join(txt_parts), encoding="utf-8"
    )

    for label, lines in sections.items():
        if isinstance(lines, dict) and "error" in lines:
            print(f"[demo] {label}: ERROR ({lines['error']})")
            continue
        if not lines:
            print(f"[demo] {label}: (no rows >= {MIN_LEN} chars)")
            continue
        n_total = len(lines)
        n_uncertain = 0
        for lp in lines:
            for c in lp["chars"]:
                if c["top3"] and c["top3"][0]["prob"] < UNCERTAIN_THRESHOLD:
                    n_uncertain += 1
        print(f"[demo] {label}: {n_total} rows, {n_uncertain} total uncertain chars (top-1<{UNCERTAIN_THRESHOLD})")
    print(f"[demo] wrote {OUT / 'topk_printed_smiles.txt'} + .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
