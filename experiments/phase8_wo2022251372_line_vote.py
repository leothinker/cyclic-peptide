"""Can two recognisers together read WO2022251372 exactly?

The ``padded`` route reaches 56 of 60 printed lines character-exact.
The four that remain are single-character confusions of a kind the
recogniser cannot resolve on its own:

======  ==============================  ==============================
cmpd    printed                         read by the shipped route
======  ==============================  ==============================
2       ``@@H](C)OC(=O)c1ccccc1``      ``@@H](C)OC(=O)c1ccccc``
10      ``@@H](C)N(C)C(=O)[C@H](C``    ``@H](C)N(C)C(=O)[C@H]C``
28      ``(C)N(C)C(=O)[C@@H](CCO``      ``(C)N(C)C(=O)[C@H](CCO``
28      ``(=O)[C@H](Cc2ccccc2)NC(C``   ``(=O)[C@H](Cc2ccccc2)N(C``
======  ==============================  ==============================

The patent prints every compound **twice** -- once in Table 1 and
once in Table 1a -- and the pipeline already knows about both copies.
The question this script answers is narrower: if a second, different
recogniser reads the same printed line, does it recover the characters
the first one lost?

Two recognisers are used:

* the English PP-OCRv4 mobile recogniser, which is what the shipped
  ``padded`` route uses;
* the upstream default PP-OCRv6 recogniser, which is what the shipped
  ``validated`` route uses.

**Every rule below is reference-free.**  A rule may look at the two
readings, at their lengths, at the bracket balance of the compound,
or at whether RDKit can parse the result -- never at the hand
transcription.  The transcription is used only to *score* the rules
afterwards, which is the only honest way to use it.

The rules are:

``v4``
    the shipped behaviour: always take the English reading.
``v6``
    always take the default reading.
``longer``
    take the longer of the two readings of the band.
``shorter``
    take the shorter of the two.
``parse``
    take the reading that makes the whole compound parse.  When both
    do, or neither does, the English reading wins.
``parse_then_v6``
    as ``parse``, but when neither choice parses, a bracket-balanced
    combination is preferred, and the default reading is the last
    resort.

The compound-level numbers are reported for the six hand-transcribed
compounds.  For all 28 compounds of Table 1a, where no transcription
exists, the reference-free quality measures are reported instead:
bracket balance, RDKit parseability and the total length.

Usage::

    .venv\\Scripts\\python.exe experiments/phase8_wo2022251372_line_vote.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import (
    EngineType,
    ModelType,
    OCRVersion,
    TaskType,
)
from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from cpd.ocr.lines import ink_bands, text_span  # noqa: E402
from wo2022251372_references import (  # noqa: E402
    CELL_FILES,
    REFERENCE_LINES,
    edit_distance,
)

RDLogger.DisableLog("rdApp.*")

CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase8"

WHITE = (255, 255, 255)

# The geometry the shipped ``padded`` route uses.
PAD_TOP = 14
PAD_BOTTOM = 15
MARGIN_PX = 6


def _english_v4() -> RapidOCR:
    return RapidOCR(
        params={
            "Rec.lang_type": "en",
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.0,
            "Global.use_cls": False,
        }
    )


def _default_v6() -> RapidOCR:
    return RapidOCR(
        params={
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.0,
            "Global.use_cls": False,
        }
    )


def _recognise_only(engine: RapidOCR, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def _read_lines(
    engine: RapidOCR, cell: np.ndarray, margin: int
) -> list[str]:
    """Read every printed line of a cell, one band at a time."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bands = ink_bands(gray)
    lines: list[str] = []
    for top, bottom in bands:
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        variant = row[:, x1:x2 + 1]
        if variant.size == 0:
            lines.append("")
            continue
        if margin:
            variant = cv2.copyMakeBorder(
                variant, 0, 0, margin, margin,
                cv2.BORDER_CONSTANT, value=WHITE,
            )
        lines.append(_recognise_only(engine, variant))
    return lines


def _parses(text: str) -> bool:
    if not text:
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _balanced(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


def _product(choices: list[list[str]]):
    """Every combination of the per-band choices, English first."""
    result: list[tuple[str, ...]] = [()]
    for options in choices:
        result = [
            prefix + (option,)
            for prefix in result
            for option in options
        ]
    return result


# --- the reference-free rules -----------------------------------------


def _rule_v4(parts: list[tuple[str, str]]) -> str:
    return "".join(v4 for v4, _ in parts)


def _rule_v6(parts: list[tuple[str, str]]) -> str:
    return "".join(v6 for _, v6 in parts)


def _rule_longer(parts: list[tuple[str, str]]) -> str:
    return "".join(
        v4 if len(v4) >= len(v6) else v6 for v4, v6 in parts
    )


def _rule_shorter(parts: list[tuple[str, str]]) -> str:
    return "".join(
        v4 if len(v4) <= len(v6) else v6 for v4, v6 in parts
    )


def _rule_parse(parts: list[tuple[str, str]]) -> str:
    """Prefer the per-band choice that makes the compound parse.

    The search is exhaustive over the bands that differ, and it is
    deterministic: the assignments are enumerated with the English
    reading first, so a tie goes to the English reading.
    """
    choices = [
        [v4] if v4 == v6 else [v4, v6] for v4, v6 in parts
    ]
    for combination in _product(choices):
        text = "".join(combination)
        if _parses(text):
            return text
    return _rule_v4(parts)


def _rule_parse_then_v6(parts: list[tuple[str, str]]) -> str:
    """As ``parse``, but fall back to a balanced and then to the default."""
    choices = [
        [v4] if v4 == v6 else [v4, v6] for v4, v6 in parts
    ]
    for combination in _product(choices):
        text = "".join(combination)
        if _parses(text):
            return text
    for combination in _product(choices):
        text = "".join(combination)
        if _balanced(text):
            return text
    return _rule_v6(parts)


RULES = {
    "v4": _rule_v4,
    "v6": _rule_v6,
    "longer": _rule_longer,
    "shorter": _rule_shorter,
    "parse": _rule_parse,
    "parse_then_v6": _rule_parse_then_v6,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engines = {
        "en_v4": _english_v4(),
        "default_v6": _default_v6(),
    }

    # --- every cell of Table 1a ---------------------------------------
    readings: dict[str, dict[str, list[str]]] = {}
    for cell_path in sorted(CELLS.glob("*.png")):
        cell = cv2.imread(str(cell_path))
        if cell is None:
            continue
        readings[cell_path.stem] = {
            name: _read_lines(engine, cell, MARGIN_PX)
            for name, engine in engines.items()
        }
    print(f"{len(readings)} cells read by both recognisers")

    # --- the six transcribed compounds -------------------------------
    rows: list[dict[str, str]] = []
    for compound_id, lines in REFERENCE_LINES.items():
        stem = Path(CELL_FILES[compound_id]).stem
        cell_readings = readings.get(stem)
        if cell_readings is None:
            continue
        for index, reference in enumerate(lines):
            v4 = cell_readings["en_v4"][index] if index < len(
                cell_readings["en_v4"]) else ""
            v6 = cell_readings["default_v6"][index] if index < len(
                cell_readings["default_v6"]) else ""
            rows.append({
                "cmpd_id": compound_id,
                "line_index": index,
                "reference": reference,
                "en_v4": v4,
                "default_v6": v6,
            })

    out_path = OUT / "line_vote.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} transcribed lines -> {out_path}")

    # --- score the rules against the transcription --------------------
    print(f"\nline level ({len(rows)} transcribed lines)")
    print(f"{'rule':<16} {'exact':>12} {'edits':>8}")
    for name, rule in RULES.items():
        exact = 0
        edits = 0
        for row in rows:
            chosen = rule([(row["en_v4"], row["default_v6"])])
            distance = edit_distance(row["reference"], chosen)
            edits += distance
            if distance == 0:
                exact += 1
        print(f"{name:<16} {exact:>7}/{len(rows)} {edits:>8}")

    print("\ncompound level")
    print(f"{'rule':<16} {'exact':>12} {'edits':>8} {'CER %':>9}")
    for name, rule in RULES.items():
        by_compound: dict[int, list[tuple[str, str]]] = {}
        for row in rows:
            by_compound.setdefault(row["cmpd_id"], []).append(
                (row["en_v4"], row["default_v6"])
            )
        exact = 0
        total = 0
        length = 0
        for compound_id, parts in by_compound.items():
            reference = "".join(REFERENCE_LINES[compound_id])
            chosen = rule(parts)
            total += edit_distance(reference, chosen)
            length += len(reference)
            if reference == chosen:
                exact += 1
        print(
            f"{name:<16} {exact:>7}/{len(by_compound)} {total:>8} "
            f"{100.0 * total / max(length, 1):>8.3f}%"
        )

    # --- reference-free quality over all 28 compounds ------------------
    print(f"\nall {len(readings)} compounds of Table 1a "
          f"(reference-free measures)")
    print(f"{'rule':<16} {'parses':>12} {'balanced':>12} {'chars':>9}")
    for name, rule in RULES.items():
        parses = 0
        balanced = 0
        characters = 0
        for cell_readings in readings.values():
            v4 = cell_readings["en_v4"]
            v6 = cell_readings["default_v6"]
            count = max(len(v4), len(v6))
            parts = [
                (
                    v4[i] if i < len(v4) else "",
                    v6[i] if i < len(v6) else "",
                )
                for i in range(count)
            ]
            text = rule(parts)
            characters += len(text)
            if _parses(text):
                parses += 1
            if _balanced(text):
                balanced += 1
        print(
            f"{name:<16} {parses:>7}/{len(readings)} "
            f"{balanced:>7}/{len(readings)} {characters:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
