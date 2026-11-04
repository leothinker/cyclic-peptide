"""Compare two reference-free combination rules on both patents.

``phase8_wo2022251372_line_vote.py`` showed that reading every printed
band with a second recogniser and choosing the combination that
RDKit can parse improves WO2022251372.  That search, however,
enumerates the combinations in an arbitrary order, and a *different*
combination that also parses could be picked even when the primary
reading is already correct.  This script compares the arbitrary order
against a **minimal-change** order:

``parse_any``
    the first combination that parses, in the order the bands are
    enumerated.
``parse_minimal``
    the combination that parses and changes the *fewest characters*
    of the primary reading.  When the primary reading already parses,
    it is the answer with zero changes.

The rule is applied to the primary route of each patent, so the
question is answerable without a reference:

* WO2022251372 -- primary: the ``padded`` route (English PP-OCRv4,
  6 px margin, no upscale), secondary: the default PP-OCRv6.
* WO2025162428 -- primary: the ``validated`` route (PP-OCRv6, 2x
  upscale, no margin), secondary: the English PP-OCRv4.

The reference is used only to score the result afterwards.

Usage::

    .venv\\Scripts\\python.exe experiments/phase8_vote_rule.py
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
from phase7_route_verify import (  # noqa: E402
    PAGES as PAGES_1,
    REFERENCES as REFERENCES_1,
)

RDLogger.DisableLog("rdApp.*")

OUT = REPO / "data" / "audit" / "phase8"
WHITE = (255, 255, 255)

# --- geometry ---------------------------------------------------------
PAD_TOP = 14
PAD_BOTTOM = 15


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


def _bands(cell: np.ndarray) -> list[np.ndarray]:
    """The printed bands of a cell, padded as the routes pad them."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    result: list[np.ndarray] = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        variant = row[:, x1:x2 + 1]
        if variant.size:
            result.append(variant)
    return result


def _read(
    engine: RapidOCR, cell: np.ndarray, margin: int, scale: int
) -> list[str]:
    """Read every band of a cell with one recogniser."""
    lines: list[str] = []
    for band in _bands(cell):
        variant = band
        if margin:
            variant = cv2.copyMakeBorder(
                variant, 0, 0, margin, margin,
                cv2.BORDER_CONSTANT, value=WHITE,
            )
        if scale != 1:
            variant = cv2.resize(
                variant, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
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


def _combinations(primary: list[str], secondary: list[str]):
    """Every combination, ordered by how much it changes the primary."""
    count = max(len(primary), len(secondary))
    options: list[list[str]] = []
    for index in range(count):
        first = primary[index] if index < len(primary) else ""
        second = secondary[index] if index < len(secondary) else ""
        options.append([first] if first == second else [first, second])

    results: list[tuple[int, tuple[str, ...]]] = []
    stack: list[tuple[int, tuple[str, ...], int]] = [(0, (), 0)]
    while stack:
        index, chosen, changes = stack.pop()
        if index == len(options):
            results.append((changes, chosen))
            continue
        for option in reversed(options[index]):
            cost = changes + (
                0
                if option == (
                    primary[index] if index < len(primary) else ""
                )
                else 1
            )
            stack.append((index + 1, chosen + (option,), cost))
    results.sort(key=lambda item: item[0])
    return ["".join(chosen) for _, chosen in results]


def _rule_parse_any(primary: list[str], secondary: list[str]) -> str:
    for text in _combinations(primary, secondary):
        if _parses(text):
            return text
    return "".join(primary)


def _rule_parse_minimal(
    primary: list[str], secondary: list[str]
) -> str:
    for text in _combinations(primary, secondary):
        if _parses(text):
            return text
    return "".join(primary)


RULES = {
    "primary only": lambda p, s: "".join(p),
    "parse_any": _rule_parse_any,
    "parse_minimal": _rule_parse_minimal,
}


def _score(name: str, rule, cases) -> None:
    print(f"\n{name}")
    print(f"{'rule':<16} {'exact':>12} {'edits':>8} {'CER %':>9}")
    for rule_name, function in RULES.items():
        exact = 0
        edits = 0
        length = 0
        for reference, primary, secondary in cases:
            chosen = function(primary, secondary)
            edits += edit_distance(reference, chosen)
            length += len(reference)
            if reference == chosen:
                exact += 1
        print(
            f"{rule_name:<16} {exact:>7}/{len(cases)} {edits:>8} "
            f"{100.0 * edits / max(length, 1):>8.3f}%"
        )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    en_v4 = _english_v4()
    default_v6 = _default_v6()

    # --- patent 2: the padded route is the primary -------------------
    cells = (
        REPO / "data" / "audit" / "phase4" / "wo2022251372"
        / "table1a_cells"
    )
    cases2 = []
    for compound_id, lines in REFERENCE_LINES.items():
        path = cells / CELL_FILES[compound_id]
        cell = cv2.imread(str(path))
        if cell is None:
            continue
        primary = _read(en_v4, cell, margin=6, scale=1)
        secondary = _read(default_v6, cell, margin=6, scale=1)
        cases2.append(("".join(lines), primary, secondary))
    _score("WO2022251372 (primary = padded)", RULES, cases2)

    # --- patent 1: the validated route is the primary ----------------
    cases1 = []
    for compound_id, reference in REFERENCES_1.items():
        page = PAGES_1[compound_id]
        path = (
            REPO / "data" / "raw" / "WO2025162428" / "golden_tables" / page
        )
        img = cv2.imread(str(path))
        if img is None:
            continue
        cell = _cell_of(img, compound_id)
        if cell is None:
            continue
        primary = _read(default_v6, cell, margin=0, scale=2)
        secondary = _read(en_v4, cell, margin=0, scale=2)
        cases1.append((reference, primary, secondary))
    _score("WO2025162428 (primary = validated)", RULES, cases1)
    return 0


def _cell_of(img: np.ndarray, compound_id: str):
    """The printed-SMILES cell of one compound on a patent-1 page."""
    import main as pipeline
    from cpd.parsers.profiles import get_profile

    profile = get_profile("WO2025162428")
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route="validated", profile=profile,
    )
    columns = extractor._detect_columns(img, [])
    rows = extractor._detect_row_boundaries(img)
    cid_col = next(c for c in columns if c.name == "cmpd_id")
    for y1, y2 in rows:
        cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
        if cid != compound_id:
            continue
        for col in columns:
            if col.name != "smiles":
                continue
            return img[y1:y2, col.x1:col.x2]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
