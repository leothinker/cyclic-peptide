"""Recognising the printed SMILES column of patent characterisation tables.

Two routes are implemented side by side.

``production`` (the historical pipeline behaviour)
    Crop the SMILES cell, trim it to its content, upscale 2x, then run
    the full detector + recogniser pipeline with the *English*
    PP-OCRv4 mobile recogniser.  The English recogniser systematically
    drops the left bracket of ``C(=O)`` -- it returns ``C=O`` -- so
    every printed SMILES in the shipped CSV has unbalanced brackets
    (0 of 335 parse with RDKit).

``validated``
    The route validated on patent page I100298 (21 printed rows of
    1038-1041, edit distance 0 on all of them, with 1038 held out of
    the configuration search).  Three things differ from ``production``:

    1. the *recogniser* is the upstream default ``PP-OCRv6_rec_small``
       instead of the English PP-OCRv4 mobile model;
    2. the text lines are found by our own ink-band segmentation rather
       than by the detector, and recognition runs in recognition-only
       mode (``use_det=False``);
    3. the crop geometry is fixed: the full-height vertical table rules
       are trimmed away, and each printed ink band is padded by 14/15 px
       vertically so the glyphs are no longer squeezed against the
       neighbouring rows.

    One single configuration is used for every row of every compound --
    there is no per-row or per-compound tuning.

What this module does **not** do: it does not judge whether the
recognised string is the correct chemical structure.  Text recognition,
RDKit parseability, the LC-MS mass gate and structure validation are
four separate levels and must not be treated as interchangeable.

Command line usage::

    .venv\\Scripts\\python.exe -m cpd.ocr.printed_smiles
        --cells data/audit/phase2/cells_text
        --route validated
        --out data/audit/printed_smiles_cells.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROUTE_PRODUCTION = "production"
ROUTE_VALIDATED = "validated"
ROUTES = (ROUTE_PRODUCTION, ROUTE_VALIDATED)

# --- validated route geometry -------------------------------------------
UPSCALE = 2
INTERPOLATION = cv2.INTER_CUBIC
PAD_TOP = 14
PAD_BOTTOM = 15
RULE_HEIGHT_RATIO = 0.80
INK_THRESHOLD = 160
MIN_BAND_HEIGHT = 8
MIN_BAND_INK = 5


def default_recogniser():
    """Upstream default RapidOCR engine (PP-OCRv6 detector + recogniser).

    Only the recogniser half is used by the validated route; the
    detector model is irrelevant because recognition runs with
    ``use_det=False``.
    """
    from rapidocr import RapidOCR

    return RapidOCR(params={})


def english_recogniser():
    """The production recogniser: English PP-OCRv4 mobile."""
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import (
        EngineType,
        ModelType,
        OCRVersion,
        TaskType,
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


def ink_bands(
    gray: np.ndarray,
    *,
    threshold: int = INK_THRESHOLD,
    min_height: int = MIN_BAND_HEIGHT,
    min_ink: int = MIN_BAND_INK,
) -> list[tuple[int, int]]:
    """Vertical bands of printed text, as ``(top, bottom)`` pixel rows.

    The left and right edges of the cell are ignored so that the
    vertical table rules cannot keep a band alive on their own.
    """
    height, width = gray.shape
    left = min(10, max(1, width // 100))
    right = max(left + 1, width - 10)
    ink = (gray[:, left:right] < threshold).astype(np.uint8)
    sums = ink.sum(axis=1)
    active = np.flatnonzero(sums > min_ink)
    if not len(active):
        return []

    groups: list[list[int]] = []
    start = previous = int(active[0])
    for value in active[1:]:
        current = int(value)
        if current > previous + 1:
            groups.append([start, previous])
            start = current
        previous = current
    groups.append([start, previous])
    return [(a, b) for a, b in groups if b - a >= min_height]


def text_span(
    gray: np.ndarray,
    *,
    threshold: int = INK_THRESHOLD,
    rule_height_ratio: float = RULE_HEIGHT_RATIO,
) -> tuple[int, int]:
    """Horizontal ink span of a row, with the vertical table rules removed.

    A patent table rule is a column that is dark over (almost) the whole
    cell height.  Such columns are excluded, and so are the one-pixel
    neighbours that belong to the same rule.
    """
    dark = gray < threshold
    height = gray.shape[0]
    rule = dark.sum(axis=0) / max(height, 1) >= rule_height_ratio
    for x in np.flatnonzero(rule).tolist():
        for dx in (-1, 1):
            if 0 <= x + dx < gray.shape[1]:
                if dark[:, x + dx].sum() / max(height, 1) >= 0.5:
                    rule[x + dx] = True
    columns = np.flatnonzero((dark.sum(axis=0) > 0) & (~rule))
    if not len(columns):
        return 0, gray.shape[1] - 1
    return int(columns[0]), int(columns[-1])


def upscale(image: np.ndarray, scale: int = UPSCALE) -> np.ndarray:
    """Upscale without binarisation (binarisation was measured to hurt)."""
    return cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=INTERPOLATION
    )


class PrintedSmilesOCR:
    """Read the printed SMILES out of one table cell."""

    def __init__(self, route: str = ROUTE_PRODUCTION) -> None:
        if route not in ROUTES:
            raise ValueError(
                f"unknown route {route!r}; expected one of {ROUTES}"
            )
        self.route = route
        self._engine = None

    @property
    def engine(self):
        """Lazily build the OCR engine (loading ONNX models is slow)."""
        if self._engine is None:
            if self.route == ROUTE_VALIDATED:
                self._engine = default_recogniser()
            else:
                self._engine = english_recogniser()
        return self._engine

    def read_cell(self, cell: np.ndarray) -> str | None:
        """Return the printed SMILES in one cell, or ``None`` if empty."""
        if cell is None or cell.size == 0:
            return None
        if self.route == ROUTE_VALIDATED:
            return self._read_cell_validated(cell)
        return self._read_cell_production(cell)

    def read_cells(self, directory: Path) -> dict[str, str]:
        """Read every ``*.png`` cell crop in ``directory``.

        The file stem is used as the key, e.g. ``I100298_1038_text``.
        """
        results: dict[str, str] = {}
        for path in sorted(Path(directory).glob("*.png")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            text = self.read_cell(image)
            if text:
                results[path.stem] = text
        return results

    def _read_cell_production(self, cell: np.ndarray) -> str | None:
        """Historical behaviour: trim, upscale, detect + recognise.

        The trim and upscale helpers are borrowed from the pipeline's
        own extractor so that this route stays bit-for-bit identical to
        what the pipeline has always produced; there is deliberately no
        second, drifting copy of that logic here.
        """
        from cpd.parsers.table_extractor import RapidOcrTableExtractor

        trimmed = RapidOcrTableExtractor._trim_to_content(cell)
        if trimmed.size == 0:
            return None
        result = self.engine(RapidOcrTableExtractor._upscale(trimmed))
        if not result:
            return None
        txts = getattr(result, "txts", None) or []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) == len(txts):
            items = sorted(
                zip(boxes, txts),
                key=lambda bt: (bt[0][0][1] + bt[0][2][1]) / 2,
            )
            parts = [t.strip() for _, t in items if t and t.strip()]
        else:
            parts = [t.strip() for t in txts if t and t.strip()]
        if not parts:
            return None
        return "".join(parts)

    def _read_cell_validated(self, cell: np.ndarray) -> str | None:
        """Validated behaviour: ink bands, trimmed rules, padded, rec-only."""
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        bands = ink_bands(gray)
        if not bands:
            return None

        parts: list[str] = []
        for top, bottom in bands:
            y1 = max(0, top - PAD_TOP)
            y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
            row = cell[y1:y2, :]
            row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
            x1, x2 = text_span(row_gray)
            variant = row[:, x1:x2 + 1]
            if variant.size == 0:
                continue
            text = self._recognise_only(upscale(variant))
            if text:
                parts.append(text)
        if not parts:
            return None
        return "".join(parts)

    def _recognise_only(self, crop: np.ndarray) -> str:
        result = self.engine(crop, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        if not txts:
            return ""
        return (txts[0] or "").strip()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read printed SMILES from cropped table cells.",
    )
    parser.add_argument(
        "--cells",
        type=Path,
        required=True,
        help="Directory of cropped cell images (*.png).",
    )
    parser.add_argument(
        "--route",
        choices=ROUTES,
        default=ROUTE_VALIDATED,
        help="OCR route to use (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write a CSV of stem -> recognised text to this path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the same content as JSON to this path.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    reader = PrintedSmilesOCR(route=args.route)
    results = reader.read_cells(args.cells)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell", "route", "length", "smiles"])
            for stem, text in results.items():
                writer.writerow([stem, args.route, len(text), text])
        print(f"wrote {args.out} ({len(results)} cells)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json} ({len(results)} cells)")
    if not args.out and not args.json:
        for stem, text in results.items():
            print(f"{stem}\t{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
