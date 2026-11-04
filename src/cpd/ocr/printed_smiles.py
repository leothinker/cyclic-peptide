"""Recognising the printed SMILES column of patent characterisation tables.

Three routes are implemented side by side.

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

``padded``
    The ink-band route of ``validated``, but with the *English*
    PP-OCRv4 recogniser and a 6 px white margin around every printed
    line and no upscale.  It was derived on patent WO2022251372
    (Table 1a, 28 compounds, serif type): ``text_span`` returns the
    first and last column that contains any ink, so the final glyph of
    a printed line ends up flush against the crop border and the
    recogniser mis-reads or drops it.  The margin restores the missing
    border without touching the pixels.

    Measured on the 60 hand-transcribed lines of that patent, the
    margin takes the total line edit count from 31 (no margin) to 5.
    The two patents need *opposite* settings, though: on
    WO2025162428 the same margin makes the result much worse
    (total edit 15 -> 95 on five reference rows), because that
    patent's rows are packed tightly enough that the ink band also
    contains fragments of the neighbouring lines.  The margin is
    therefore a separate route, not a change to the existing ones.

    The route deliberately does *not* upscale, unlike ``validated``.
    That was measured, not assumed: over the 28 Table 1a compounds,
    no upscale gives 21/28 bracket balanced, 17/28 RDKit-parseable
    and 19/28 after the repair chain, while 2x gives 20/28, 15/28 and
    17/28.  The edit distance to the six hand transcriptions is the
    same either way (0, 1, 0, 0, 2, 2), so the difference is in
    bracket balance rather than in character accuracy.

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

from cpd.ocr.lines import (
    INK_THRESHOLD,
    INTERPOLATION,
    MIN_BAND_HEIGHT,
    MIN_BAND_INK,
    RULE_HEIGHT_RATIO,
    UPSCALE,
    WHITE,
    ink_bands,
    text_span,
    upscale,
)

ROUTE_PRODUCTION = "production"
ROUTE_VALIDATED = "validated"
ROUTE_PADDED = "padded"
ROUTE_VALIDATED_VOTE = "validated_vote"
ROUTE_PADDED_VOTE = "padded_vote"
ROUTES = (
    ROUTE_PRODUCTION,
    ROUTE_VALIDATED,
    ROUTE_PADDED,
    ROUTE_VALIDATED_VOTE,
    ROUTE_PADDED_VOTE,
)

# Routes that segment the printed lines with our own ink-band
# segmentation instead of the detector.  They share the crop geometry;
# they differ in the recogniser and in the margin.
INK_BAND_ROUTES = (
    ROUTE_VALIDATED,
    ROUTE_PADDED,
    ROUTE_VALIDATED_VOTE,
    ROUTE_PADDED_VOTE,
)

# The routes that read every printed line with a second recogniser and
# let the two readings settle the characters they disagree on.
VOTE_ROUTES = (ROUTE_VALIDATED_VOTE, ROUTE_PADDED_VOTE)

# --- validated route geometry -------------------------------------------
PAD_TOP = 14
PAD_BOTTOM = 15

# --- padded route geometry ----------------------------------------------
# White border added left and right of every printed line, measured in
# the pixels of the original crop (i.e. before the 2x upscale).
PADDED_MARGIN_PX = 6

# The padded route recognises the band at its original size.  Measured
# on WO2022251372 (28 compounds): 1x gives 21/28 bracket balanced and
# 17/28 RDKit-parseable, 2x gives 20/28 and 15/28.  The character
# accuracy against the hand transcriptions is identical either way.
PADDED_SCALE = 1

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


def english_recogniser_only():
    """English PP-OCRv4 mobile recogniser, with no detector configured.

    The padded route never runs the detector -- it segments the lines
    itself and recognises them one by one -- so the detector model is
    pure overhead here.  This is the configuration the 6 px margin was
    measured with.
    """
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import (
        EngineType,
        ModelType,
        OCRVersion,
        TaskType,
    )

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


class PrintedSmilesOCR:
    """Read the printed SMILES out of one table cell."""

    def __init__(self, route: str = ROUTE_PRODUCTION) -> None:
        if route not in ROUTES:
            raise ValueError(
                f"unknown route {route!r}; expected one of {ROUTES}"
            )
        self.route = route
        self._engine = None
        self._second_engine = None

    @property
    def engine(self):
        """Lazily build the OCR engine (loading ONNX models is slow)."""
        if self._engine is None:
            if self.route in (ROUTE_VALIDATED, ROUTE_VALIDATED_VOTE):
                self._engine = default_recogniser()
            elif self.route in (ROUTE_PADDED, ROUTE_PADDED_VOTE):
                self._engine = english_recogniser_only()
            else:
                self._engine = english_recogniser()
        return self._engine

    @property
    def second_engine(self):
        """The second recogniser of the voting routes.

        The vote needs two recognisers that make *different* mistakes.
        The English PP-OCRv4 mobile model and the upstream PP-OCRv6
        model are such a pair: on the transcribed lines of
        WO2022251372 the first one misses 5 characters and the second
        one misses 29, and they miss different ones.
        """
        if self._second_engine is None:
            if self.route in (ROUTE_VALIDATED, ROUTE_VALIDATED_VOTE):
                self._second_engine = english_recogniser_only()
            else:
                self._second_engine = default_recogniser()
        return self._second_engine

    def read_cell(self, cell: np.ndarray) -> str | None:
        """Return the printed SMILES in one cell, or ``None`` if empty."""
        if cell is None or cell.size == 0:
            return None
        if self.route == ROUTE_VALIDATED_VOTE:
            return self._read_cell_voted(
                cell, margin=0, scale=UPSCALE
            )
        if self.route == ROUTE_PADDED_VOTE:
            return self._read_cell_voted(
                cell, margin=PADDED_MARGIN_PX, scale=PADDED_SCALE
            )
        if self.route == ROUTE_VALIDATED:
            return self._read_cell_validated(cell)
        if self.route == ROUTE_PADDED:
            return self._read_cell_padded(cell)
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
        return self._read_cell_ink_bands(cell, margin=0, scale=UPSCALE)

    def _read_cell_padded(self, cell: np.ndarray) -> str | None:
        """Ink bands plus a white margin around every printed line.

        The pixels are unchanged; only a white border is added on the
        left and the right of each line, so the final glyph of the line
        no longer touches the crop border.  The band is recognised at
        its original size; see ``PADDED_SCALE``.
        """
        return self._read_cell_ink_bands(
            cell, margin=PADDED_MARGIN_PX, scale=PADDED_SCALE
        )

    def _read_cell_voted(
        self, cell: np.ndarray, *, margin: int, scale: int
    ) -> str | None:
        """Read the printed lines twice and let the two readings vote.

        The geometry is the one of the route this vote was derived
        from -- the margin and the upscale are the measured settings
        of that patent.  Only the *reading* differs: every printed
        line is recognised by the route's own recogniser and by the
        other one, and the two readings are combined by
        :func:`cpd.ocr.vote.vote_lines`.
        """
        from cpd.ocr.vote import vote_lines

        primary = self._read_cell_ink_bands(
            cell, margin=margin, scale=scale, engine=self.engine,
            as_lines=True,
        )
        if not primary:
            return None
        secondary = self._read_cell_ink_bands(
            cell, margin=margin, scale=scale, engine=self.second_engine,
            as_lines=True,
        )
        result = vote_lines(primary, secondary)
        return result.text or None

    def _read_cell_ink_bands(
        self,
        cell: np.ndarray,
        *,
        margin: int,
        scale: int,
        engine=None,
        as_lines: bool = False,
    ) -> str | None:
        """Shared ink-band reader for the ``validated``/``padded`` routes."""
        if engine is None:
            engine = self.engine
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
            if margin:
                variant = cv2.copyMakeBorder(
                    variant,
                    0,
                    0,
                    margin,
                    margin,
                    cv2.BORDER_CONSTANT,
                    value=WHITE,
                )
            if scale != 1:
                variant = upscale(variant, scale)
            text = self._recognise_only(variant, engine=engine)
            if text:
                parts.append(text)
        if not parts:
            return None
        if as_lines:
            return parts  # type: ignore[return-value]
        return "".join(parts)

    def _recognise_only(self, crop: np.ndarray, engine=None) -> str:
        if engine is None:
            engine = self.engine
        result = engine(crop, use_det=False, use_rec=True)
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
