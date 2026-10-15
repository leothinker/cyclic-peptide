"""Unified table extractor producing compounds + assays from patent JPGs.

Three table layouts appear in WO2025162428's golden tables, and the
extractor handles them all with one code path:

  * 3-col SMILES table           (Cmpd # | Structure | SMILES)
  * 7-col activity table         (Cmpd # | Structure | RT | MS | Method | Polarity | KD)
  * 6-col dense activity table   (Compound # | RT | MS | Method | Polarity | KD)

Pipeline for every image:

  1. Detect the layout. First try to OCR the top header band and look
     for canonical field names (``SMILES``, ``KD (nM)``, ...). If that
     finds nothing (continuation pages usually have no header at all)
     fall back to **counting the visible vertical dividers** in the
     image body -- 2 inner dividers means 3 columns (smiles), 6 means 7
     (activity), 5 means 6 (dense). As a final safety net, reuse the
     previous image's columns when this is also inconclusive.
  2. Use the layout's predefined column widths (fractions of image
     width). We deliberately do **not** use individual word bounding
     boxes as column edges -- the word "Structure" is much narrower
     than the structure column itself, and using word widths would
     shrink the cropped cell to just the header text.
  3. Detect horizontal row boundaries with OpenCV morphology. The
     first inner horizontal line marks the bottom of the header band;
     data rows start there, so the first row slice is the first
     *data* row, not the header itself. The first slice may also be
     a single trailing ``[cmpd_id | structure | SMILES]`` row above
     a dense activity table (see ``_detect_first_row_columns``).
  4. For each row, OCR every text cell; for the ``structure`` column,
     crop the cell to a PNG under ``cells_dir``.

Output is **two flat lists**, joined on ``cmpd_id`` by the caller:

  * :class:`CompoundRecord` carries ``cmpd_id``, ``smiles``, RDKit
    canonical form, descriptors, and the cropped structure-image path.
  * :class:`AssayRecord`  carries ``cmpd_id``, ``kd_nm``, ``rt_min``,
    ``ms_mz``, ``lcms_method``, ``ms_polarity``.

Continuation pages without a header (``I100382``, ``I100330``) reuse
the visual column structure (or the previous image's layout when even
the dividers are ambiguous). Header-only images (``I100287``) emit
zero records and are filtered out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from cpd.parsers.headers import canonical_field


# ---------- row schemas ----------


@dataclass(frozen=True)
class CompoundRecord:
    """One cyclic peptide: SMILES, RDKit descriptors, cropped structure image."""

    cmpd_id: str
    smiles: str | None = None
    canonical_smiles: str | None = None
    molecular_weight: float | None = None
    heavy_atom_count: int | None = None
    is_valid: bool | None = None
    structure_image: str | None = None  # path relative to cells_dir parent
    source_image: str | None = None
    # OCR trust score from the SMILES-cell top-k distribution. All three
    # stay None when the recogniser was built with rec_top_k=0.
    ocr_avg_entropy: float | None = None
    ocr_min_top1: float | None = None
    ocr_low_conf_count: int | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class AssayRecord:
    """One activity measurement (KD + LCMS) for a cyclic peptide."""

    cmpd_id: str
    kd_nm: float | None = None
    rt_min: float | None = None
    ms_mz: float | None = None
    lcms_method: str | None = None
    ms_polarity: str | None = None
    source_image: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class OcrStats:
    """Per-cell OCR trust score from the SMILES-cell top-k distribution.

    ``avg_entropy`` is the mean of ``-sum(p * log2(p))`` over every
    character position in the cell. Higher entropy = the recogniser
    spread probability across many candidates = less trustworthy.
    ``min_top1`` is the lowest top-1 probability seen across the cell
    (0.0 if no top-k data is available). ``low_conf_count`` is the
    number of character positions whose top-1 probability is below
    ``LOW_CONF_PROB_THRESHOLD`` (0.7) -- a count of "I had to guess"
    moments that the swap-based fix cannot always repair.
    """

    avg_entropy: float | None = None
    min_top1: float | None = None
    low_conf_count: int | None = None


# Top-1 probability below this is counted as a low-confidence character.
# Empirically PP-OCRv4 returns top-1 ~0.99 for confident characters and
# ~0.5-0.7 for genuinely ambiguous ones (0/O, 1/l, etc.).
LOW_CONF_PROB_THRESHOLD = 0.7
# ---------- column geometry ----------


@dataclass(frozen=True)
class Column:
    """A column defined by horizontal pixel slice ``[x1, x2)`` in the source image."""

    name: str
    x1: int
    x2: int
    is_image: bool = False  # True for the structure column (saved as PNG)

    @property
    def width(self) -> int:
        return self.x2 - self.x1


# Predefined column widths (fractions of image width). These are
# hand-tuned on the WO2025162428 golden tables; update them here if a
# new patent shows a different ratio.
_ACTIVITY_FRACS: tuple[tuple[str, float, float, bool], ...] = (
    ("cmpd_id", 0.00, 0.09, False),
    ("structure", 0.10, 0.48, True),
    ("rt_min", 0.48, 0.57, False),
    ("ms_mz", 0.58, 0.67, False),
    ("lcms_method", 0.67, 0.78, False),
    ("ms_polarity", 0.79, 0.88, False),
    ("kd_nm", 0.89, 1.00, False),
)
_DENSE_ACTIVITY_FRACS: tuple[tuple[str, float, float, bool], ...] = (
    ("cmpd_id", 0.00, 0.17, False),
    ("rt_min", 0.19, 0.34, False),
    ("ms_mz", 0.35, 0.50, False),
    ("lcms_method", 0.51, 0.67, False),
    ("ms_polarity", 0.68, 0.79, False),
    ("kd_nm", 0.80, 0.97, False),
)
_SMILES_FRACS: tuple[tuple[str, float, float, bool], ...] = (
    ("cmpd_id", 0.00, 0.10, False),
    ("structure", 0.11, 0.56, True),
    ("smiles", 0.58, 0.99, False),
)


def _scale_fracs(width: int, fracs: Sequence[tuple[str, float, float, bool]]) -> list[Column]:
    return [
        Column(name, int(x1 * width), int(x2 * width), is_image)
        for name, x1, x2, is_image in fracs
    ]


def default_columns(width: int, layout: str = "activity") -> list[Column]:
    """Return the predefined column layout for a given table kind."""
    if layout == "smiles":
        return _scale_fracs(width, _SMILES_FRACS)
    if layout == "dense":
        return _scale_fracs(width, _DENSE_ACTIVITY_FRACS)
    return _scale_fracs(width, _ACTIVITY_FRACS)


def detect_layout(fields: set[str]) -> str | None:
    """Pick a layout string from the canonical field names found in the header.

    Returns ``None`` when no canonical field names are recognised, so the
    caller can fall back to visual column-count detection or the previous
    page's layout (continuation pages often have no header text).
    """
    if not fields:
        return None
    if "smiles" in fields:
        return "smiles"
    if "kd_nm" in fields and "structure" not in fields:
        return "dense"
    if "kd_nm" in fields:
        return "activity"
    return None  # unknown mix -> let caller fall back


def detect_layout_from_columns(n_columns: int) -> str | None:
    """Pick a layout from the count of visible columns in the image body.

    Columns here means "data columns between full-height vertical lines";
    i.e. ``n_dividers + 1``. The three target layouts have very
    distinctive counts so the mapping is unambiguous:

      * 3 columns  -> smiles
      * 7 columns  -> activity (with structure)
      * 6 columns  -> dense activity (no structure)

    Anything else returns ``None`` and the caller keeps the previous
    page's layout. The off-by-one counts ``5`` and ``8`` are deliberately
    mapped to ``None``: a dense table whose morphology picks up one extra
    stroke and an activity table whose morphology drops one divider are
    both plausible, and the layout that the column gets mis-applied to
    (e.g. activity onto a dense page) is far more harmful than trusting
    the previous image's layout. So we let the caller fall back to
    ``prev_columns`` instead of guessing.
    """
    if n_columns == 3:
        return "smiles"
    if n_columns == 7:
        return "activity"
    if n_columns == 6:
        return "dense"
    return None


# ---------- main extractor ----------


class RapidOcrTableExtractor:
    """OpenCV grid detection + RapidOCR per-cell extraction."""

    HEADER_FRAC: float = 0.12       # top slice used for header OCR
    ROW_LINE_GAP_PX: int = 10       # two horizontal lines within this gap merge
    MIN_ROW_PX: int = 30            # rows shorter than this are dropped
    MIN_HEADER_PX: int = 40         # floor on header-band height (avoid 0-px crops)
    COLUMN_GAP_PX: int = 25         # two vertical lines within this gap merge
    COLUMN_BODY_FRAC: float = 0.15  # skip header band when counting dividers
    COLUMN_BORDER_PX: int = 50      # ignore page borders when counting dividers
    BORDER_SNAP_PX: int = 90        # max px a column edge can be moved by snapping
    FIRST_ROW_MIN_PX: int = 50      # skip first-row probe when the slice is thinner
    ROW_BORDER_PX: int = 8          # horizontal-line border band on each side

    def __init__(self, rec_lang: str = 'en', rec_top_k: int = 0) -> None:
        """Build the OCR engine(s).

        ``rec_lang`` selects the language of the *recogniser*:

        * ``'en'`` (default) -- Chinese detector (best boxes on dense
          patent text) + English recogniser (clean ASCII dictionary,
          much better for SMILES / numbers / parentheses). This is the
          hybrid setting validated in our golden-table experiments.
        * ``'ch'`` -- both detector and recogniser in Chinese; the
          default in the upstream RapidOCR package. Cell-level SMILES
          often comes out wrong (``C`` vs ``(``, ``0`` vs ``O``).

        The detector (always Chinese) and recogniser are separate
        engines so that header OCR can still see Chinese field names
        while every cell-level OCR call goes through the recogniser
        with the ASCII-friendly dictionary.

        ``rec_top_k`` (default ``0``) is the per-character top-k
        returned by the recogniser. When ``> 1``, SMILES cells are
        re-validated against RDKit across every top-k swap and the
        first canonical SMILES that parses wins. The legacy
        single-argmax behaviour is preserved when ``rec_top_k == 0``.
        """
        from rapidocr import RapidOCR  # local import keeps the module lightweight
        # Detector is always the Chinese model -- it has the cleanest
        # box proposals on dense patent text. Cell-level OCR runs the
        # full pipeline so the recogniser always sees a tight
        # text-line crop, never a sparse cell.
        self.engine = RapidOCR()
        self.rec_lang = rec_lang
        if rec_lang == 'en':
            from rapidocr.utils.typings import (
                EngineType, ModelType, OCRVersion, TaskType,
            )
            self.rec_engine = RapidOCR(params={
                'Det.lang_type': 'en',
                'Det.ocr_version': OCRVersion.PPOCRV4,
                'Det.model_type': ModelType.MOBILE,
                'Det.engine_type': EngineType.ONNXRUNTIME,
                'Det.task_type': TaskType.DET,
                'Rec.lang_type': 'en',
                'Rec.ocr_version': OCRVersion.PPOCRV4,
                'Rec.model_type': ModelType.MOBILE,
                'Rec.engine_type': EngineType.ONNXRUNTIME,
                'Rec.task_type': TaskType.REC,
                'Global.text_score': 0.0,
                'Global.use_cls': False,
            })
        else:
            self.rec_engine = self.engine

        # Attach the top-k parameter to whichever recogniser we will
        # use for SMILES cells. The plumbing goes through three places
        # because the patched RapidOCR reads top_k at three different
        # moments: the ``RapidOCR.__init__`` captures ``return_top_k``
        # from ``cfg.Global`` into a Python attribute, the
        # ``recognize_txt`` method then copies it into ``cfg.Rec.top_k``
        # for the lazy-loaded ``TextRecognizer``, and the patched
        # ``TextRecognizer.__call__`` reads it via ``cfg.get("top_k", 1)``.
        # Setting all three keeps every code path consistent. 0 keeps the
        # original argmax behaviour (cheaper, no top-k array materialised).
        self.rec_top_k = max(0, int(rec_top_k))
        if self.rec_top_k > 1:
            self.rec_engine.return_top_k = self.rec_top_k
            self.rec_engine.cfg.Global.return_top_k = self.rec_top_k
            self.rec_engine.cfg.Rec["top_k"] = self.rec_top_k

    # ---- public API ----

    def extract_records(
        self,
        image_path: Path,
        cells_dir: Path | None = None,
        prev_columns: list[Column] | None = None,
    ) -> tuple[list[CompoundRecord], list[AssayRecord], list[Column]]:
        """Extract compounds + assays from one table image."""
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0 or min(img.shape[:2]) < self.MIN_HEADER_PX:
            return [], [], prev_columns or []

        columns = self._detect_columns(img, prev_columns)
        if not columns:
            columns = prev_columns or default_columns(img.shape[1], "activity")

        rows = self._detect_row_boundaries(img)
        # Single-row continuation pages (often a wide 70-px strip with
        # one cmpd row) make the vertical-line finder unreliable: it
        # tends to over-count dividers (the 2173 row on I100382 was
        # getting 7 detected columns instead of the dense 6). When we
        # have the previous image's layout, trust it for these thin
        # pages so the column widths and the cmpd_id x-range carry
        # over correctly.
        if prev_columns and len(rows) <= 1:
            columns = list(prev_columns)
        compounds, assays = self._extract_rows(
            img, rows, columns, image_path, cells_dir
        )
        return compounds, assays, columns

    # ---- column / layout detection ----

    def _detect_columns(
        self,
        img: np.ndarray,
        prev_columns: list[Column] | None = None,
    ) -> list[Column]:
        """Pick a column layout for this image.

        Strategy, in order: header OCR -> visual column count -> reuse
        ``prev_columns``. The three targets are mutually exclusive so
        the fallback chain only fires when the previous step genuinely
        produced no signal.
        """
        h, w = img.shape[:2]
        y_end = max(self.MIN_HEADER_PX, int(h * self.HEADER_FRAC))
        header = img[0:y_end, :] if y_end > 0 else None

        layout: str | None = None
        if header is not None and header.size:
            layout = detect_layout(self._scan_header_fields(header))
        if layout is None:
            layout = detect_layout_from_columns(self._count_columns(img))
        if layout is None and prev_columns:
            return list(prev_columns)
        if layout is None:
            layout = "activity"  # last-resort default
        return self._snap_to_borders(img, default_columns(w, layout))

    def _scan_header_fields(self, header_img: np.ndarray) -> set[str]:
        """Return the set of canonical field names found in the header band."""
        result = self.engine(self._upscale(header_img))
        if not result:
            return set()
        return {
            field
            for text in self._txts(result)
            if (field := canonical_field(text)) is not None
        }

    def _body_slice(self, img: np.ndarray) -> np.ndarray:
        """Return the image body (below the header band)."""
        h = img.shape[0]
        return img[int(h * self.COLUMN_BODY_FRAC):h, :]

    def _vertical_lines_in(
        self,
        slice_img: np.ndarray,
        *,
        threshold_frac: float = 0.4,
    ) -> list[int]:
        """Return x positions of vertical lines spanning ~threshold_frac of the slice.

        A tall vertical morphological kernel collapses short text strokes
        into noise and keeps only rules that span most of the slice
        height; the resulting clusters are filtered against the page
        borders. This is the single source of truth for every vertical
        divider detector in this class.
        """
        if slice_img.size == 0:
            return []
        h, w = slice_img.shape[:2]
        gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, int(h * 0.4)))
        )
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        xs = np.where(np.sum(lines, axis=0) > int(h * threshold_frac))[0]
        if len(xs) == 0:
            return []
        clusters = self._cluster(xs, gap=self.COLUMN_GAP_PX)
        return [
            x for x in clusters
            if self.COLUMN_BORDER_PX < x < w - self.COLUMN_BORDER_PX
        ]

    def _count_columns(self, img: np.ndarray) -> int:
        """Count visible data columns in the image body.

        Uses a stricter threshold (0.5) than snapping so faint text
        strokes do not get counted as dividers.
        """
        return len(self._vertical_lines_in(self._body_slice(img),
                                           threshold_frac=0.5)) + 1

    def _detect_first_row_columns(
        self,
        img: np.ndarray,
        rows: Sequence[tuple[int, int]],
    ) -> list[Column] | None:
        """Return a column layout for the first row when it differs.

        Some images (e.g. ``I100376``) start with a single 3-col SMILES-
        style row above the dense activity table. ``_detect_columns``
        only inspects the body below the header, so it picks the dense
        layout and the leading SMILES row would be dropped without this
        override. Returns ``None`` to let the caller fall back to the
        main columns. Thin slices are treated as header bands and
        ignored.
        """
        if not rows:
            return None
        y1, y2 = rows[0]
        if y2 - y1 < self.FIRST_ROW_MIN_PX:
            return None
        xs = self._vertical_lines_in(img[y1:y2, :])
        layout = detect_layout_from_columns(len(xs) + 1) if xs else None
        return default_columns(img.shape[1], layout) if layout else None

    def _detect_vertical_borders(self, img: np.ndarray) -> list[int]:
        """Return x positions of full-height vertical rules in the body."""
        return self._vertical_lines_in(self._body_slice(img))

    def _snap_to_borders(
        self,
        img: np.ndarray,
        columns: list[Column],
    ) -> list[Column]:
        """Adjust each column's x1/x2 to the nearest detected vertical border.

        Falls back to the original boundaries when no border is within
        ``BORDER_SNAP_PX`` so a bad detection cannot move the crop wildly.
        """
        borders = self._detect_vertical_borders(img)
        if not borders:
            return columns
        snapped: list[Column] = []
        for col in columns:
            best_left = col.x1
            best_left_dist = self.BORDER_SNAP_PX + 1
            best_right = col.x2
            best_right_dist = self.BORDER_SNAP_PX + 1
            for b in borders:
                dl = abs(b - col.x1)
                if dl < best_left_dist:
                    best_left_dist = dl
                    best_left = b
                dr = abs(b - col.x2)
                if dr < best_right_dist:
                    best_right_dist = dr
                    best_right = b
            snapped.append(Column(col.name, best_left, best_right, col.is_image))
        return snapped

    # ---- row boundary detection ----

    def _detect_row_boundaries(self, img: np.ndarray) -> list[tuple[int, int]]:
        """Horizontal-line morphology. First inner line = bottom of header."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.4), 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        ys = np.where(np.sum(lines, axis=1) > 0)[0]
        clusters = self._cluster(ys, gap=self.ROW_LINE_GAP_PX)
        if not clusters:
            return []

        # Border band scales with image height: a fixed 40-px band excludes
        # every line on short images (e.g. I100382 is 70 px tall and would
        # become empty). 8 px on each side clears page borders on any
        # reasonable size while keeping the bulk of the body.
        band_lo = self.ROW_BORDER_PX
        band_hi = max(band_lo + 1, h - self.ROW_BORDER_PX)
        inner = [y for y in clusters if band_lo < y < band_hi]

        # Single-row continuation pages / header-only images fall through
        # here; treat the whole image as one slice instead of dropping it.
        if len(inner) < 2:
            inner = []

        # ``inner[0]`` is the bottom edge of the header band for normal
        # single-table pages; data rows start there. We DO include ``0`` as
        # a divider so a page with a continuation row above the main table
        # (e.g. I100376's cmpd 2173 above the dense activity table) still
        # emits that row as a slice. For ordinary tables the resulting
        # first slice is the header band; OCR fails to find a 4-digit
        # cmpd_id and the row is dropped downstream.
        dividers = sorted({0} | set(inner) | {h})
        slices: list[tuple[int, int]] = []
        for i in range(len(dividers) - 1):
            y1 = dividers[i] + (1 if i > 0 else 0)
            y2 = dividers[i + 1] - (1 if i + 1 < len(dividers) - 1 else 0)
            if y2 - y1 >= self.MIN_ROW_PX:
                slices.append((y1, y2))
        return slices

    @staticmethod
    def _cluster(ys: Iterable[int], *, gap: int) -> list[int]:
        ys = [int(y) for y in ys]
        if not ys:
            return []
        clusters: list[list[int]] = [[ys[0]]]
        for y in ys[1:]:
            if y - clusters[-1][-1] <= gap:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        return [int(np.mean(c)) for c in clusters]

    # ---- cell-level extraction ----

    def _extract_rows(
        self,
        img: np.ndarray,
        rows: Sequence[tuple[int, int]],
        columns: Sequence[Column],
        image_path: Path,
        cells_dir: Path | None,
    ) -> tuple[list[CompoundRecord], list[AssayRecord]]:
        compounds: dict[str, CompoundRecord] = {}
        assays: dict[str, AssayRecord] = {}
        row_smiles_stats: dict[str, OcrStats] = {}

        cid_col = next((c for c in columns if c.name == "cmpd_id"), None)
        if cid_col is None or cid_col.x2 <= cid_col.x1:
            return [], []

        # Some images open with a single SMILES-style row before the main
        # table (see ``_detect_first_row_columns``). The override only makes
        # sense when the main body is a dense activity table (6 cols); for
        # SMILES or 7-col activity tables the first slice is the header
        # band, and a wrongly-triggered override shrinks the structure
        # column (I100371 was getting 892-px crops with the leftmost F
        # atom clipped because the override fired on a merged
        # header+row slice).
        first_row_columns = (
            self._detect_first_row_columns(img, rows)
            if len(columns) == 6
            else None
        )

        for slice_idx, (y1, y2) in enumerate(rows):
            slice_columns = (
                first_row_columns
                if slice_idx == 0 and first_row_columns is not None
                else columns
            )
            cid_col_local = next(
                (c for c in slice_columns if c.name == "cmpd_id"), cid_col
            )
            cmpd_id = self._read_cmpd_id(
                img[y1:y2, cid_col_local.x1:cid_col_local.x2]
            )
            if cmpd_id is None:
                continue

            row_values: dict[str, str | None] = {}
            structure_path: str | None = None

            for col in slice_columns:
                if col.name == "cmpd_id" or col.x2 <= col.x1:
                    continue
                cell = img[y1:y2, col.x1:col.x2]
                if col.is_image:
                    if cells_dir is not None and structure_path is None:
                        structure_path = self._save_structure_crop(
                            cell, cmpd_id, cells_dir, image_path.stem
                        )
                    continue
                if col.name == "smiles" and self.rec_top_k > 1:
                    text, ocr_stats = self._ocr_smiles_with_stats(cell)
                    row_values[col.name] = text
                    row_smiles_stats[cmpd_id] = ocr_stats
                else:
                    row_values[col.name] = self._ocr_cell(cell, col.name)

            has_smiles = "smiles" in row_values and row_values["smiles"]
            # Build a CompoundRecord whenever we have a structure crop
            # OR a SMILES string. SMILES validation still runs when
            # present; structure-only rows get is_valid=False with
            # SMILES fields left null.
            if structure_path is not None or has_smiles:
                fixed, chem = None, None
                if has_smiles:
                    # Lazy import: cpd.chem depends on RDKit, which we do
                    # not require just for schema import or light tests.
                    from cpd.chem import auto_repair_smiles, validate_and_enrich
                    fixed = auto_repair_smiles(row_values["smiles"] or "")
                    chem = validate_and_enrich(fixed) if fixed else None
                stats = row_smiles_stats.get(cmpd_id) or OcrStats()
                compounds[cmpd_id] = CompoundRecord(
                    cmpd_id=cmpd_id,
                    smiles=fixed or None,
                    canonical_smiles=chem.canonical_smiles if chem else None,
                    molecular_weight=chem.molecular_weight if chem else None,
                    heavy_atom_count=chem.heavy_atom_count if chem else None,
                    is_valid=chem is not None,
                    structure_image=structure_path,
                    source_image=image_path.name,
                    ocr_avg_entropy=stats.avg_entropy,
                    ocr_min_top1=stats.min_top1,
                    ocr_low_conf_count=stats.low_conf_count,
                )

            if {"kd_nm", "rt_min", "ms_mz"} & row_values.keys():
                assays[cmpd_id] = AssayRecord(
                    cmpd_id=cmpd_id,
                    kd_nm=_as_float(row_values.get("kd_nm")),
                    rt_min=_as_float(row_values.get("rt_min")),
                    ms_mz=_as_float(row_values.get("ms_mz")),
                    lcms_method=row_values.get("lcms_method"),
                    ms_polarity=row_values.get("ms_polarity"),
                    source_image=image_path.name,
                )

        return list(compounds.values()), list(assays.values())

    def _read_cmpd_id(self, cell: np.ndarray) -> str | None:
        """Find the 4-digit compound id in the leftmost cell."""
        if cell.size == 0:
            return None
        # cmpd_id cells are tall narrow columns where the 4-digit text
        # sits in the middle surrounded by whitespace and the cell's own
        # border lines. We try the Chinese (default) engine first because
        # the EN recogniser (PP-OCRv4 mobile) routinely misreads dense-
        # table cmpd_ids as 2-character garbage ("1037" -> "?, 4, C")
        # while the Chinese recogniser returns the digit string cleanly.
        # Falling back to the EN engine only matters when the Chinese
        # detector finds nothing on a tight crop.
        for engine in (self.engine, self.rec_engine):
            result = engine(self._upscale(cell))
            if not result:
                continue
            for text in self._txts(result):
                for token in text.split():
                    if re.fullmatch(r"\d{4}", token.strip()):
                        return token.strip()
        return None

    def _ocr_cell(self, cell: np.ndarray, field_name: str) -> str | None:
        """OCR one cell. SMILES lines concatenate; other cells join with spaces.

        When the recogniser was built with ``rec_top_k > 1`` and the
        cell is a SMILES column, we run the recogniser once and then
        feed every top-k swap through RDKit; the first canonical
        SMILES that parses wins. This catches bracket/paren confusion
        (``C)[C@H]`` instead of ``(C)[C@H]``) and 0/O swaps that the
        argmax alone cannot disambiguate.
        """
        if cell.size == 0:
            return None
        cell = self._trim_to_content(cell)
        if cell.size == 0:
            return None
        # Activity-table cells (KD/RT/m/z/method/polarity) are short and
        # dense; the English mobile recogniser over-segments them into
        # one box per character (``1.554`` -> ``['1', '.', '5', '5', '4']``)
        # while the Chinese default recogniser reads them cleanly once
        # we strip the table borders and re-upscale. SMILES is kept on
        # the EN recogniser since CH routinely returns empty detections
        # for those long strings.
        result = self.rec_engine(self._upscale(cell))
        if field_name != "smiles" and self._looks_over_segmented(result):
            for eng in (self.engine, self.rec_engine):
                stripped = self._strip_borders(cell)
                if stripped.size:
                    upscaled = self._upscale(stripped)
                else:
                    upscaled = self._upscale(cell)
                if max(upscaled.shape[:2]) < 100:
                    upscaled = cv2.resize(
                        upscaled,
                        (upscaled.shape[1] * 2, upscaled.shape[0] * 2),
                        interpolation=cv2.INTER_CUBIC,
                    )
                candidate = eng(upscaled)
                if candidate and (self._txts(candidate) or []):
                    result = candidate
                    break
        if not result:
            return None
        boxes = self._boxes(result)
        txts = self._txts(result)
        if boxes is not None:
            items = sorted(
                zip(boxes, txts),
                key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2),
            )
        else:
            items = list(zip([None] * len(txts), txts))
        parts = [t.strip() for _, t in items if t and t.strip()]
        if not parts:
            return None
        joiner = "" if field_name == "smiles" else " "
        text = joiner.join(parts)

        if field_name == "smiles" and self.rec_top_k > 1:
            return self._fix_smiles_with_top_k(result, parts)
        return text

    def _save_structure_crop(
        self,
        cell: np.ndarray,
        cmpd_id: str,
        cells_dir: Path,
        src_stem: str,
    ) -> str | None:
        """Write the structure cell as a PNG; return path relative to cells_dir.parent.

        Two guard rails:

        1. **Content check.** Binarise the cell and reject crops whose
           dark-pixel ratio is too low (<2%, effectively empty) or
           suspiciously uniform (>55%) -- the signature of a text cell
           (RT/MS/Method) wrongly labelled as the structure column
           because the layout detector picked the wrong table type.
        2. **Unique filename.** ``src_stem`` is the source image filename
           without extension; including it in the output name prevents
           two images containing the same ``cmpd_id`` from clobbering
           each other's crop on disk.
        """
        if cell.size == 0:
            return None
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = float(binary.mean()) / 255.0
        if dark_ratio < 0.02 or dark_ratio > 0.55:
            return None

        cells_dir.mkdir(parents=True, exist_ok=True)
        # 12 px white margin so molecules drawn close to the column edge
        # (e.g. the F group on the left of cmpd 2143) still get a white
        # margin in the cropped file instead of touching the boundary.
        padded = cv2.copyMakeBorder(
            cell, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        out = cells_dir / f"{src_stem}_{cmpd_id}_struct.png"
        cv2.imwrite(str(out), padded)
        return str(out.relative_to(cells_dir.parent))

    # ---- RapidOCR result adapters (new SDK vs legacy list-of-tuples) ----

    @staticmethod
    def _boxes(result):
        """Return per-region boxes from a RapidOCR result.

        New SDK returns ``RapidOCROutput`` with a ``.boxes`` ndarray;
        legacy SDK returns a list of (box, txt, score) tuples. Both are
        iterable so the same loop works. ``None`` is returned only when
        the recogniser-only path produced an empty ``TextRecOutput``
        (no boxes available); callers should skip spatial sorting in
        that case and use the textual emission order instead.
        """
        if hasattr(result, "boxes") and result.boxes is not None:
            return result.boxes
        if hasattr(result, "txts"):
            try:
                return [r[0] for r in result]
            except TypeError:
                return None
        return [r[0] for r in result]

    @staticmethod
    def _txts(result):
        return result.txts if hasattr(result, "txts") else [r[1] for r in result]

    @staticmethod
    def _upscale(img: np.ndarray) -> np.ndarray:
        """Upscale small crops 2x for better OCR recall."""
        h, w = img.shape[:2]
        if max(h, w) < 50:
            return img
        return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _strip_borders(img: np.ndarray, pad: int = 20) -> np.ndarray:
        """Drop the thick vertical table-border strokes on the left/right of a cell.

        Activity-table cells are wrapped in a heavy black gridline. The
        Chinese default recogniser's text detector refuses to fire when
        those long vertical strokes dominate the image; trimming ~20 px
        off each side keeps the digit/letter content intact and lets
        detection succeed.
        """
        if img.ndim < 2 or img.shape[1] <= 2 * pad:
            return img
        return img[:, pad:img.shape[1] - pad]

    def _looks_over_segmented(self, result, min_boxes: int = 3) -> bool:
        """True when the recogniser returned one short box per character.

        The EN mobile recogniser splits short numeric cells like ``1.554``
        into ``['1', '.', '5', '5', '4']``. That sort of output is
        unusable downstream; we detect it by counting boxes whose text is
        a single printable character and ask the caller to retry with the
        CH recogniser on a stripped cell.
        """
        txts = self._txts(result) or []
        if len(txts) < min_boxes:
            return False
        single = sum(1 for t in txts if t and len(t.strip()) <= 1)
        return single >= min_boxes

    @staticmethod
    def _trim_to_content(img: np.ndarray, pad: int = 4) -> np.ndarray:
        """Crop a sparse cell to the bounding box of its dark content.

        Even with the full pipeline (det+rec), a tall sparse cell like
        a SMILES column with whitespace at the top makes the detector
        less reliable and the rec output noisy. Tightening to the
        bounding box first improves both steps.
        """
        if img.size == 0:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None or len(coords) == 0:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        h_full, w_full = img.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_full, x + w + pad)
        y1 = min(h_full, y + h + pad)
        cropped = img[y0:y1, x0:x1]
        return cropped if cropped.size else img


    # ---- top-k SMILES validation ----

    def _fix_smiles_with_top_k(
        self, result, parts: list[str]
    ) -> str | None:
        """Pick the first RDKit-valid SMILES among top-k candidates.

        ``result`` is the full RapidOCROutput; its ``top_k_lines``
        carries per-character alternatives. ``parts`` is the
        already-sorted text rows; we rebuild the full SMILES
        string and then try every per-position top-k swap.
        """
        top_k_lines = getattr(result, "top_k_lines", None)
        if not top_k_lines:
            return "".join(parts) or None
        from cpd.chem import validate_and_enrich

        full_text = "".join(parts)
        if not full_text:
            return None
        char_alts: list[list[tuple[str, float]]] = []
        for row_text, row_alts in zip(parts, top_k_lines):
            # The recogniser may emit None for an empty line;
            # fall back to a single-argmax list so the greedy swap
            # loop below does not crash.
            row_alts = row_alts or []
            for c_i, ch in enumerate(row_text):
                if c_i < len(row_alts):
                    char_alts.append(row_alts[c_i])
                else:
                    char_alts.append([(ch, 1.0)])

        seen: set[str] = set()

        def try_text(t: str) -> str | None:
            if t in seen:
                return None
            seen.add(t)
            chem = validate_and_enrich(t)
            if chem is not None:
                return chem.canonical_smiles or t
            return None

        hit = try_text(full_text)
        if hit is not None:
            return hit

        # Greedy single-swap search. For each char position with more
        # than one alternative, try the next-best alt in place.
        # O(top_k * len(smiles)) worst case; in practice most
        # positions have only 1 candidate so the loop ends after
        # the first OCR error is corrected.
        chars = list(full_text)
        for pos, alts in enumerate(char_alts):
            if len(alts) <= 1:
                continue
            original = chars[pos]
            for alt_ch, _p in alts[1:]:
                if alt_ch == original:
                    continue
                chars[pos] = alt_ch
                cand_text = "".join(chars)
                hit = try_text(cand_text)
                if hit is not None:
                    return hit
                chars[pos] = original

        return full_text



    def _compute_ocr_stats(self, result, parts: list[str]) -> OcrStats:
        """Aggregate per-char top-k entropy across a SMILES cell.

        Returns OcrStats with all None fields when ``result`` has no
        top-k data (e.g. the recogniser was built with
        ``rec_top_k=0`` or the SDK did not attach ``top_k_lines``).
        """
        top_k_lines = getattr(result, "top_k_lines", None)
        if not top_k_lines:
            return OcrStats()

        entropies: list[float] = []
        top1s: list[float] = []
        for row_text, row_alts in zip(parts, top_k_lines):
            row_alts = row_alts or []
            for c_i, ch in enumerate(row_text):
                if c_i >= len(row_alts):
                    # No alternatives -- trust the argmax fully.
                    top1s.append(1.0)
                    entropies.append(0.0)
                    continue
                alts = row_alts[c_i]
                probs = [float(p) for _, p in alts if p is not None]
                if not probs:
                    top1s.append(1.0)
                    entropies.append(0.0)
                    continue
                top1s.append(probs[0])
                h = 0.0
                for p in probs:
                    if p > 0.0:
                        h -= p * (p and __import__("math").log2(p))
                entropies.append(h)

        if not entropies:
            return OcrStats()
        low = sum(1 for p in top1s if p < LOW_CONF_PROB_THRESHOLD)
        return OcrStats(
            avg_entropy=round(sum(entropies) / len(entropies), 4),
            min_top1=round(min(top1s), 4),
            low_conf_count=low,
        )

    def _ocr_smiles_with_stats(
        self, cell: np.ndarray
    ) -> tuple[str | None, OcrStats]:
        """OCR a SMILES cell, returning the (text, OcrStats) pair.

        The text is the same one ``_ocr_cell`` would have produced
        (including the top-k swap fix when ``rec_top_k > 1``); the
        stats are computed from the same top-k distribution. Returns
        ``(None, OcrStats())`` when the cell is empty or the OCR
        engine yields nothing.
        """
        if cell.size == 0:
            return None, OcrStats()
        cell = self._trim_to_content(cell)
        if cell.size == 0:
            return None, OcrStats()
        result = self.rec_engine(self._upscale(cell))
        if not result:
            return None, OcrStats()
        boxes = self._boxes(result)
        txts = self._txts(result)
        if boxes is not None:
            items = sorted(
                zip(boxes, txts),
                key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2),
            )
        else:
            items = list(zip([None] * len(txts), txts))
        parts = [t.strip() for _, t in items if t and t.strip()]
        if not parts:
            return None, OcrStats()
        text = "".join(parts)  # SMILES lines concatenate (no space join)
        stats = self._compute_ocr_stats(result, parts)
        if self.rec_top_k > 1:
            fixed = self._fix_smiles_with_top_k(result, parts)
            if fixed:
                text = fixed
        return text, stats

# ---------- helpers ----------


def _as_float(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


__all__ = [
    "AssayRecord",
    "Column",
    "CompoundRecord",
    "RapidOcrTableExtractor",
    "default_columns",
    "detect_layout",
    "detect_layout_from_columns",
]



