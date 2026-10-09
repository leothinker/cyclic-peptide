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

    def __init__(self) -> None:
        from rapidocr import RapidOCR  # local import keeps the module lightweight
        self.engine = RapidOCR()

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
                compounds[cmpd_id] = CompoundRecord(
                    cmpd_id=cmpd_id,
                    smiles=fixed or None,
                    canonical_smiles=chem.canonical_smiles if chem else None,
                    molecular_weight=chem.molecular_weight if chem else None,
                    heavy_atom_count=chem.heavy_atom_count if chem else None,
                    is_valid=chem is not None,
                    structure_image=structure_path,
                    source_image=image_path.name,
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
        result = self.engine(self._upscale(cell))
        if not result:
            return None
        for text in self._txts(result):
            for token in text.split():
                if re.fullmatch(r"\d{4}", token.strip()):
                    return token.strip()
        return None

    def _ocr_cell(self, cell: np.ndarray, field_name: str) -> str | None:
        """OCR one cell. SMILES lines concatenate; other cells join with spaces."""
        if cell.size == 0:
            return None
        result = self.engine(self._upscale(cell))
        if not result:
            return None
        items = sorted(
            zip(self._boxes(result), self._txts(result)),
            key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2),
        )
        parts = [t.strip() for _, t in items if t and t.strip()]
        if not parts:
            return None
        joiner = "" if field_name == "smiles" else " "
        return joiner.join(parts)

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
        return result.boxes if hasattr(result, "boxes") else [r[0] for r in result]

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
