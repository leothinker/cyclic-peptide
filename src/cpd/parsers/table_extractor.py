"""Unified table extractor producing compounds + assays from patent JPGs.

Three table layouts appear in WO2025162428's golden tables, and the
extractor handles them all with one code path:

  * 3-col SMILES table           (Cmpd # | Structure | SMILES)
  * 7-col activity table         (Cmpd # | Structure | RT | MS | Method | Polarity | KD)
  * 6-col dense activity table   (Compound # | RT | MS | Method | Polarity | KD)

The pipeline for every image is:

  1. OCR the top header band and map each detected header cell to a
     canonical field name via :mod:`cpd.parsers.headers` (so ``Compound #``
     and ``Cmpd #`` collapse to the same field).
  2. Detect horizontal row boundaries with OpenCV morphology.
  3. For each row, OCR every text cell; for the ``structure`` column,
     crop the cell to a PNG under ``cells_dir``.

Output is **two flat lists**, joined on ``cmpd_id`` by the caller:

  * :class:`CompoundRecord` carries ``cmpd_id``, ``smiles``, RDKit
    canonical form, descriptors, and the cropped structure-image path.
  * :class:`AssayRecord`  carries ``cmpd_id``, ``kd_nm``, ``rt_min``,
    ``ms_mz``, ``lcms_method``, ``ms_polarity``.

Continuation pages without a header (``I100382``) reuse the previous
image's column layout (passed via ``prev_columns``) so we never lose a
row. Header-only images (``I100287``) emit zero records and are filtered
out by the row count.
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


# Default 7-col activity layout (used when no header text is detected).
# Values are fractions of image width and get scaled at construction time.
_DEFAULT_ACTIVITY_FRACS: tuple[tuple[str, float, float, bool], ...] = (
    ("cmpd_id", 0.00, 0.13, False),
    ("structure", 0.13, 0.58, True),
    ("rt_min", 0.58, 0.72, False),
    ("ms_mz", 0.72, 0.82, False),
    ("lcms_method", 0.82, 0.89, False),
    ("ms_polarity", 0.89, 0.95, False),
    ("kd_nm", 0.95, 1.00, False),
)


def _default_activity_columns(width: int) -> list[Column]:
    return [
        Column(name, int(x1 * width), int(x2 * width), is_image)
        for name, x1, x2, is_image in _DEFAULT_ACTIVITY_FRACS
    ]


# ---------- main extractor ----------


class RapidOcrTableExtractor:
    """OpenCV grid detection + RapidOCR per-cell extraction."""

    name: str = "rapidocr"

    # Header band covers the top 15% of the image (table-header row).
    HEADER_FRAC: float = 0.15

    # Two horizontal lines within ~10 px belong to the same border.
    ROW_LINE_GAP_PX: int = 10

    # A row needs at least this many vertical pixels to be kept.
    MIN_ROW_PX: int = 30

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
        """Extract compounds + assays from one table image.

        ``cells_dir`` is the directory where the structure column is
        cropped to per-row PNGs. ``prev_columns`` is reused for
        continuation pages whose header band is empty.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return [], [], prev_columns or []

        columns = self._detect_columns(img)
        if not columns:
            columns = prev_columns or _default_activity_columns(img.shape[1])

        rows = self._detect_row_boundaries(img)
        compounds, assays = self._extract_rows(
            img, rows, columns, image_path, cells_dir
        )
        return compounds, assays, columns

    def extract_smiles(
        self,
        image_path: Path,
        cells_dir: Path | None = None,
    ) -> list[CompoundRecord]:
        """Back-compat helper: return only the CompoundRecords of one image."""
        compounds, _, _ = self.extract_records(image_path, cells_dir)
        return compounds

    def extract_activity(
        self,
        image_path: Path,
        cells_dir: Path | None = None,
    ) -> list[AssayRecord]:
        """Back-compat helper: return only the AssayRecords of one image."""
        _, assays, _ = self.extract_records(image_path, cells_dir)
        return assays

    # ---- column detection ----

    def _detect_columns(self, img: np.ndarray) -> list[Column]:
        """OCR the header band, then map each cell to a canonical field."""
        h, w = img.shape[:2]
        full_width = w
        crop = self._crop(img, 0, int(h * self.HEADER_FRAC))
        upscaled = self._scale(crop)
        result = self.engine(upscaled)
        if not result:
            return []

        items = sorted(
            zip(self._boxes(result), self._txts(result)),
            key=lambda bt: (bt[0][0][0] + bt[0][1][0]) / 2,
        )
        scale_back = self._scale_factor(upscaled.shape[1], full_width)

        # Group cells that map to the same canonical field (e.g. wrapped text)
        # into one column whose x-range spans the union of all matched cells.
        grouped: dict[str, list[tuple[int, int]]] = {}
        for box, text in items:
            field = canonical_field(text)
            if field is None:
                continue
            x1 = int(box[0][0] * scale_back)
            x2 = int(box[1][0] * scale_back)
            grouped.setdefault(field, []).append((min(x1, x2), max(x1, x2)))

        columns = [
            Column(name, lo, hi, name == "structure")
            for name, spans in grouped.items()
            for lo, hi in [_union(spans)]
        ]
        columns.sort(key=lambda c: c.x1)

        if columns and columns[0].name != "cmpd_id":
            columns = self._insert_cmpd_id_column(columns, w)

        return self._fill_gaps(columns, w)

    @staticmethod
    def _scale_factor(cropped_width: int, full_width: int) -> float:
        return full_width / max(cropped_width, 1)

    def _scale(self, img: np.ndarray) -> np.ndarray:
        """Upscale a crop 2x for better OCR recall."""
        h, w = img.shape[:2]
        if max(h, w) < 50:
            return img
        return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _crop(img: np.ndarray, y1: int, y2: int) -> np.ndarray:
        h = img.shape[0]
        return img[max(0, y1) : min(h, y2), :]

    @staticmethod
    def _insert_cmpd_id_column(columns: list[Column], w: int) -> list[Column]:
        """If header OCR missed ``cmpd_id``, prepend a 12% wide column."""
        if not columns:
            return _default_activity_columns(w)
        x2 = int(columns[0].x1)
        head = Column("cmpd_id", 0, max(x2, int(w * 0.12)), False)
        return [head, *columns]

    @staticmethod
    def _fill_gaps(columns: list[Column], w: int) -> list[Column]:
        """Stretch each column to abut its neighbour so cell crops don't gap."""
        if not columns:
            return []
        out: list[Column] = []
        prev_x2 = 0
        for c in columns:
            x1 = max(c.x1, prev_x2)
            if x1 < c.x2:
                out.append(Column(c.name, x1, c.x2, c.is_image))
            prev_x2 = c.x2
        return out

    # ---- row boundary detection ----

    def _detect_row_boundaries(self, img: np.ndarray) -> list[tuple[int, int]]:
        """Use horizontal-line morphology to split the image into rows."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.4), 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        ys = np.where(np.sum(lines, axis=1) > 0)[0]
        clusters = self._cluster(ys, gap=self.ROW_LINE_GAP_PX)
        if not clusters:
            return []

        # Drop the header band: keep only lines that sit below it.
        header_h = self._first_inner(clusters, h)
        inner = [y for y in clusters if header_h < y < h - 30]
        dividers = sorted({0, *inner, h})

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

    @staticmethod
    def _first_inner(clusters: Sequence[int], h: int) -> int:
        """Y of the first horizontal line strictly inside the table body."""
        for y in clusters:
            if 40 < y < h - 40:
                return y
        return int(h * 0.15)

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
        if cid_col is None:
            return [], []

        for y1, y2 in rows:
            cid_cell = img[y1:y2, cid_col.x1:cid_col.x2]
            cmpd_id = self._read_cmpd_id(cid_cell)
            if cmpd_id is None:
                continue

            row_values: dict[str, str | None] = {}
            structure_path: str | None = None

            for col in columns:
                if col.name == "cmpd_id":
                    continue
                cell = img[y1:y2, col.x1:col.x2]
                if col.is_image:
                    if cells_dir is not None and structure_path is None:
                        structure_path = self._save_structure_crop(
                            cell, cmpd_id, cells_dir
                        )
                    continue
                row_values[col.name] = self._ocr_cell(cell, col.name)

            if "smiles" in row_values and row_values["smiles"]:
                # Lazy import: cpd.chem depends on RDKit, which we do not
                # require just for schema import or lightweight tests.
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

            activity_keys = {"kd_nm", "rt_min", "ms_mz"}
            if activity_keys & row_values.keys():
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
        result = self.engine(self._scale(cell))
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
        result = self.engine(self._scale(cell))
        if not result:
            return None
        items = sorted(
            zip(self._boxes(result), self._txts(result)),
            key=lambda bt: (bt[0][0][1] + bt[0][2][1]) / 2,
        )
        parts = [t.strip() for _, t in items if t and t.strip()]
        if not parts:
            return None
        joiner = "" if field_name == "smiles" else " "
        return joiner.join(parts)

    def _save_structure_crop(
        self, cell: np.ndarray, cmpd_id: str, cells_dir: Path
    ) -> str | None:
        """Write the structure cell as a PNG; return its path relative to cells_dir.parent."""
        if cell.size == 0:
            return None
        cells_dir.mkdir(parents=True, exist_ok=True)
        padded = cv2.copyMakeBorder(
            cell, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        out = cells_dir / f"{cmpd_id}_struct.png"
        cv2.imwrite(str(out), padded)
        return str(out.relative_to(cells_dir.parent))

    # ---- RapidOCR result adapters (new SDK vs legacy list-of-tuples) ----

    @staticmethod
    def _boxes(result):
        return result.boxes if hasattr(result, "boxes") else [r[0] for r in result]

    @staticmethod
    def _txts(result):
        return result.txts if hasattr(result, "txts") else [r[1] for r in result]


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


def _union(spans: Sequence[tuple[int, int]]) -> tuple[int, int]:
    return min(s[0] for s in spans), max(s[1] for s in spans)


__all__ = [
    "AssayRecord",
    "Column",
    "CompoundRecord",
    "RapidOcrTableExtractor",
]
