"""Classify FullText JPGs into ``activity`` / ``smiles`` / ``structure`` / ``other``.

Why this exists
---------------
WO2025162428 ships 426 figures in its FullText/ directory: 9 Markush
formulas, 85 inline ``<img>`` captioned to ``Compound N`` (each paired with
a structure drawing in synthesis prose), and ~332 unknowns (cover artwork,
reaction schemes, etc.).

The user's breakthrough observation: a small slice of these images are
*tabular* pages, each containing 4 rows x 7 columns of pre-computed data:

  - activity table: ``Cmpd # | Structure | LCMS RT | MS (m/z) | LCMS
    Method | MS Polarity | G12V GDP KD nM``
  - smiles table: ``Cmpd # | Structure | SMILES``

These two table types already carry the canonical SMILES + KD we need.
No OCSR or VLM is required -- the only work left is finding which images
they are and joining by ``Cmpd #``.

This module does the *finding* part, with three independent signals:

  1. **Size + aspect + width** of each raster (no OCR needed):
       * ``width >= 1940 px`` AND ``>= 500 KB`` -> SMILES table
         (the SMILES column itself is wide enough to bump the canvas)
       * ``1880 <= width < 1940 px`` AND ``150 <= size < 500 KB`` -> activity
       * small + sparse -> likely structure figure
       * everything else -> ``other`` (text page, cover artwork, ...)
  2. **Contiguous-range grouping**: tables live in numbered blocks
     (e.g. ``I100280..I100295`` are all activity tables because each table
     page holds 4 rows and the patent uses sequential numbering for them).
     We collapse each block into a :class:`TableBlock`.
  3. **Optional OCR/VLM hint** via :class:`TableKeywordDetector`. When the
     local environment has pytesseract, easyocr, or a callable VLM hook, we
     run a cheap header-region OCR and look for keywords ``G12V``, ``KD``,
     ``SMILES``. When no detector is wired, this signal is skipped (the
     size+range heuristic alone is good enough for this patent).

Outputs
-------
* :func:`scan_directory` -> list[:class:`TableImage`]  (one per JPG)
* :func:`find_tables`    -> list[:class:`TableBlock`]   (contiguous groups)
* :func:`find_by_type`   -> dict ``{table_type -> [TableImage, ...]}``
* :func:`write_findings_json` / :func:`load_findings_json`

Designed to degrade gracefully
------------------------------
The whole module is stdlib + Pillow. No external OCR / VLM / ML deps. The
:class:`TableKeywordDetector` is the only seam; the default implementation
is a no-op, so the pipeline still works on locked-down machines.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

from PIL import Image

# Keywords that disambiguate activity vs smiles tables.
_ACTIVITY_KEYWORDS = ("g12v", "kd", "k d", "gdp", "lcms", "rt (min)", "polarity")
_SMILES_KEYWORDS = ("smiles",)

# Filename pattern: PCTCN2025075389-ftappb-I100001.jpg
_NUM_RE = re.compile(r"I(\d+)\.jpg$", re.IGNORECASE)


# ---------- dataclasses ----------


class TableType:
    """String-typed enum-like (kept as str for easy JSON serialisation)."""

    ACTIVITY = "activity"
    SMILES = "smiles"
    STRUCTURE = "structure"
    OTHER = "other"
    UNKNOWN = "unknown"

    ALL = (ACTIVITY, SMILES, STRUCTURE, OTHER, UNKNOWN)


@dataclass(frozen=True)
class TableImage:
    """One FullText JPG and its classification."""

    image_path: Path
    filename: str
    numeric_id: int
    width: int
    height: int
    size_bytes: int
    aspect: float                      # width / height
    table_type: str = TableType.UNKNOWN
    confidence: float = 0.0            # 0..1
    detected_keywords: list[str] = field(default_factory=list)
    is_table_candidate: bool = False   # pre-clustering flag

    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TableImage":
        return cls(
            image_path=Path(d["image_path"]),
            filename=d["filename"],
            numeric_id=int(d["numeric_id"]),
            width=int(d["width"]),
            height=int(d["height"]),
            size_bytes=int(d["size_bytes"]),
            aspect=float(d["aspect"]),
            table_type=str(d.get("table_type", TableType.UNKNOWN)),
            confidence=float(d.get("confidence", 0.0)),
            detected_keywords=list(d.get("detected_keywords") or []),
            is_table_candidate=bool(d.get("is_table_candidate", False)),
        )


@dataclass(frozen=True)
class TableBlock:
    """A contiguous range of TableImages likely belonging to the same table series."""

    table_type: str
    start_id: int
    end_id: int
    images: tuple[TableImage, ...]
    dominant_keywords: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        """Best guess: 4 compounds per page for this patent's table layout."""
        return len(self.images) * 4

    def to_dict(self) -> dict:
        return {
            "table_type": self.table_type,
            "start_id": self.start_id,
            "end_id": self.end_id,
            "n_pages": len(self.images),
            "dominant_keywords": list(self.dominant_keywords),
            "filenames": [img.filename for img in self.images],
        }


# ---------- keyword detector (pluggable) ----------


class TableKeywordDetector(Protocol):
    """Pluggable keyword detection over a PIL image.

    Implementations may OCR locally (pytesseract/easyocr) or call a remote
    VLM. Return an iterable of keywords (lowercased) found in the image.
    """

    def detect(self, image: Image.Image) -> Iterable[str]: ...


class NullKeywordDetector:
    """Default detector: returns nothing.

    Keeps the pipeline runnable on machines without OCR installed. The
    size+range heuristic still classifies most pages correctly for this
    patent, so missing OCR is not a blocker.
    """

    def detect(self, image: Image.Image) -> Iterable[str]:
        return ()


class HeaderKeywordDetector:
    """Cheap header-only OCR detector.

    Crops the top 18% of the image (the table-header band), runs pytesseract
    if available, and looks for our keywords. If pytesseract isn't installed
    the detector falls back to :class:`NullKeywordDetector` behaviour.
    """

    HEADER_FRAC = 0.18

    def __init__(self, *, ocr_callable: Callable[[Image.Image], str] | None = None) -> None:
        self._ocr = ocr_callable

    def detect(self, image: Image.Image) -> Iterable[str]:
        if self._ocr is None:
            try:
                import pytesseract  # type: ignore  # noqa: F401
            except ImportError:
                return ()
            self._ocr = lambda im: pytesseract.image_to_string(im)

        w, h = image.size
        header = image.crop((0, 0, w, max(1, int(h * self.HEADER_FRAC))))
        try:
            text = self._ocr(header).lower()
        except Exception:
            return ()

        hits: list[str] = []
        if any(k in text for k in _ACTIVITY_KEYWORDS):
            hits.append("activity")
        if any(k in text for k in _SMILES_KEYWORDS):
            hits.append("smiles")
        return hits


# ---------- public API ----------


# Size+aspect+width thresholds tuned on WO2025162428's FullText layout.
STRUCTURE_MAX_BYTES = 80_000     # pure structure figures are sparse
TABLE_MIN_ASPECT = 0.65          # letter/A4-ish landscape
TABLE_MAX_ASPECT = 1.40          # not too wide -> not a panoramic banner

# SMILES vs activity discrimination (WO2025162428 specific):
#   activity pages render at width 1909-1924 px (no SMILES column) and
#   weigh 250-400 KB.
#   SMILES pages render at width 1945 px (the SMILES column itself is wide
#   enough to bump the canvas) and weigh 550-800 KB.
# Text-heavy prose pages with embedded figures can be >=500 KB but stay
# below 1900 px wide, so they don't trip the SMILES rule.
SMILES_MIN_WIDTH_PX = 1940
SMILES_MIN_BYTES = 500_000
ACTIVITY_MIN_WIDTH_PX = 1880
ACTIVITY_MAX_WIDTH_PX = 1930
ACTIVITY_MIN_BYTES = 150_000
ACTIVITY_MAX_BYTES = 499_999


def _classify_size(width: int, height: int, size_bytes: int) -> str:
    """Return activity/smiles/structure/other based on width+size+aspect.

    Priority:
      1. SMILES  (width >= SMILES_MIN_WIDTH_PX AND size >= SMILES_MIN_BYTES)
      2. activity (ACTIVITY_* width+size band AND aspect within bounds)
      3. structure (small + sparse)
      4. other   (everything else: text pages, cover artwork, ...)
    """
    aspect = width / max(height, 1)
    in_table_aspect = TABLE_MIN_ASPECT <= aspect <= TABLE_MAX_ASPECT

    if width >= SMILES_MIN_WIDTH_PX and size_bytes >= SMILES_MIN_BYTES:
        return TableType.SMILES
    if (
        ACTIVITY_MIN_WIDTH_PX <= width < ACTIVITY_MAX_WIDTH_PX
        and ACTIVITY_MIN_BYTES <= size_bytes < ACTIVITY_MAX_BYTES
        and in_table_aspect
    ):
        return TableType.ACTIVITY
    if size_bytes <= STRUCTURE_MAX_BYTES and in_table_aspect:
        return TableType.STRUCTURE
    return TableType.OTHER


def scan_directory(
    img_dir: Path,
    *,
    detector: TableKeywordDetector | None = None,
    recursive: bool = False,
    pattern: str = "*.jpg",
) -> list[TableImage]:
    """Walk ``img_dir``, classify every ``<pattern>`` JPG.

    Parameters
    ----------
    img_dir
        Directory of FullText JPGs.
    detector
        Optional keyword detector for finer activity/smiles disambiguation.
        ``None`` uses :class:`NullKeywordDetector` (size+width+range only).
    recursive
        Whether to descend into subdirectories.
    """
    if detector is None:
        detector = NullKeywordDetector()

    out: list[TableImage] = []
    it = img_dir.rglob(pattern) if recursive else img_dir.glob(pattern)
    for f in sorted(it):
        if not f.is_file():
            continue
        m = _NUM_RE.search(f.name)
        if not m:
            continue  # skip filenames that don't match PCTCN...-I######.jpg
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            # Pillow failed to open -- still record as unknown.
            w, h = 0, 0
        sz = f.stat().st_size
        aspect = w / max(h, 1)
        size_bucket = _classify_size(w, h, sz)

        with Image.open(f) as im_for_ocr:
            try:
                keywords = list(detector.detect(im_for_ocr))
            except Exception:
                keywords = []

        # Refine table_type using keywords (if any).
        table_type = size_bucket
        is_table_candidate = size_bucket in (
            TableType.ACTIVITY, TableType.SMILES, TableType.OTHER,
        ) and sz >= ACTIVITY_MIN_BYTES
        confidence = 0.6 if is_table_candidate else 0.3
        if "activity" in keywords and "smiles" in keywords:
            table_type = TableType.OTHER  # ambiguous
            confidence = 0.2
        elif "activity" in keywords:
            table_type = TableType.ACTIVITY
            confidence = 0.95
            is_table_candidate = True
        elif "smiles" in keywords:
            table_type = TableType.SMILES
            confidence = 0.95
            is_table_candidate = True

        out.append(
            TableImage(
                image_path=f.resolve(),
                filename=f.name,
                numeric_id=int(m.group(1)),
                width=w,
                height=h,
                size_bytes=sz,
                aspect=round(aspect, 3),
                table_type=table_type,
                confidence=round(confidence, 2),
                detected_keywords=keywords,
                is_table_candidate=is_table_candidate,
            )
        )
    return out


def find_tables(
    images: list[TableImage], *,
    gap_tolerance: int = 2,
    table_types: Iterable[str] = (TableType.ACTIVITY, TableType.SMILES),
) -> list[TableBlock]:
    """Group ``images`` into contiguous blocks of the requested table types.

    A block is broken when the gap between consecutive numeric_ids exceeds
    ``gap_tolerance`` (i.e. an unrelated image slipped into the series).

    Images whose ``table_type`` isn't in ``table_types`` are skipped.
    """
    wanted = set(table_types)
    by_type: dict[str, list[TableImage]] = defaultdict(list)
    for img in images:
        if img.table_type in wanted:
            by_type[img.table_type].append(img)
    for k in by_type:
        by_type[k].sort(key=lambda i: i.numeric_id)

    blocks: list[TableBlock] = []
    for ttype, lst in by_type.items():
        if not lst:
            continue
        start = lst[0]
        prev = start
        bucket = [start]
        for cur in lst[1:]:
            if cur.numeric_id - prev.numeric_id > gap_tolerance:
                blocks.append(_close_block(ttype, bucket))
                bucket = [cur]
            else:
                bucket.append(cur)
            prev = cur
        blocks.append(_close_block(ttype, bucket))
    return sorted(blocks, key=lambda b: (b.table_type, b.start_id))


def _close_block(table_type: str, lst: list[TableImage]) -> TableBlock:
    kws: list[str] = []
    for img in lst:
        kws.extend(img.detected_keywords)
    dominant = sorted({k for k in kws if k in ("activity", "smiles")})
    return TableBlock(
        table_type=table_type,
        start_id=lst[0].numeric_id,
        end_id=lst[-1].numeric_id,
        images=tuple(lst),
        dominant_keywords=tuple(dominant),
    )


def find_by_type(
    images: list[TableImage], *,
    table_type: str,
) -> list[TableImage]:
    """Convenience filter: all TableImages of a given type."""
    return [i for i in images if i.table_type == table_type]


def write_findings_json(
    images: list[TableImage],
    blocks: list[TableBlock],
    out_path: Path,
    *,
    patent_id: str,
    source_dir: Path,
) -> None:
    """Persist the scan + clustering for human review and downstream loaders."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    type_counts: dict[str, int] = defaultdict(int)
    for img in images:
        type_counts[img.table_type] += 1
    payload = {
        "patent_id": patent_id,
        "source_dir": str(source_dir),
        "n_images": len(images),
        "type_counts": dict(type_counts),
        "blocks": [b.to_dict() for b in blocks],
        "images": [img.to_dict() for img in images],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_findings_json(path: Path) -> tuple[list[TableImage], list[TableBlock]]:
    """Inverse of :func:`write_findings_json` (no PatentId context)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    images = [TableImage.from_dict(d) for d in data["images"]]
    blocks: list[TableBlock] = []
    for b in data["blocks"]:
        imgs = [i for i in images if i.filename in b["filenames"]]
        blocks.append(TableBlock(
            table_type=b["table_type"],
            start_id=int(b["start_id"]),
            end_id=int(b["end_id"]),
            images=tuple(imgs),
            dominant_keywords=tuple(b.get("dominant_keywords") or []),
        ))
    return images, blocks


__all__ = [
    "TableType",
    "TableImage",
    "TableBlock",
    "TableKeywordDetector",
    "NullKeywordDetector",
    "HeaderKeywordDetector",
    "scan_directory",
    "find_tables",
    "find_by_type",
    "write_findings_json",
    "load_findings_json",
    "STRUCTURE_MAX_BYTES",
    "ACTIVITY_MIN_BYTES",
    "ACTIVITY_MAX_BYTES",
    "SMILES_MIN_BYTES",
    "SMILES_MIN_WIDTH_PX",
    "ACTIVITY_MIN_WIDTH_PX",
    "ACTIVITY_MAX_WIDTH_PX",
]
