"""Crop isolated chemical-structure figures from patent PDF pages.

Why this exists
---------------
`PyMuPdfExtractor` dumps every embedded raster XRef per page, which for a
ChemDraw-produced patent like WO2025162428 yields ~270 full-page rasters
that mix cover artwork, barcodes, text, and structure drawings. Downstream
OCSR (e.g. DECIMER) needs isolated single-structure crops to produce
useful SMILES.

Strategy
--------
1. Render each requested page at high DPI.
2. Collect vector stroke rectangles from `page.get_drawings()` -- ChemDraw
   structures are composed of many short line/curve segments, each with a
   tiny bbox.
3. Cluster nearby rectangles via union-find (Chebyshev gap, distance
   scales with page size), turning hundreds of segments into a handful of
   figure candidates.
4. If the page has no vector content, fall back to the bboxes of embedded
   raster XRefs from `page.get_image_info()`.
5. Drop noise:
     * Edge-margin frames  -> WIPO logo, barcodes, page numbers
     * Barcode-shaped slivers (extreme aspect ratio + small area)
     * Degenerate short rectangles (text underlines / horizontal rules)
     * Full-page rectangles (full-page tables)
     * Bboxes containing dense text rows (analytical tables)
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CropRegion:
    """One isolated structure figure cropped out of a page."""

    page: int
    index: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in page-pixel coords
    path: Path
    source: str  # "vector" | "raster_fallback"


@dataclass(frozen=True)
class _Box:
    """Internal bounding box in page-pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def expanded(self, pad: float) -> "_Box":
        return _Box(self.x0 - pad, self.y0 - pad,
                    self.x1 + pad, self.y1 + pad)

    def clipped(self, page_w: float, page_h: float) -> "_Box":
        return _Box(
            max(0.0, self.x0), max(0.0, self.y0),
            min(page_w, self.x1), min(page_h, self.y1),
        )

    def union(self, other: "_Box") -> "_Box":
        return _Box(min(self.x0, other.x0), min(self.y0, other.y0),
                    max(self.x1, other.x1), max(self.y1, other.y1))


class StructureCropper:
    """Crop isolated chemical-structure figures from patent PDF pages."""

    DEFAULT_DPI = 200
    DEFAULT_PADDING_PX = 12
    MIN_OUTPUT_BYTES = 5_000

    # ----- filtering thresholds (tune by subclassing if needed) -----
    EDGE_MARGIN_FRAC = 0.05       # drop bboxes within 5% of any page edge
    FULL_PAGE_AREA_FRAC = 0.85    # drop bboxes covering >85% of the page
    MIN_BOX_HEIGHT_PX = 12        # drop short bboxes (text underline / rule)
    BARCODE_AREA_FRAC = 0.05      # small + wide bboxes = barcodes
    BARCODE_ASPECT = 5.0
    MERGE_FRAC = 0.015            # merge distance as fraction of min page dim
    MERGE_MIN_PX = 16             # absolute floor for the merge distance
    # Text-density signals for "this is a table, not a structure":
    TABLE_MIN_TEXT_CHARS = 350
    TABLE_MIN_TEXT_LINES = 8      # distinct text-row buckets (int(y)//10)

    def __init__(
        self,
        out_dir: Path,
        *,
        patent_id: str,
        dpi: int = DEFAULT_DPI,
        padding_px: int = DEFAULT_PADDING_PX,
        min_output_bytes: int = MIN_OUTPUT_BYTES,
    ) -> None:
        self.out_dir = out_dir / patent_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.patent_id = patent_id
        self.dpi = dpi
        self.padding_px = padding_px
        self.min_output_bytes = min_output_bytes

    # ---------- public API ----------

    def crop_pages(
        self, pdf_path: Path, pages: list[int] | None = None,
    ) -> Iterator[CropRegion]:
        """Yield CropRegion per surviving structure figure.

        `pages` is 1-indexed; None means every page in the PDF.
        """
        import pymupdf  # lazy: this dep is in the [pdf] extra
        doc = pymupdf.open(pdf_path)
        try:
            page_iter: list[int] = (
                list(range(1, len(doc) + 1))
                if pages is None
                else [p for p in pages if 1 <= p <= len(doc)]
            )
            scale = self.dpi / 72.0
            for page_idx in page_iter:
                yield from self._crop_one_page(doc[page_idx - 1], page_idx, scale)
        finally:
            doc.close()

    # ---------- internals ----------

    def _crop_one_page(
        self, page, page_idx: int, scale: float,
    ) -> Iterator[CropRegion]:
        import pymupdf
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
        page_w, page_h = pix.width, pix.height

        boxes = self._boxes_from_drawings(page, scale)
        source = "vector"
        if not boxes:
            boxes = self._boxes_from_image_info(page, scale)
            source = "raster_fallback"
        if not boxes:
            return

        merged = self._cluster(boxes, page_w, page_h)
        kept = self._filter(merged, page_w, page_h, page, scale)
        if not kept:
            return

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        for idx, box in enumerate(kept):
            padded = box.expanded(self.padding_px).clipped(page_w, page_h)
            x0, y0 = int(round(padded.x0)), int(round(padded.y0))
            x1, y1 = int(round(padded.x1)), int(round(padded.y1))
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue  # too tiny after padding-clipping
            crop_img = img.crop((x0, y0, x1, y1))
            out_path = self.out_dir / f"p{page_idx:04d}_s{idx:02d}.png"
            crop_img.save(out_path, format="PNG", optimize=False)
            if out_path.stat().st_size < self.min_output_bytes:
                out_path.unlink(missing_ok=True)
                continue
            yield CropRegion(
                page=page_idx, index=idx, bbox=(x0, y0, x1, y1),
                path=out_path, source=source,
            )

    def _boxes_from_drawings(self, page, scale: float) -> list[_Box]:
        """Collect stroke/curve rectangles from vector drawings."""
        boxes: list[_Box] = []
        for d in page.get_drawings():
            rect = d.get("rect")
            if rect is None:
                continue
            if rect.width <= 0 or rect.height <= 0:
                continue
            boxes.append(_Box(
                rect.x0 * scale, rect.y0 * scale,
                rect.x1 * scale, rect.y1 * scale,
            ))
        return boxes

    def _boxes_from_image_info(self, page, scale: float) -> list[_Box]:
        """Fallback: bounding box of each embedded raster XRef."""
        boxes: list[_Box] = []
        for info in page.get_image_info(xrefs=True):
            bbox = info.get("bbox")
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append(_Box(x0 * scale, y0 * scale, x1 * scale, y1 * scale))
        return boxes

    def _cluster(
        self, boxes: list[_Box], page_w: float, page_h: float,
    ) -> list[_Box]:
        """Union-find merge of nearby rectangles (Chebyshev gap)."""
        if not boxes:
            return []
        merge_dist = max(self.MERGE_MIN_PX, self.MERGE_FRAC * min(page_w, page_h))
        parent = list(range(len(boxes)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                if self._gap(boxes[i], boxes[j]) <= merge_dist:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        groups: dict[int, _Box] = {}
        for i, b in enumerate(boxes):
            r = find(i)
            groups[r] = groups[r].union(b) if r in groups else b
        return list(groups.values())

    @staticmethod
    def _gap(a: _Box, b: _Box) -> float:
        """Chebyshev gap: max horizontal/vertical separation (0 if touching)."""
        dx = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
        dy = max(0.0, max(a.y0 - b.y1, b.y0 - a.y1))
        return max(dx, dy)

    def _filter(
        self, boxes: list[_Box], page_w: float, page_h: float,
        page, scale: float,
    ) -> list[_Box]:
        margin_x = page_w * self.EDGE_MARGIN_FRAC
        margin_y = page_h * self.EDGE_MARGIN_FRAC
        page_area = page_w * page_h
        text_blocks = page.get_text("blocks")

        kept: list[_Box] = []
        for b in boxes:
            if (b.x0 < margin_x or b.x1 > page_w - margin_x or
                    b.y0 < margin_y or b.y1 > page_h - margin_y):
                continue
            if b.height < self.MIN_BOX_HEIGHT_PX:
                continue
            aspect = b.width / max(b.height, 1.0)
            if aspect > self.BARCODE_ASPECT and b.area < page_area * self.BARCODE_AREA_FRAC:
                continue
            if b.area > page_area * self.FULL_PAGE_AREA_FRAC:
                continue
            if self._looks_like_table(b, text_blocks, scale):
                continue
            kept.append(b)
        kept.sort(key=lambda b: (round(b.y0, 1), b.x0))
        return kept

    def _looks_like_table(self, box: _Box, text_blocks, scale: float) -> bool:
        """A dense text grid inside the bbox signals a table, not a structure."""
        chars = 0
        rows: set[int] = set()
        for blk in text_blocks:
            x0, y0, x1, y1, text, *_ = blk
            x0 *= scale
            y0 *= scale
            x1 *= scale
            y1 *= scale
            if x1 < box.x0 or x0 > box.x1 or y1 < box.y0 or y0 > box.y1:
                continue
            chars += len(text.strip())
            rows.add(int(y0) // 10)
        return (chars >= self.TABLE_MIN_TEXT_CHARS and
                len(rows) >= self.TABLE_MIN_TEXT_LINES)


# ---------- CLI helper ----------

def parse_pages(spec: str | None, total: int) -> list[int]:
    """Parse ``'1-5'`` / ``'1,2,93'`` / ``'all'`` / None into 1-indexed pages.

    Out-of-range and unparseable tokens are silently dropped; duplicates are
    removed. Reversed ranges like ``'10-1'`` are accepted.
    """
    if spec is None or spec.strip().lower() in ("", "all"):
        return list(range(1, total + 1))
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip().replace(" ", "")
        if not token:
            continue
        if "-" in token:
            try:
                a, b = token.split("-", 1)
                start, end = int(a), int(b)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            try:
                out.add(int(token))
            except ValueError:
                continue
    return sorted(p for p in out if 1 <= p <= total)


__all__ = ["CropRegion", "StructureCropper", "parse_pages"]
