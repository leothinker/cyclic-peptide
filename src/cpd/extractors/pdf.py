"""PyMuPDF-based extractor: per-page text + dumped images.

Heavy dep lives in the `pdf` extra so the core install stays slim. PageContent
is defined in `cpd.extractors.base` and imported here to avoid a circular
import between base and pdf.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from cpd.extractors.base import PageContent, PatentExtractor


class PyMuPdfExtractor(PatentExtractor):
    """Split a patent PDF into per-page text + saved embedded images."""

    def __init__(self, image_dir: Path) -> None:
        self._image_dir = image_dir
        self._image_dir.mkdir(parents=True, exist_ok=True)

    def extract_pages(self, pdf_path: Path) -> Iterator[PageContent]:
        import pymupdf  # local import so the dep is only needed when used

        doc = pymupdf.open(pdf_path)
        try:
            for idx, page in enumerate(doc, start=1):
                images = self._dump_images(doc, page, idx)
                yield PageContent(page=idx, text=page.get_text(), images=images)
        finally:
            doc.close()

    def _dump_images(self, doc: "object", page: "object", page_idx: int) -> list[Path]:
        import pymupdf

        paths: list[Path] = []
        for img_idx, info in enumerate(page.get_images(full=True)):
            xref = info[0]
            try:
                pix = pymupdf.Pixmap(doc, xref)
            except Exception:
                continue  # skip broken XRefs instead of failing the whole page
            if pix.n - pix.alpha >= 4:  # CMYK or other -> RGB
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            out = self._image_dir / f"p{page_idx:04d}_i{img_idx:02d}.png"
            pix.save(out)
            paths.append(out)
            pix = None
        return paths