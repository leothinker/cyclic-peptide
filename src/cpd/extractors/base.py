"""Patent extractor ABC + shared page dataclass.

Kept dependency-free: concrete backends (e.g. PyMuPdfExtractor) live in their
own modules and import from here. PageContent lives here (not in pdf.py) to
avoid the circular import pdf -> base -> pdf.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageContent:
    """One page of an extracted patent: text + saved embedded image paths."""

    page: int
    text: str
    images: list[Path] = field(default_factory=list)


class PatentExtractor(ABC):
    """Abstract PDF / document extractor."""

    @abstractmethod
    def extract_pages(self, pdf_path: Path) -> Iterator[PageContent]:
        """Yield per-page text + saved image paths."""
