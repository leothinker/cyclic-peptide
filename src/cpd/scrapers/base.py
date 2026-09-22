"""Patent scraper ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod

from cpd.models.patent import PatentDocument


class PatentFetcher(ABC):
    """Backend-agnostic patent fetcher."""

    @abstractmethod
    def fetch(self, patent_id: str, *, save_html: bool = True) -> PatentDocument:
        """Download and parse a patent's public page; return a normalized doc."""
