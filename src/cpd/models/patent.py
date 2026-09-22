"""Patent document schema shared across scrapers."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class PatentDocument(BaseModel):
    """A normalized patent record. Scraper backends map their HTML/JSON to this."""

    patent_id: str = Field(description="Publication number, e.g. 'WO2025162428'")
    title: str = ""
    abstract: str = ""
    applicants: list[str] = Field(default_factory=list)
    publication_date: date | None = None
    source_url: HttpUrl
    pdf_url: HttpUrl | None = None
    local_html: Path | None = None
    local_pdf: Path | None = None

    model_config = {"arbitrary_types_allowed": True}
