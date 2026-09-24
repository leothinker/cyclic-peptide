"""Chemical structure OCR interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StructureReader(ABC):
    """Image -> SMILES interface."""

    @abstractmethod
    def read(self, image_path: Path | str) -> str | None:
        """Return a SMILES string for a chemical structure figure, or None on failure."""
